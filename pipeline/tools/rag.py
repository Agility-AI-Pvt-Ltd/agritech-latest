"""
pipeline/tools/rag.py  –  RAG search tool using Qdrant + multi-collection strategy.

Generates 4 targeted sub-queries (POP / Fertilizer / Pest / Production)
and queries each Qdrant collection independently for higher recall.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from core.config import settings
from pipeline.logging_utils import log_llm_call


# ──────────────────────────────────────────────────────────────────────────────
# Sub-query generation (translates + expands farmer query → 4 DB-specific queries)
# ──────────────────────────────────────────────────────────────────────────────

_SUB_QUERY_SYSTEM = """You are an agricultural assistant. Given a user query about farming, and optionally recent conversation context, generate 4 specific sub-questions to search our vector database.
The databases contain English documents. Therefore, YOU MUST TRANSLATE THE QUERIES INTO ENGLISH before returning them, even if the user query is in Hindi or another language.

The databases are:
1. pop_query: Deals with Package of Practices (POP) like sowing, harvesting, seeds, spacing, yield, and water management.
2. fertilizer_query: Deals with fertilizers, nutrient computation, soil health, and application methods.
3. pest_query: Deals with pests, diseases, weeds, and their management strategies.
4. production_query: Deals with overall maize production, cultivation guidelines, crop management, and agronomy.

Return ONLY a valid JSON object with these exactly 4 keys ("pop_query", "fertilizer_query", "pest_query", "production_query"), mapping to the strictly ENGLISH generated questions. Do not include markdown formatting or extra text."""


def generate_sub_queries(
    query: str,
    chat_history: list | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Ask the LLM to generate 4 targeted sub-queries for each Qdrant collection."""
    from pipeline.llm_factory import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    try:
        llm = get_llm()
        context_str = ""
        if chat_history:
            context_str = "Recent Chat History:\n" + "\n".join(
                [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history]
            ) + "\n\n"

        messages = [
            SystemMessage(content=_SUB_QUERY_SYSTEM),
            HumanMessage(content=f"{context_str}Current Query to expand: {query}"),
        ]

        msg = llm.invoke(messages)
        log_llm_call(
            conversation_id=conversation_id,
            user_id=user_id,
            source="tool.generate_sub_queries",
            request={"query": query, "chat_history_count": len(chat_history or [])},
            response={"has_content": bool(getattr(msg, "content", ""))},
        )

        content = msg.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        return json.loads(content)

    except Exception as exc:
        log_llm_call(
            conversation_id=conversation_id,
            user_id=user_id,
            source="tool.generate_sub_queries.error",
            request={"query": query, "chat_history_count": len(chat_history or [])},
            error=str(exc),
        )
        # Fallback: use original query for all collections
        return {
            "pop_query":        query,
            "fertilizer_query": query,
            "pest_query":       query,
            "production_query": query,
        }


# ──────────────────────────────────────────────────────────────────────────────
# RAG search
# ──────────────────────────────────────────────────────────────────────────────

_COLLECTION_MAP = {
    "pop_query":        "spring_corn_pop_db",
    "fertilizer_query": "spring_corn_fertilizers_db",
    "pest_query":       "spring_corn_pest_and_diseases_db",
    "production_query": "maize_production_manual_db",
}


def _extract_hits(result: Any) -> List[Any]:
    """Handle qdrant-client response shape differences across versions."""
    if result is None:
        return []
    if isinstance(result, list):
        return result

    points = getattr(result, "points", None)
    if isinstance(points, list):
        return points

    inner_result = getattr(result, "result", None)
    if isinstance(inner_result, list):
        return inner_result

    inner_points = getattr(inner_result, "points", None)
    if isinstance(inner_points, list):
        return inner_points

    return []


def _search_collection(
    qdrant_client,
    *,
    collection_name: str,
    sub_query: str,
    query_vector: List[float],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Execute a single collection search and normalize the hits."""
    chunks: List[Dict[str, Any]] = []
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )
    for hit in _extract_hits(response):
        payload = getattr(hit, "payload", None) or {}
        chunks.append(
            {
                "collection": collection_name,
                "sub_query": sub_query,
                "score": round(float(hit.score), 4),
                "content": payload.get("page_content", ""),
                "metadata": payload.get("metadata", {}),
            }
        )
    return chunks


def execute_rag_search(
    query: str,
    top_k: int = 2,
    crop_stage: str | None = None,
    qdrant_client=None,
    chat_history: list | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> Dict[str, Any]:
    """Run similarity search across all 4 Qdrant collections using sub-query expansion."""
    if qdrant_client is None:
        return {"error": "Qdrant client not initialized", "chunks": []}

    try:
        from pipeline.llm_factory import get_embedding_model
        encoder = get_embedding_model()

        sub_queries = generate_sub_queries(
            query,
            chat_history,
            conversation_id=conversation_id,
            user_id=user_id,
        )

        search_plan = [
            (key, _COLLECTION_MAP[key], sub_queries.get(key, query))
            for key in _COLLECTION_MAP
        ]
        query_texts = [sub_query for _, _, sub_query in search_plan]
        encoded_vectors = encoder.encode(query_texts, normalize_embeddings=True)
        query_vectors = [vector.tolist() for vector in encoded_vectors]

        chunks: List[Dict[str, Any]] = []
        max_workers = min(len(search_plan), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _search_collection,
                    qdrant_client,
                    collection_name=coll_name,
                    sub_query=sub_query,
                    query_vector=query_vectors[idx],
                    top_k=top_k,
                ): coll_name
                for idx, (_, coll_name, sub_query) in enumerate(search_plan)
            }

            for future in as_completed(future_map):
                coll_name = future_map[future]
                try:
                    chunks.extend(future.result())
                except Exception as coll_e:
                    print(f"[!] RAG Search Warning ({coll_name}): {coll_e}")

        if crop_stage:
            try:
                from pipeline.tools.maize_faq import execute_faq_search_by_crop_stage

                faq_result = execute_faq_search_by_crop_stage(
                    query=query,
                    crop_stage=crop_stage,
                    top_k=top_k,
                    qdrant_client=qdrant_client,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                for entry in faq_result.get("entries", []):
                    chunks.append(
                        {
                            "collection": settings.maize_faq_collection_name,
                            "sub_query": query,
                            "score": entry.get("score", 1.0 if entry.get("lookup_mode") == "direct_lookup" else 0.0),
                            "content": entry.get("recommendation", ""),
                            "metadata": {
                                "crop_stage": entry.get("crop_stage"),
                                "subtopic": entry.get("subtopic"),
                                "question": entry.get("question"),
                                "category": entry.get("category"),
                                "lookup_mode": entry.get("lookup_mode"),
                            },
                        }
                    )
            except Exception as faq_exc:
                print(f"[!] FAQ Search Warning: {faq_exc}")

        return {"query": query, "sub_queries": sub_queries, "chunks": chunks}

    except Exception as e:
        return {"error": str(e), "chunks": []}
