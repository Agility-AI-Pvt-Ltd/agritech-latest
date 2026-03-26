"""
pipeline/tools/rag.py  –  RAG search tool using Qdrant + multi-collection strategy.

Generates 4 targeted sub-queries (POP / Fertilizer / Pest / Production)
and queries each Qdrant collection independently for higher recall.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

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


def execute_rag_search(
    query: str,
    top_k: int = 2,
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

        chunks: List[Dict[str, Any]] = []
        for key, coll_name in _COLLECTION_MAP.items():
            sub_q        = sub_queries.get(key, query)
            query_vector = encoder.encode(sub_q).tolist()

            try:
                response = qdrant_client.query_points(
                    collection_name=coll_name,
                    query=query_vector,
                    limit=top_k,
                )
                print(response)
                for hit in response.points:
                    payload = hit.payload or {}
                    chunks.append({
                        "collection": coll_name,
                        "sub_query":  sub_q,
                        "score":      round(float(hit.score), 4),
                        "content":    payload.get("page_content", ""),
                        "metadata":   payload.get("metadata", {}),
                    })
            except Exception as coll_e:
                print(f"[!] RAG Search Warning ({coll_name}): {coll_e}")

        return {"query": query, "sub_queries": sub_queries, "chunks": chunks}

    except Exception as e:
        return {"error": str(e), "chunks": []}
