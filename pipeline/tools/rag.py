"""
pipeline/tools/rag.py  –  RAG search tool using Qdrant + multi-collection strategy.

Generates 4 targeted sub-queries (POP / Fertilizer / Pest / Production)
and queries each Qdrant collection independently for higher recall.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List

from core.config import settings
from pipeline.logging_utils import append_user_event_log, log_llm_call


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

Important retrieval alignment rules:
- Prefer vocabulary likely to appear in agronomy manuals and markdown headings.
- Expand farmer wording into manual-style keywords and synonyms.
- When relevant, include terms like: maize, corn, sweet corn, zaid maize, package of practices, POP, land preparation, field preparation, sowing, seed rate, spacing, irrigation, weed management, nutrient management, FYM, farmyard manure, basal dose, top dressing, urea, DAP, NPK, pest management, disease symptoms, disease control, crop protection, harvesting.
- For pest/disease questions, mention symptoms, diagnosis, control, and recommended management.
- For fertilizer questions, mention nutrient management, fertilizer recommendation, dose, application method, and FYM/farmyard manure when relevant.

Return ONLY a valid JSON object with these exactly 4 keys ("pop_query", "fertilizer_query", "pest_query", "production_query"), mapping to the strictly ENGLISH generated questions. Do not include markdown formatting or extra text."""


_DOMAIN_HINTS = {
    "pop_query": [
        "maize package of practices",
        "POP",
        "land preparation",
        "field preparation",
        "sowing",
        "seed rate",
        "spacing",
        "irrigation",
        "weed management",
    ],
    "fertilizer_query": [
        "maize nutrient management",
        "fertilizer recommendation",
        "fertilizer dose",
        "application method",
        "FYM",
        "farmyard manure",
        "basal dose",
        "top dressing",
        "urea DAP NPK",
    ],
    "pest_query": [
        "maize pest and disease management",
        "disease symptoms",
        "diagnosis",
        "control",
        "crop protection",
        "fungicide",
        "insecticide",
        "weed control",
    ],
    "production_query": [
        "maize production manual",
        "maize cultivation",
        "agronomy",
        "crop management",
        "land preparation",
        "soil requirements",
        "planting",
        "harvesting",
    ],
}


def _keyword_augmented_query(query: str, key: str) -> str:
    base = (query or "").strip()
    hints = ", ".join(_DOMAIN_HINTS.get(key, []))
    if not base:
        return hints
    return f"{base}. Focus on: {hints}."


def _normalize_sub_queries(candidate: dict, original_query: str) -> dict:
    normalized: dict[str, str] = {}
    for key in _COLLECTION_MAP:
        proposed = candidate.get(key)
        if isinstance(proposed, str) and proposed.strip():
            normalized[key] = proposed.strip()
        else:
            normalized[key] = _keyword_augmented_query(original_query, key)
    return normalized


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

        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Sub-query generator did not return a JSON object")
        return _normalize_sub_queries(parsed, query)

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
            key: _keyword_augmented_query(query, key)
            for key in _COLLECTION_MAP
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return str(value)


def _safe_log_token(value: str | None, default: str) -> str:
    raw = (value or "").strip() or default
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    return safe[:120] or default


def _write_rag_call_log(
    *,
    conversation_id: str | None,
    user_id: str | None,
    query: str,
    crop_stage: str | None,
    top_k: int,
    sub_queries: Dict[str, Any] | None = None,
    chunks: List[Dict[str, Any]] | None = None,
    faq_result: Dict[str, Any] | None = None,
    timings_ms: Dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        logs_dir = os.path.join(project_root, "logs", "rag_calls")
        os.makedirs(logs_dir, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        suffix = uuid.uuid4().hex[:8]
        conv = _safe_log_token(conversation_id, "unknown_conv")
        user = _safe_log_token(user_id, "unknown_user")
        file_name = f"{ts}_{user}_{conv}_{suffix}.json"
        file_path = os.path.join(logs_dir, file_name)

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "query": query,
            "crop_stage": crop_stage,
            "top_k": top_k,
            "sub_queries": _json_safe(sub_queries or {}),
            "retrieved_chunks": _json_safe(chunks or []),
            "faq_result": _json_safe(faq_result),
            "timings_ms": _json_safe(timings_ms or {}),
            "error": error,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as log_exc:
        print(f"[!] Failed to write rag call log: {log_exc}")


def _search_collection(
    qdrant_client,
    *,
    collection_name: str,
    sub_query: str,
    query_vector: List[float],
    top_k: int,
) -> Dict[str, Any]:
    """Execute a single collection search and normalize the hits."""
    start = time.perf_counter()
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
    return {
        "chunks": chunks,
        "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 2),
    }


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

    rag_start = time.perf_counter()
    try:
        from pipeline.llm_factory import get_embedding_model
        encoder = get_embedding_model()

        sub_query_start = time.perf_counter()
        sub_queries = generate_sub_queries(
            query,
            chat_history,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        sub_query_ms = (time.perf_counter() - sub_query_start) * 1000.0

        search_plan = [
            (key, _COLLECTION_MAP[key], sub_queries.get(key, query))
            for key in _COLLECTION_MAP
        ]
        query_texts = [sub_query for _, _, sub_query in search_plan]
        embedding_start = time.perf_counter()
        encoded_vectors = encoder.encode(query_texts, normalize_embeddings=True)
        query_vectors = [vector.tolist() for vector in encoded_vectors]
        embedding_ms = (time.perf_counter() - embedding_start) * 1000.0

        chunks: List[Dict[str, Any]] = []
        retrieval_start = time.perf_counter()
        collection_timings_ms: Dict[str, float] = {}
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
                    future_result = future.result()
                    chunks.extend(future_result.get("chunks", []))
                    collection_timings_ms[coll_name] = future_result.get("elapsed_ms", 0.0)
                except Exception as coll_e:
                    print(f"[!] RAG Search Warning ({coll_name}): {coll_e}")
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        faq_result: Dict[str, Any] | None = None
        faq_ms = 0.0
        if crop_stage:
            try:
                from pipeline.tools.maize_faq import execute_faq_search_by_crop_stage

                faq_start = time.perf_counter()
                faq_result = execute_faq_search_by_crop_stage(
                    query=query,
                    crop_stage=crop_stage,
                    top_k=top_k,
                    qdrant_client=qdrant_client,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                faq_ms = (time.perf_counter() - faq_start) * 1000.0
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
                faq_result = {"error": str(faq_exc), "entries": []}

        total_ms = (time.perf_counter() - rag_start) * 1000.0
        timings_payload = {
            "sub_query_generation": round(sub_query_ms, 2),
            "embedding": round(embedding_ms, 2),
            "retrieval": round(retrieval_ms, 2),
            "faq_in_rag": round(faq_ms, 2),
            "total": round(total_ms, 2),
            "per_collection": collection_timings_ms,
        }

        _write_rag_call_log(
            conversation_id=conversation_id,
            user_id=user_id,
            query=query,
            crop_stage=crop_stage,
            top_k=top_k,
            sub_queries=sub_queries,
            chunks=chunks,
            faq_result=faq_result,
            timings_ms=timings_payload,
        )

        append_user_event_log(
            user_id=user_id,
            event_type="rag_retrieval",
            payload={
                "conversation_id": conversation_id,
                "query": query,
                "crop_stage": crop_stage,
                "top_k": top_k,
                "timings_ms": timings_payload,
                "sub_queries": sub_queries,
                "retrieved_documents": chunks,
                "faq_result": faq_result,
            },
        )

        return {"query": query, "sub_queries": sub_queries, "chunks": chunks}

    except Exception as e:
        error_timings = {
            "total": round((time.perf_counter() - rag_start) * 1000.0, 2),
        }
        _write_rag_call_log(
            conversation_id=conversation_id,
            user_id=user_id,
            query=query,
            crop_stage=crop_stage,
            top_k=top_k,
            chunks=[],
            timings_ms=error_timings,
            error=str(e),
        )

        append_user_event_log(
            user_id=user_id,
            event_type="rag_retrieval",
            payload={
                "conversation_id": conversation_id,
                "query": query,
                "crop_stage": crop_stage,
                "top_k": top_k,
                "error": str(e),
                "timings_ms": error_timings,
            },
        )
        return {"error": str(e), "chunks": []}
