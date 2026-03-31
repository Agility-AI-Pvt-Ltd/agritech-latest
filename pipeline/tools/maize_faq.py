"""
pipeline/tools/maize_faq.py  –  Maize FAQ crop-stage tooling.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from core.config import settings
from pipeline.logging_utils import append_user_event_log, log_llm_call


def _normalize_question(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower())
    return " ".join(cleaned.split())


@lru_cache(maxsize=1)
def _load_maize_tree() -> Dict[str, Any]:
    with open(settings.maize_faq_tree_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_stage_label(stage_label: str) -> Tuple[int, int | None]:
    digits = [int(num) for num in re.findall(r"\d+", stage_label or "")]
    if not digits:
        return 0, None
    if len(digits) == 1:
        return digits[0], digits[0]
    return digits[0], digits[1]


def _resolve_crop_stage(days_since_sowing: int) -> str:
    tree = _load_maize_tree()
    stage_labels = list(tree.get("metadata", {}).get("stages", []))
    if not stage_labels:
        raise ValueError("No crop stages configured in maize knowledge tree metadata")

    parsed = sorted((_parse_stage_label(label)[0], label) for label in stage_labels)
    resolved = parsed[0][1]
    for start_day, label in parsed:
        if days_since_sowing >= start_day:
            resolved = label
        else:
            break
    return resolved


def execute_set_crop_stage(sowing_date: str) -> Dict[str, Any]:
    """Resolve the current maize crop stage from a normalized sowing date."""
    try:
        sowing = datetime.strptime(sowing_date, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "Invalid sowing_date format. Expected YYYY-MM-DD."}

    today = datetime.now().date()
    days_since_sowing = (today - sowing).days
    if days_since_sowing < 0:
        return {
            "error": "Sowing date is in the future.",
            "sowing_date": sowing_date,
            "current_date": today.isoformat(),
        }

    crop_stage = _resolve_crop_stage(days_since_sowing)
    return {
        "sowing_date": sowing_date,
        "current_date": today.isoformat(),
        "days_since_sowing": days_since_sowing,
        "crop_stage": crop_stage,
        "source": "maize_knowledge_tree.metadata.stages",
    }


def _iter_stage_entries(crop_stage: str | None) -> List[Tuple[str, str, Dict[str, Any]]]:
    tree = _load_maize_tree()
    entries: List[Tuple[str, str, Dict[str, Any]]] = []
    knowledge_tree = tree.get("knowledge_tree", {})
    stages = [crop_stage] if crop_stage else list(knowledge_tree.keys())
    for stage_name in stages:
        stage_tree = knowledge_tree.get(stage_name) or {}
        for subtopic in stage_tree.get("subtopics", []):
            for entry in subtopic.get("entries", []):
                entries.append((stage_name, subtopic.get("subtopic"), entry))
    return entries


def _direct_lookup(crop_stage: str | None, query: str, english_query: str | None = None) -> Dict[str, Any] | None:
    normalized_candidates = [_normalize_question(query)]
    if english_query:
        normalized_candidates.append(_normalize_question(english_query))

    for stage_name, subtopic_name, entry in _iter_stage_entries(crop_stage):
        normalized_question = _normalize_question(entry.get("question", ""))
        if normalized_question in normalized_candidates:
            return {
                "lookup_mode": "direct_lookup",
                "crop_stage": stage_name,
                "subtopic": subtopic_name,
                "question": entry.get("question"),
                "category": entry.get("category"),
                "recommendation": entry.get("recommendation"),
            }
    return None


_FAQ_QUERY_TRANSLATION_SYSTEM = """You convert a farmer's maize FAQ question into concise English for vector search.
Return ONLY a valid JSON object with this shape:
{"english_query": "translated or normalized English question"}
Do not add markdown or extra text."""


def _to_english_query(
    query: str,
    crop_stage: str,
    *,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """Translate or normalize a farmer FAQ query into English before embedding."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from pipeline.llm_factory import get_llm

        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=_FAQ_QUERY_TRANSLATION_SYSTEM),
                HumanMessage(
                    content=(
                        f"Crop stage: {crop_stage}\n"
                        f"Original farmer question: {query}"
                    )
                ),
            ]
        )
        content = (getattr(response, "content", "") or "").strip()
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif content.startswith("```"):
            content = content.split("```", 1)[1].split("```", 1)[0].strip()
        parsed = json.loads(content)
        english_query = (parsed.get("english_query") or "").strip()

        log_llm_call(
            conversation_id=conversation_id,
            user_id=user_id,
            source="tool.faq_query_translation",
            request={"query": query, "crop_stage": crop_stage},
            response={"english_query": english_query},
        )

        return english_query or query
    except Exception as exc:
        log_llm_call(
            conversation_id=conversation_id,
            user_id=user_id,
            source="tool.faq_query_translation.error",
            request={"query": query, "crop_stage": crop_stage},
            error=str(exc),
        )
        return query


def execute_faq_search_by_crop_stage(
    query: str,
    crop_stage: str | None = None,
    top_k: int = 3,
    qdrant_client=None,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> Dict[str, Any]:
    """
    Search maize FAQ entries within the current crop stage.
    First try exact question match in the knowledge tree, then semantic search in Qdrant.
    """
    print(
        f"[FAQ] Search invoked | crop_stage={crop_stage} | top_k={top_k} | query={query}"
    )
    faq_start = time.perf_counter()
    translation_start = time.perf_counter()
    english_query = _to_english_query(
        query,
        crop_stage or "unknown",
        conversation_id=conversation_id,
        user_id=user_id,
    )
    translation_ms = (time.perf_counter() - translation_start) * 1000.0
    direct_lookup_start = time.perf_counter()
    direct_hit = _direct_lookup(crop_stage, query, english_query=english_query)
    direct_lookup_ms = (time.perf_counter() - direct_lookup_start) * 1000.0
    if direct_hit:
        print(
            f"[FAQ] Direct hit | stage={direct_hit.get('crop_stage')} | "
            f"subtopic={direct_hit.get('subtopic')} | question={direct_hit.get('question')}"
        )
        print(f"[FAQ] Recommendation: {direct_hit.get('recommendation', '')}")
        total_ms = (time.perf_counter() - faq_start) * 1000.0
        append_user_event_log(
            user_id=user_id,
            event_type="faq_retrieval",
            payload={
                "conversation_id": conversation_id,
                "query": query,
                "english_query": english_query,
                "crop_stage": crop_stage,
                "top_k": top_k,
                "hit": True,
                "lookup_mode": "direct_lookup",
                "retriever_time_ms": round(direct_lookup_ms, 2),
                "timings_ms": {
                    "translation": round(translation_ms, 2),
                    "direct_lookup": round(direct_lookup_ms, 2),
                    "total": round(total_ms, 2),
                },
                "entries": [direct_hit],
            },
        )
        return {
            "query": query,
            "english_query": english_query,
            "crop_stage": crop_stage,
            "lookup_mode": "direct_lookup",
            "entries": [direct_hit],
        }

    if qdrant_client is None:
        return {
            "error": "Qdrant client not initialized",
            "query": query,
            "crop_stage": crop_stage,
            "entries": [],
        }

    try:
        from qdrant_client.http import models as qdrant_models
        from pipeline.llm_factory import get_embedding_model

        encoder = get_embedding_model()
        semantic_start = time.perf_counter()
        query_vector = encoder.encode(english_query, normalize_embeddings=True).tolist()
        stage_filter = None
        if crop_stage:
            stage_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.crop_stage",
                        match=qdrant_models.MatchValue(value=crop_stage),
                    )
                ]
            )

        response = qdrant_client.query_points(
            collection_name=settings.maize_faq_collection_name,
            query=query_vector,
            query_filter=stage_filter,
            limit=top_k,
        )

        entries: List[Dict[str, Any]] = []
        for hit in response.points:
            payload = hit.payload or {}
            metadata = payload.get("metadata", {})
            entry = {
                "lookup_mode": "semantic_search",
                "score": round(float(hit.score), 4),
                "crop_stage": metadata.get("crop_stage"),
                "subtopic": metadata.get("subtopic"),
                "question": payload.get("question", ""),
                "category": payload.get("category", ""),
                "recommendation": payload.get("recommendation", ""),
                "search_text": payload.get("search_text", ""),
            }
            print(
                f"[FAQ] Semantic hit | score={entry['score']} | stage={entry.get('crop_stage')} | "
                f"subtopic={entry.get('subtopic')} | question={entry.get('question')}"
            )
            print(f"[FAQ] Recommendation: {entry.get('recommendation', '')}")
            entries.append(entry)

        if not entries:
            print("[FAQ] No FAQ entries matched in semantic search.")

        semantic_ms = (time.perf_counter() - semantic_start) * 1000.0
        total_ms = (time.perf_counter() - faq_start) * 1000.0
        append_user_event_log(
            user_id=user_id,
            event_type="faq_retrieval",
            payload={
                "conversation_id": conversation_id,
                "query": query,
                "english_query": english_query,
                "crop_stage": crop_stage,
                "top_k": top_k,
                "hit": bool(entries),
                "lookup_mode": "semantic_search" if entries else "no_hit",
                "retriever_time_ms": round(semantic_ms, 2),
                "timings_ms": {
                    "translation": round(translation_ms, 2),
                    "direct_lookup": round(direct_lookup_ms, 2),
                    "semantic_retrieval": round(semantic_ms, 2),
                    "total": round(total_ms, 2),
                },
                "entries": entries,
                "message": None if entries else "No FAQ entries matched.",
            },
        )

        return {
            "query": query,
            "english_query": english_query,
            "crop_stage": crop_stage,
            "lookup_mode": "semantic_search",
            "entries": entries,
        }
    except Exception as exc:
        append_user_event_log(
            user_id=user_id,
            event_type="faq_retrieval",
            payload={
                "conversation_id": conversation_id,
                "query": query,
                "english_query": english_query,
                "crop_stage": crop_stage,
                "top_k": top_k,
                "hit": False,
                "lookup_mode": "error",
                "timings_ms": {
                    "translation": round(translation_ms, 2),
                    "direct_lookup": round(direct_lookup_ms, 2),
                    "total": round((time.perf_counter() - faq_start) * 1000.0, 2),
                },
                "error": str(exc),
                "entries": [],
            },
        )
        return {
            "error": str(exc),
            "query": query,
            "english_query": None,
            "crop_stage": crop_stage,
            "entries": [],
        }
