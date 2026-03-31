"""
pipeline/logging_utils.py  –  LLM call and per-user event logging utilities.

Writes per-conversation .summary.json and .llm_calls.jsonl files
under <project_root>/logs/chat_sessions/, plus one append-only JSONL file
per user directly under <project_root>/logs/.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict


def _safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_safe(v) for v in value]
    return str(value)


def _base_logs_dir() -> str:
    # Walk up from this file to project root (pipeline/../)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(project_root, "logs", "chat_sessions")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _root_logs_dir() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _safe_user_filename(user_id: str | None) -> str:
    raw = (user_id or "unknown_user").strip() or "unknown_user"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return safe[:120] or "unknown_user"


def append_user_event_log(
    *,
    user_id: str | None,
    event_type: str,
    payload: Dict[str, Any] | None = None,
) -> str | None:
    """Append a structured event to the single per-user JSONL log file."""
    try:
        logs_dir = _root_logs_dir()
        file_name = f"user_{_safe_user_filename(user_id)}.jsonl"
        file_path = os.path.join(logs_dir, file_name)

        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id or "unknown_user",
            "event_type": event_type,
            "payload": _safe(payload or {}),
        }

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        return file_path
    except Exception:
        return None


def log_llm_call(
    *,
    conversation_id: str | None,
    user_id: str | None,
    source: str,
    request: Dict[str, Any] | None = None,
    response: Dict[str, Any] | None = None,
    error: str | None = None,
) -> int:
    """Append one LLM-call event and update per-chat call counter.

    Returns updated total LLM call count for the chat.
    """
    conv_id = (conversation_id or "unknown").strip() or "unknown"
    logs_dir = _base_logs_dir()

    summary_path = os.path.join(logs_dir, f"{conv_id}.summary.json")
    events_path  = os.path.join(logs_dir, f"{conv_id}.llm_calls.jsonl")

    summary: Dict[str, Any] = {
        "conversation_id": conv_id,
        "user_id": user_id,
        "llm_call_count": 0,
        "per_source_counts": {},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                summary.update(loaded)
        except Exception:
            pass

    per_source = summary.get("per_source_counts") or {}
    if not isinstance(per_source, dict):
        per_source = {}

    summary["llm_call_count"] = int(summary.get("llm_call_count", 0)) + 1
    per_source[source] = int(per_source.get(source, 0)) + 1
    summary["per_source_counts"] = per_source
    summary["user_id"] = user_id or summary.get("user_id")
    summary["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    event = {
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "conversation_id": conv_id,
        "user_id":         user_id,
        "source":          source,
        "llm_call_count":  summary["llm_call_count"],
        "request":         _safe(request or {}),
        "response":        _safe(response or {}),
        "error":           error,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    return int(summary["llm_call_count"])
