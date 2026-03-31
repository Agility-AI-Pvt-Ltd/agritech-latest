"""
pipeline/tools/dispatcher.py  –  Tool router + JSON logging helpers.

dispatch_tool() is the single entry point called by the agent to execute
any tool by name. It routes to the correct implementation module, captures
results, and writes a JSON log file for observability.
"""
from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from pipeline.logging_utils import append_user_event_log


# ──────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────────────

def _json_safe(value: Any) -> Any:
    """Best-effort conversion to JSON-serialisable types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_tool_log(
    tool_name: str,
    params: Dict[str, Any],
    result: Dict[str, Any],
    *,
    conversation_id: str | None = None,
    user_id: str | None = None,
    call_id: str | None = None,
    status: str = "ok",
    error: str | None = None,
) -> None:
    """Write one JSON file per tool call under <project_root>/logs/tool_calls/."""
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        logs_dir     = os.path.join(project_root, "logs", "tool_calls")
        os.makedirs(logs_dir, exist_ok=True)

        ts        = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        suffix    = uuid.uuid4().hex[:8]
        file_name = f"{ts}_{tool_name}_{suffix}.json"
        file_path = os.path.join(logs_dir, file_name)

        payload = {
            "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
            "tool":            tool_name,
            "status":          status,
            "conversation_id": conversation_id,
            "user_id":         user_id,
            "call_id":         call_id,
            "input":           _json_safe(params),
            "output":          _json_safe(result),
            "error":           error,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    except Exception as log_err:
        print(f"[!] Tool logging failed for {tool_name}: {log_err}")


def _append_user_tool_log(
    tool_name: str,
    params: Dict[str, Any],
    result: Dict[str, Any],
    *,
    conversation_id: str | None = None,
    user_id: str | None = None,
    call_id: str | None = None,
    status: str = "ok",
    error: str | None = None,
    elapsed_ms: float | None = None,
) -> None:
    append_user_event_log(
        user_id=user_id,
        event_type="tool_call",
        payload={
            "tool": tool_name,
            "status": status,
            "conversation_id": conversation_id,
            "call_id": call_id,
            "elapsed_ms": round(elapsed_ms, 2) if elapsed_ms is not None else None,
            "input": _json_safe(params),
            "output": _json_safe(result),
            "error": error,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def dispatch_tool(
    tool_name: str,
    params: Dict[str, Any],
    qdrant_client=None,
    chat_history: list | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    call_id: str | None = None,
) -> Dict[str, Any]:
    """Route a tool call by name to its implementation and log the result.

    To add a new tool:
        1. Implement execute_<tool>() in a new tools/<tool>.py file.
        2. Import it here and add an elif branch.
        3. Add its schema to tools/schemas.py.
    """
    result: Dict[str, Any]
    status  = "ok"
    err_msg: str | None = None
    start = time.perf_counter()

    try:
        if tool_name == "rag_search":
            from pipeline.tools.rag import execute_rag_search
            result = execute_rag_search(
                qdrant_client=qdrant_client,
                chat_history=chat_history,
                conversation_id=conversation_id,
                user_id=user_id,
                **params,
            )

        elif tool_name == "faq_search_by_crop_stage":
            from pipeline.tools.maize_faq import execute_faq_search_by_crop_stage
            result = execute_faq_search_by_crop_stage(
                qdrant_client=qdrant_client,
                conversation_id=conversation_id,
                user_id=user_id,
                **params,
            )

        elif tool_name == "set_crop_stage":
            from pipeline.tools.maize_faq import execute_set_crop_stage
            result = execute_set_crop_stage(**params)

        elif tool_name == "bighaat_search":
            from pipeline.tools.bighaat import execute_bighaat_search
            result = execute_bighaat_search(**params)

        elif tool_name == "get_weather":
            from pipeline.tools.weather import execute_get_weather
            result = execute_get_weather(**params)

        elif tool_name == "geocode_location":
            from pipeline.tools.weather import execute_geocode_location
            result = execute_geocode_location(**params)

        elif tool_name == "web_search":
            from pipeline.tools.web_search import execute_web_search
            result = execute_web_search(**params)

        elif tool_name == "get_current_datetime":
            from pipeline.tools.datetime_tool import execute_get_current_datetime
            result = execute_get_current_datetime(**params)

        else:
            result = {"error": f"Unknown tool: {tool_name}"}

    except Exception as exc:
        status  = "error"
        err_msg = str(exc)
        result  = {
            "error":     err_msg,
            "tool":      tool_name,
            "traceback": traceback.format_exc(),
        }

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    _write_tool_log(
        tool_name,
        params,
        result,
        conversation_id=conversation_id,
        user_id=user_id,
        call_id=call_id,
        status=status,
        error=err_msg,
    )
    _append_user_tool_log(
        tool_name,
        params,
        result,
        conversation_id=conversation_id,
        user_id=user_id,
        call_id=call_id,
        status=status,
        error=err_msg,
        elapsed_ms=elapsed_ms,
    )
    return result
