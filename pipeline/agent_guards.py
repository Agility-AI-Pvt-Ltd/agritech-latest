"""
pipeline/agent_guards.py  –  Deterministic prechecks for the agent node.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from pipeline.state import AgentState
from pipeline.tools import dispatch_tool

TEMPORAL_KEYWORDS = [
    "today", "tomorrow", "next day", "date", "day", "time", "current date", "current time",
    "aaj", "kal", "aaj kal", "aajkal", "aaj ka", "kal ka", "tarikh", "tareekh", "din", "samay", "waqt",
    "आज", "कल", "तारीख", "दिन", "समय", "वक्त",
]

MAIZE_KEYWORDS = [
    "maize", "corn", "sweet corn", "spring corn", "makka", "makkaa", "makai",
    "मक्का", "भुट्टा",
]

MAIZE_STAGE_CRITICAL_KEYWORDS = [
    "pesticide", "insecticide", "fungicide", "fertilizer", "urea", "dap", "npk",
    "spray", "dose", "dosage", "disease", "pest", "weed", "irrigation", "improve",
    "yield", "growth", "protect", "stage", "management", "advisory",
    "कीटनाशक", "फफूंदनाशक", "खाद", "यूरिया", "छिड़काव", "छिड़काव", "मात्रा",
    "रोग", "कीट", "सिंचाई", "उपज", "बचाव", "सलाह", "अवस्था",
]

MAIZE_FAQ_KEYWORDS = [
    "soil", "seed", "sowing", "variety", "varieties", "temperature", "climate",
    "fertilizer", "nutrition", "micronutrient", "irrigation", "weed", "harvest",
    "maturity", "pest", "disease", "spray", "dose", "management", "recommendation",
    "मिट्टी", "बीज", "बुवाई", "किस्म", "तापमान", "जलवायु", "खाद", "पोषण",
    "सूक्ष्म पोषक", "सिंचाई", "खरपतवार", "कटाई", "परिपक्वता", "कीट", "रोग",
    "छिड़काव", "छिड़काव", "मात्रा", "सलाह", "उपयुक्त",
]

MAIZE_ADVICE_KEYWORDS = [
    "advice", "advise", "suggest", "recommend", "what should i do", "what to do",
    "guide", "guidance", "management", "care", "treatment", "control", "prevent",
    "save crop", "protect crop", "plan", "next step",
    "सलाह", "क्या करूं", "क्या करूँ", "क्या करना", "उपाय", "सुझाव", "सिफारिश",
    "मार्गदर्शन", "प्रबंधन", "देखभाल", "इलाज", "नियंत्रण", "बचाव",
]

MAIZE_SOWING_DATE_REQUIREMENT = "maize_sowing_date"

SOWING_DATE_QUERY_KEYWORDS = [
    "sowing date", "planting date", "when sown", "when did i sow", "when was it sown",
    "bijai date", "buwai date", "bowai date", "बुवाई की तारीख", "बोवाई की तारीख",
    "बुवाई तारीख", "बोवाई तारीख", "कब बोई", "कब बोया",
]


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in keywords)


def _is_maize_stage_critical_query(text: str) -> bool:
    return _contains_any(text, MAIZE_KEYWORDS) and _contains_any(text, MAIZE_STAGE_CRITICAL_KEYWORDS)


def _last_assistant_asked_sowing_date(state: AgentState) -> bool:
    history = state.get("chat_history") or []
    for message in reversed(history):
        if message.get("role") != "assistant":
            continue
        content = (message.get("content") or "").lower()
        return any(
            hint in content
            for hint in [
                "sowing date",
                "planting date",
                "बुवाई की तारीख",
                "बोवाई की तारीख",
                "बुवाई",
                "बोवाई",
            ]
        )
    return False


def _asked_for_requirement(message_content: str, requirement: str | None) -> bool:
    content = (message_content or "").lower()
    if requirement == MAIZE_SOWING_DATE_REQUIREMENT:
        return any(
            hint in content
            for hint in [
                "sowing date",
                "planting date",
                "बुवाई की तारीख",
                "बोवाई की तारीख",
                "बुवाई",
                "बोवाई",
            ]
        )
    return False


def get_pending_user_intent(state: AgentState) -> str | None:
    """Return the pending user intent, reconstructing it from history if needed."""
    pending_query = (state.get("pending_user_intent") or "").strip()
    if pending_query:
        return pending_query

    history = state.get("chat_history") or []
    if len(history) < 2:
        return None

    requirement = state.get("pending_requirement") or MAIZE_SOWING_DATE_REQUIREMENT
    last_assistant_idx = None
    for idx in range(len(history) - 1, -1, -1):
        message = history[idx]
        if message.get("role") != "assistant":
            continue
        if _asked_for_requirement(message.get("content") or "", requirement):
            last_assistant_idx = idx
            break

    if last_assistant_idx is None:
        return None

    for idx in range(last_assistant_idx - 1, -1, -1):
        message = history[idx]
        if message.get("role") != "user":
            continue
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if requirement == MAIZE_SOWING_DATE_REQUIREMENT:
            if _is_maize_stage_critical_query(content):
                return content
            if not is_sowing_date_query(content):
                return content

    return None


def needs_datetime_tool(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    return bool(text) and any(keyword in text for keyword in TEMPORAL_KEYWORDS)


def extract_sowing_date_from_text(user_text: str) -> str | None:
    text = (user_text or "").strip()
    if not text:
        return None

    iso_match = re.search(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b", text)
    if iso_match:
        year, month, day = iso_match.groups()
        return f"{year}-{month}-{day}"

    dmy_match = re.search(r"\b(\d{2})[-/](\d{2})[-/](\d{4})\b", text)
    if dmy_match:
        day, month, year = dmy_match.groups()
        return f"{year}-{month}-{day}"

    return None


def _extract_relative_sowing_days(user_text: str) -> int | None:
    text = (user_text or "").strip().lower()
    if not text:
        return None

    patterns = [
        r"\b(\d{1,3})\s*(day|days)\s*ago\b",
        r"\b(\d{1,3})\s*(din)\s*(pehle|pahle|phle|phale)\b",
        r"\b(\d{1,3})\s*(दिन)\s*(पहले)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def get_latest_datetime_tool_result(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    for tool_call in reversed(tool_calls):
        if tool_call.get("tool") == "get_current_datetime":
            result = tool_call.get("result") or {}
            if isinstance(result, dict) and result.get("date"):
                return result
    return None


def apply_sowing_date_to_state(state: AgentState, sowing_date: str) -> None:
    print(
        "[SowingDate] Applying sowing date to runtime state: "
        f"{sowing_date} | conversation_id={state.get('conversation_id')} | "
        f"user_id={state.get('user_id')}"
    )
    state["user_sowing_date"] = sowing_date
    profile = dict(state.get("user_profile") or {})
    profile["sowing_date"] = sowing_date
    state["user_profile"] = profile


def resolve_relative_sowing_date(user_text: str, tool_calls: list[dict[str, Any]]) -> tuple[str | None, bool]:
    """
    Resolve relative sowing phrasing after a datetime tool call.
    Returns (resolved_date, needs_datetime_tool).
    """
    days_ago = _extract_relative_sowing_days(user_text)
    if days_ago is None:
        print(f"[SowingDate] No relative sowing pattern detected in input: {user_text!r}")
        return None, False

    current_dt = get_latest_datetime_tool_result(tool_calls)
    if not current_dt:
        print(
            "[SowingDate] Relative sowing date detected but datetime tool result is missing. "
            f"days_ago={days_ago}"
        )
        return None, True

    base_date = datetime.strptime(current_dt["date"], "%Y-%m-%d").date()
    sowing_date = base_date - timedelta(days=days_ago)
    print(
        "[SowingDate] Resolved relative sowing date from datetime tool: "
        f"input={user_text!r} | current_date={current_dt['date']} | "
        f"days_ago={days_ago} | sowing_date={sowing_date.isoformat()}"
    )
    return sowing_date.isoformat(), False


def needs_maize_sowing_date(user_text: str, state: AgentState) -> bool:
    text = (user_text or "").strip()
    profile = state.get("user_profile") or {}
    if not text:
        return False
    if not _contains_any(text, MAIZE_KEYWORDS):
        return False
    if is_sowing_date_query(text):
        return False
    if not (
        _is_maize_stage_critical_query(text)
        or _contains_any(text, MAIZE_FAQ_KEYWORDS)
        or _contains_any(text, MAIZE_ADVICE_KEYWORDS)
    ):
        return False
    if state.get("user_sowing_date") or profile.get("sowing_date"):
        return False
    if state.get("user_crop_stage") or profile.get("crop_stage"):
        return False
    return True


def should_route_to_stage_faq(user_text: str, state: AgentState) -> bool:
    """
    Deterministically route maize stage-specific advisory questions to the FAQ tool
    when the crop stage is already known.
    """
    text = (user_text or "").strip()
    if not text:
        return False
    if not state.get("user_crop_stage"):
        return False
    if is_sowing_date_query(text):
        return False
    return _is_maize_stage_critical_query(text)


def should_route_to_maize_faq(user_text: str, state: AgentState) -> bool:
    """
    Route maize FAQ-like questions to FAQ retrieval even when crop stage is unknown.
    This covers general maize questions like soil, seed, sowing, climate, etc.
    """
    text = (user_text or "").strip()
    if not text:
        return False
    if not _contains_any(text, MAIZE_KEYWORDS):
        return False
    if is_sowing_date_query(text):
        return False
    if state.get("user_crop_stage"):
        return should_route_to_stage_faq(text, state) or _contains_any(text, MAIZE_FAQ_KEYWORDS)
    return _contains_any(text, MAIZE_FAQ_KEYWORDS)


def should_interpret_relative_sowing_date(user_text: str, state: AgentState) -> bool:
    """Only treat relative phrases like '20 din pehle' as sowing-date input in sowing-date context."""
    if _extract_relative_sowing_days(user_text) is None:
        return False

    text = (user_text or "").strip()
    if _contains_any(text, SOWING_DATE_QUERY_KEYWORDS):
        return True

    return _last_assistant_asked_sowing_date(state)


def is_sowing_date_query(user_text: str) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False
    return _contains_any(text, MAIZE_KEYWORDS) and _contains_any(text, SOWING_DATE_QUERY_KEYWORDS)


def is_sowing_date_reply(user_text: str, state: AgentState) -> bool:
    """True when the user is likely replying to the assistant's request for sowing date."""
    text = (user_text or "").strip()
    if not text:
        return False
    if extract_sowing_date_from_text(text):
        return True
    if _extract_relative_sowing_days(text) is not None and _last_assistant_asked_sowing_date(state):
        return True
    return False


def auto_call_datetime_tool(
    state: AgentState,
    tool_calls: list[dict[str, Any]],
    loop_count: int,
    qdrant_client=None,
    *,
    call_suffix: str,
) -> bool:
    """Call the datetime tool once and update state for a re-loop. Returns True if called."""
    already_called = any((tc.get("tool") == "get_current_datetime") for tc in tool_calls)
    if already_called:
        return False

    call_id = f"auto_datetime_{call_suffix}_{loop_count}"
    result = dispatch_tool(
        "get_current_datetime",
        {},
        qdrant_client=qdrant_client,
        chat_history=state.get("chat_history"),
        conversation_id=state.get("conversation_id"),
        user_id=state.get("user_id"),
        call_id=call_id,
    )
    tool_calls.append(
        {
            "id": call_id,
            "tool": "get_current_datetime",
            "params": {},
            "result": result,
        }
    )
    state["tool_calls"] = tool_calls
    state["loop_count"] = loop_count + 1
    state["needs_more_info"] = True
    return True
