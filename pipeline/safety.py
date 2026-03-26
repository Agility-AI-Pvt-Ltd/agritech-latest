"""
pipeline/safety.py  –  Front-door safety gate for /api/chat.

Contains:
- heuristic security blocking
- low-information / garbage-input blocking
- optional safety-model classification
- graph routing helper and non-persistence reason set
"""
from __future__ import annotations

import json
import re

from core.config import settings
from pipeline.logging_utils import log_llm_call
from pipeline.prompts.safety_prompt import SAFETY_SYSTEM
from pipeline.state import AgentState

NON_PERSISTED_SAFETY_REASONS = {
    "empty_query",
    "too_short",
    "symbols_only",
    "symbol_noise",
    "numeric_noise",
    "abuse_only",
    "gibberish_query",
    "sql_like_noise",
}

_HEURISTIC_SAFETY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(ignore|bypass|override).{0,40}\b(system|developer|previous) instructions?\b", re.I), "prompt_injection"),
    (re.compile(r"\b(reveal|show|print|dump|leak).{0,40}\b(system prompt|developer prompt|hidden prompt|chain.of.thought|cot)\b", re.I), "prompt_exfiltration"),
    (re.compile(r"\b(api key|token|secret|password|credential|env var|environment variable)\b", re.I), "secret_exfiltration"),
    (re.compile(r"\b(malware|ransomware|botnet|ddos|phishing|keylogger|stealer|trojan|exploit)\b", re.I), "malware_or_attack"),
    (re.compile(r"\b(sql injection|sqli|xss|csrf|reverse shell|shellcode|payload)\b", re.I), "exploit_payload"),
    (re.compile(r"\b(run|execute)\b.{0,30}\b(bash|shell|terminal|command|powershell|cmd)\b", re.I), "command_execution_probe"),
]

_ABUSE_WORDS = {
    "madarchod", "mc", "bhenchod", "bc", "chutiya", "gandu", "harami",
    "fuck", "fucking", "bitch", "asshole", "bastard", "idiot",
}


def should_route_from_safety(state: AgentState) -> str:
    """End graph early when the safety node blocks the request."""
    if state.get("safety_decision") == "block":
        from langgraph.graph import END
        return END
    return "agent"


def _friendly_block_message() -> str:
    return (
        "माफ कीजिए, मैं केवल सुरक्षित और वैध कृषि सलाह से जुड़े सवालों में मदद कर सकता हूँ। "
        "अगर आपको फसल, मौसम, कीट, बीमारी, खाद या खेती प्रबंधन पर सलाह चाहिए, तो वही प्रश्न पूछें।"
    )


def _friendly_invalid_message() -> str:
    return (
        "कृपया अपना सवाल साफ़ शब्दों में लिखें। मैं फसल, मौसम, कीट, बीमारी, खाद "
        "और खेती प्रबंधन से जुड़े प्रश्नों में मदद कर सकता हूँ।"
    )


def _is_symbol_heavy(text: str) -> bool:
    if not text:
        return False
    meaningful = sum(ch.isalnum() or ch.isspace() for ch in text)
    return meaningful / max(len(text), 1) < 0.4


def _is_low_information_query(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped:
        return "block", "empty_query"

    lowered = stripped.lower()
    words = re.findall(r"[a-zA-Z]+|[\u0900-\u097F]+", lowered)
    digits = re.findall(r"\d", lowered)

    if len(stripped) <= 3:
        return "block", "too_short"

    if re.fullmatch(r"[\W_]+", stripped):
        return "block", "symbols_only"

    if _is_symbol_heavy(stripped) and len(words) <= 1:
        return "block", "symbol_noise"

    if digits and not words and len(digits) >= 4:
        return "block", "numeric_noise"

    if words and all(word in _ABUSE_WORDS for word in words) and len(words) <= 6:
        return "block", "abuse_only"

    if words:
        long_words = [w for w in words if len(w) >= 4]
        unique_ratio = len(set(words)) / max(len(words), 1)
        vowelish = sum(ch in "aeiou" for ch in "".join(words))
        if len(long_words) >= 2 and unique_ratio > 0.8 and vowelish <= 2:
            return "block", "gibberish_query"

    if re.search(r"\bselect\b", lowered) and ("from" in lowered or "ffrom" in lowered) and len(words) <= 4:
        return "block", "sql_like_noise"

    return None


def _heuristic_safety_check(user_text: str) -> tuple[str, str] | None:
    text = (user_text or "").strip()
    if settings.chat_low_info_blocking_enabled:
        low_info = _is_low_information_query(text)
        if low_info:
            return low_info

    if settings.chat_security_blocking_enabled:
        for pattern, reason in _HEURISTIC_SAFETY_PATTERNS:
            if pattern.search(text):
                return "block", reason
    return None


def safety_node(state: AgentState, safety_llm=None) -> AgentState:
    """Pre-agent safety gate to block prompt-injection and bad input."""
    query = state.get("raw_input", "") or ""
    state["safety_decision"] = "allow"
    state["safety_reason"] = None

    if not settings.chat_safety_enabled:
        return state

    heuristic = _heuristic_safety_check(query)
    if heuristic:
        decision, reason = heuristic
        log_llm_call(
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
            source="graph.safety_gate.heuristic",
            request={"query_length": len(query)},
            response={"decision": decision, "reason": reason},
        )
        state["safety_decision"] = decision
        state["safety_reason"] = reason
        state["needs_more_info"] = False
        if reason in NON_PERSISTED_SAFETY_REASONS:
            state["final_response"] = _friendly_invalid_message()
        else:
            state["final_response"] = _friendly_block_message()
        return state

    if not settings.chat_safety_model_classification_enabled:
        state["safety_decision"] = "allow"
        state["safety_reason"] = "safety_model_classification_disabled"
        return state

    if safety_llm is None:
        if settings.chat_safety_fail_closed:
            state["safety_decision"] = "block"
            state["safety_reason"] = "safety_model_unavailable"
            state["needs_more_info"] = False
            state["final_response"] = _friendly_block_message()
        return state

    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        response = safety_llm.invoke([
            SystemMessage(content=SAFETY_SYSTEM),
            HumanMessage(content=query),
        ])
        content = (getattr(response, "content", "") or "").strip()
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif content.startswith("```"):
            content = content.split("```", 1)[1].split("```", 1)[0].strip()
        parsed = json.loads(content)
        decision = (parsed.get("decision") or "allow").strip().lower()
        reason = (parsed.get("reason") or "").strip() or None
        user_message = (parsed.get("user_message") or "").strip()

        log_llm_call(
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
            source="graph.safety_gate",
            request={"query_length": len(query)},
            response={"decision": decision, "reason": reason},
        )

        if decision == "block":
            state["safety_decision"] = "block"
            state["safety_reason"] = reason or "blocked_by_safety_model"
            state["needs_more_info"] = False
            state["final_response"] = user_message or _friendly_block_message()
            return state

        state["safety_decision"] = "allow"
        state["safety_reason"] = reason
        return state
    except Exception as exc:
        log_llm_call(
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
            source="graph.safety_gate.error",
            request={"query_length": len(query)},
            error=str(exc),
        )
        if settings.chat_safety_fail_closed:
            state["safety_decision"] = "block"
            state["safety_reason"] = "safety_check_error"
            state["needs_more_info"] = False
            state["final_response"] = _friendly_block_message()
        else:
            state["safety_decision"] = "allow"
            state["safety_reason"] = "safety_check_error_bypassed"
        return state
