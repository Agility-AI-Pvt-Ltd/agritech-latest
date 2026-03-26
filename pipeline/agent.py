"""
pipeline/agent.py  –  The single LangGraph agent node.

Flow inside ONE node:
  1. Build messages (system + history + user query + any tool results so far).
  2. Call LLM with bound tools.
  3. If LLM wants a tool → set needs_more_info=True (loop back in graph).
  4. If LLM gives final answer → set final_response, needs_more_info=False.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from pipeline.state import AgentState
from pipeline.tools import TOOLS, dispatch_tool
from pipeline.logging_utils import log_llm_call
from pipeline.prompts.system_prompt import SYSTEM_PROMPT

MAX_LOOPS = 5  # safety cap


# ──────────────────────────────────────────────────────────────────────────────
# Profile context helpers
# ──────────────────────────────────────────────────────────────────────────────

def _profile_block(profile: dict | None) -> str:
    """Build a human-readable profile context string to append to the system prompt."""
    if not profile:
        return ""
    lines = ["\n--- Known User Profile ---"]
    if profile.get("name"):            lines.append(f"Name: {profile['name']}")
    if profile.get("language"):        lines.append(f"Preferred language: {profile['language']}")
    if profile.get("location"):        lines.append(f"Location: {profile['location']}")
    if profile.get("state"):           lines.append(f"State: {profile['state']}")
    if profile.get("country"):         lines.append(f"Country: {profile['country']}")
    if profile.get("latitude") is not None and profile.get("longitude") is not None:
        lines.append(f"Coordinates: {profile['latitude']}, {profile['longitude']}")
    if profile.get("farm_size_acres"): lines.append(f"Farm size: {profile['farm_size_acres']} acres")
    if profile.get("soil_type"):       lines.append(f"Soil type: {profile['soil_type']}")
    if profile.get("crops"):           lines.append(f"Crops: {', '.join(profile['crops'])}")
    extra = profile.get("extra_facts") or {}
    for k, v in extra.items():         lines.append(f"{k}: {v}")
    lines.append("Use this profile to personalize your responses without asking again.")
    lines.append("--- End of Profile ---")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Message building
# ──────────────────────────────────────────────────────────────────────────────

def _build_messages(state: AgentState) -> list:
    """Construct the full message list for the LLM call."""
    profile_ctx    = _profile_block(state.get("user_profile"))
    system_content = SYSTEM_PROMPT + profile_ctx
    messages       = [SystemMessage(content=system_content)]

    # Rolling summary of older compressed messages
    summary = state.get("conversation_summary")
    if summary:
        messages.append(SystemMessage(
            content=f"[Earlier conversation summary — treat as established context]\n{summary}"
        ))

    # Recency priority note
    history = state.get("chat_history") or []
    if history:
        messages.append(SystemMessage(
            content=(
                "IMPORTANT: The conversation history below is ordered oldest→newest. "
                "More recent messages carry HIGHER priority. When information conflicts, "
                "always prefer the most recent statement."
            )
        ))

    # Group turns (user + assistant pairs) and label by recency
    pairs: list[tuple] = []
    i = 0
    while i < len(history):
        user_msg = history[i]     if i     < len(history) else None
        asst_msg = history[i + 1] if i + 1 < len(history) else None
        pairs.append((user_msg, asst_msg))
        i += 2

    total_turns = len(pairs)
    for turn_idx, (user_msg, asst_msg) in enumerate(pairs, start=1):
        age = total_turns - turn_idx  # 0 = most recent
        if age == 0:
            label = f"[Turn {turn_idx}/{total_turns} — MOST RECENT ↑ highest priority]"
        elif age <= 1:
            label = f"[Turn {turn_idx}/{total_turns} — recent]"
        else:
            label = f"[Turn {turn_idx}/{total_turns} — older, lower priority]"

        if user_msg:
            messages.append(HumanMessage(content=f"{label}\n{user_msg.get('content', '')}"))
        if asst_msg:
            messages.append(AIMessage(content=f"{label}\n{asst_msg.get('content', '')}"))

    # Current user query
    messages.append(HumanMessage(content=state.get("raw_input", "")))

    # Inject prior tool call history so LLM knows what was already tried
    for call in (state.get("tool_calls") or []):
        tool_name = call.get("tool")
        params    = call.get("params", {})
        result    = call.get("result", {})

        messages.append(
            AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_name,
                    "args": params,
                    "id":   call.get("id", f"call_{tool_name}"),
                }],
            )
        )
        messages.append(
            ToolMessage(
                content=json.dumps(result),
                tool_call_id=call.get("id", f"call_{tool_name}"),
            )
        )

    return messages


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic temporal check
# ──────────────────────────────────────────────────────────────────────────────

_TEMPORAL_KEYWORDS = [
    "today", "tomorrow", "next day", "date", "day", "time", "current date", "current time",
    "aaj", "kal", "aaj kal", "aajkal", "aaj ka", "kal ka", "tarikh", "tareekh", "din", "samay", "waqt",
    "आज", "कल", "तारीख", "दिन", "समय", "वक्त",
]


def _needs_datetime_tool(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False
    return any(k in text for k in _TEMPORAL_KEYWORDS)


# ──────────────────────────────────────────────────────────────────────────────
# Agent node
# ──────────────────────────────────────────────────────────────────────────────

def agent_node(state: AgentState, llm, qdrant_client=None) -> AgentState:
    """
    Main decision-making node.
    1. Translates Studio Chat UI `messages` → `raw_input` seamlessly.
    2. Builds context + prompts LLM.
    3. Handles tool calling vs final response.
    """
    errors:     List[str]             = list(state.get("errors") or [])
    tool_calls: List[Dict[str, Any]]  = list(state.get("tool_calls") or [])
    loop_count: int                   = state.get("loop_count", 0)

    # Safety cap
    if loop_count >= MAX_LOOPS:
        state["needs_more_info"] = False
        state["final_response"]  = (
            state.get("final_response")
            or "I've gathered information but reached the processing limit. Please refine your question."
        )
        return state

    # [Studio intercept] Grab latest Human message if invoked from Studio Chat UI
    if state.get("messages") and not state.get("raw_input"):
        last_msg = state["messages"][-1]
        if last_msg.type == "human":
            state["raw_input"] = last_msg.content

    print(f"\n[Agent Node] Processing input: {state.get('raw_input')}")

    # Deterministic temporal handling — force datetime call before answering
    if _needs_datetime_tool(state.get("raw_input", "")):
        already_called = any((tc.get("tool") == "get_current_datetime") for tc in tool_calls)
        if not already_called:
            call_id = f"auto_datetime_{loop_count}"
            result  = dispatch_tool(
                "get_current_datetime",
                {},
                qdrant_client=qdrant_client,
                chat_history=state.get("chat_history"),
                conversation_id=state.get("conversation_id"),
                user_id=state.get("user_id"),
                call_id=call_id,
            )
            tool_calls.append({
                "id":     call_id,
                "tool":   "get_current_datetime",
                "params": {},
                "result": result,
            })
            state["tool_calls"]    = tool_calls
            state["loop_count"]    = loop_count + 1
            state["needs_more_info"] = True
            return state

    messages = _build_messages(state)

    try:
        if state.get("tools_used", False):
            # After tool results — force a final answer
            response = llm.invoke(messages)
            studio_msgs = list(state.get("messages", []))
            studio_msgs.append(AIMessage(content=response.content or ""))
            state["messages"] = studio_msgs
        else:
            # First pass — bind tools and let LLM decide
            llm_with_tools = llm.bind_tools(
                [{
                    "type": "function",
                    "function": {
                        "name":        t.get("name"),
                        "description": t.get("description"),
                        "parameters":  t.get("parameters", {}),
                    },
                } for t in TOOLS]
            )
            response = llm_with_tools.invoke(messages)

            studio_msgs = list(state.get("messages", []))
            if response.content:
                studio_msgs.append(AIMessage(content=response.content))
            state["messages"] = studio_msgs

    except Exception as e:
        log_llm_call(
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
            source="agent.error",
            request={"messages_count": len(messages), "loop_count": loop_count},
            error=str(e),
        )
        state["needs_more_info"] = False
        state["final_response"]  = "Sorry, I encountered an error. Please try again."
        return state

    log_llm_call(
        conversation_id=state.get("conversation_id"),
        user_id=state.get("user_id"),
        source="agent.invoke",
        request={
            "messages_count": len(messages),
            "loop_count":     loop_count,
            "tools_mode":     not state.get("tools_used", False),
        },
        response={
            "has_content":       bool(getattr(response, "content", "")),
            "tool_calls_count":  len((getattr(response, "tool_calls", None) or [])),
        },
    )

    # ── Check if LLM wants to call a tool ─────────────────────────────────────
    raw_tool_calls = getattr(response, "tool_calls", None) or []
    if not raw_tool_calls:
        raw_tool_calls = response.additional_kwargs.get("tool_calls") or []

    if raw_tool_calls:
        for tc in raw_tool_calls:
            # Normalize across providers
            if isinstance(tc, dict):
                name     = tc.get("name") or tc.get("function", {}).get("name", "")
                args_raw = tc.get("args") or tc.get("function", {}).get("arguments", "{}")
                call_id  = tc.get("id", f"call_{name}_{loop_count}")
            else:
                name     = tc.name
                args_raw = tc.args
                call_id  = getattr(tc, "id", f"call_{name}_{loop_count}")

            if isinstance(args_raw, str):
                try:
                    params = json.loads(args_raw)
                except json.JSONDecodeError:
                    params = {}
            else:
                params = dict(args_raw)

            if name == "web_search":
                params.setdefault("state",   state.get("user_state")   or "Uttar Pradesh")
                params.setdefault("country", state.get("user_country") or "India")

            result = dispatch_tool(
                name,
                params,
                qdrant_client=qdrant_client,
                chat_history=state.get("chat_history"),
                conversation_id=state.get("conversation_id"),
                user_id=state.get("user_id"),
                call_id=call_id,
            )

            tool_calls.append({"id": call_id, "tool": name, "params": params, "result": result})

            if name == "rag_search" and "chunks" in result:
                existing = list(state.get("retrieved_chunks") or [])
                existing.extend(result["chunks"])
                state["retrieved_chunks"] = existing
            elif name == "geocode_location" and "error" not in result:
                state["user_location"] = result.get("location") or result.get("resolved_address") or params.get("address")
                state["user_state"] = result.get("state") or state.get("user_state")
                state["user_country"] = result.get("country") or state.get("user_country")
                state["user_latitude"] = result.get("latitude")
                state["user_longitude"] = result.get("longitude")

        state["tool_calls"]    = tool_calls
        state["loop_count"]    = loop_count + 1
        state["needs_more_info"] = True
        return state

    # ── LLM gave a final answer ────────────────────────────────────────────────
    state["tool_calls"]    = tool_calls
    state["loop_count"]    = loop_count + 1
    state["needs_more_info"] = False
    state["final_response"]  = response.content or ""
    return state
