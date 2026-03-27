"""
pipeline/agent.py  –  The single LangGraph agent node.

Flow inside ONE node:
  1. Run deterministic prechecks.
  2. Build messages (system + history + user query + any tool results so far).
  3. Call LLM with bound tools.
  4. If LLM wants a tool → set needs_more_info=True (loop back in graph).
  5. If LLM gives final answer → set final_response, needs_more_info=False.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from pipeline.agent_context import build_messages
from pipeline.agent_guards import (
    MAIZE_SOWING_DATE_REQUIREMENT,
    apply_sowing_date_to_state,
    auto_call_datetime_tool,
    extract_sowing_date_from_text,
    get_pending_user_intent,
    get_latest_datetime_tool_result,
    is_sowing_date_reply,
    is_sowing_date_query,
    needs_datetime_tool,
    needs_maize_sowing_date,
    resolve_relative_sowing_date,
    should_interpret_relative_sowing_date,
)
from pipeline.logging_utils import log_llm_call
from pipeline.prompts.sowing_date_prompt import SOWING_DATE_RESOLUTION_SYSTEM
from pipeline.state import AgentState
from pipeline.tools import TOOLS, dispatch_tool

MAX_LOOPS = 5  # safety cap


def _set_pending_requirement(
    state: AgentState,
    *,
    original_query: str,
    requirement: str,
    pending_context: Dict[str, Any] | None = None,
) -> None:
    state["pending_user_intent"] = original_query
    state["pending_requirement"] = requirement
    state["pending_context"] = dict(pending_context or {})


def _clear_pending_requirement(state: AgentState) -> None:
    state["pending_user_intent"] = None
    state["pending_requirement"] = None
    state["pending_context"] = {}


def _build_resume_payload(state: AgentState, *, resolved_fact_ack: str, context_patch: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Resume a previously blocked user intent after a missing detail has been provided.
    """
    pending_query = get_pending_user_intent(state)
    if not pending_query:
        return None

    requirement = state.get("pending_requirement") or MAIZE_SOWING_DATE_REQUIREMENT
    pending_context = dict(state.get("pending_context") or {})
    pending_context.update(context_patch)
    _clear_pending_requirement(state)
    return {
        "resolved_fact_ack": resolved_fact_ack,
        "resumed_query": pending_query,
        "requirement": requirement,
        "context_patch": pending_context,
    }


def _apply_resume_payload(state: AgentState, resume_payload: Dict[str, Any]) -> None:
    context_patch = dict(resume_payload.get("context_patch") or {})
    sowing_date = context_patch.get("user_sowing_date")
    if sowing_date:
        apply_sowing_date_to_state(state, sowing_date)

    state["raw_input"] = resume_payload.get("resumed_query") or state.get("raw_input", "")
    state["resume_pending_intent"] = True
    state["resume_acknowledgment"] = resume_payload.get("resolved_fact_ack")


def _resolve_sowing_date_with_llm(llm, state: AgentState, tool_calls: List[Dict[str, Any]]) -> str | None:
    """Fallback to the LLM for multilingual sowing-date understanding."""
    current_dt = get_latest_datetime_tool_result(tool_calls)
    if not current_dt or not current_dt.get("date"):
        return None

    query = state.get("raw_input", "")
    prompt = (
        f"Current date: {current_dt['date']}\n"
        f"Farmer reply: {query}\n"
        "Extract maize sowing date if clearly implied."
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=SOWING_DATE_RESOLUTION_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        content = (getattr(response, "content", "") or "").strip()
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif content.startswith("```"):
            content = content.split("```", 1)[1].split("```", 1)[0].strip()
        parsed = json.loads(content)
        sowing_date = parsed.get("sowing_date")

        log_llm_call(
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
            source="agent.resolve_sowing_date",
            request={"query": query, "current_date": current_dt["date"]},
            response={"sowing_date": sowing_date, "reason": parsed.get("reason")},
        )

        if isinstance(sowing_date, str) and sowing_date:
            return sowing_date
        return None
    except Exception as exc:
        log_llm_call(
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
            source="agent.resolve_sowing_date.error",
            request={"query": query, "current_date": current_dt["date"]},
            error=str(exc),
        )
        return None


def _run_prechecks(state: AgentState, tool_calls: List[Dict[str, Any]], loop_count: int, llm, qdrant_client=None) -> bool:
    """Run deterministic checks before the main LLM call. Returns True if the node should exit early."""
    if state.get("resume_pending_intent"):
        return False

    if is_sowing_date_query(state.get("raw_input", "")):
        sowing_date = state.get("user_sowing_date") or state.get("user_profile", {}).get("sowing_date")
        state["needs_more_info"] = False
        if sowing_date:
            state["final_response"] = f"आपकी मक्का की बुवाई की तारीख {sowing_date} है।"
        else:
            state["final_response"] = (
                "मेरे पास अभी आपकी मक्का की बुवाई की तारीख नहीं है। "
                "कृपया बुवाई की तारीख बताइए, जैसे `2026-03-06` या `20 दिन पहले`."
            )
        return True

    sowing_reply = is_sowing_date_reply(state.get("raw_input", ""), state)
    extracted_sowing_date = extract_sowing_date_from_text(state.get("raw_input", ""))
    if extracted_sowing_date:
        apply_sowing_date_to_state(state, extracted_sowing_date)
        if sowing_reply:
            resume_payload = _build_resume_payload(
                state,
                resolved_fact_ack=f"ठीक है, मैंने आपकी मक्का की बुवाई की तारीख {extracted_sowing_date} सेव कर ली है।",
                context_patch={"user_sowing_date": extracted_sowing_date},
            )
            if resume_payload:
                _apply_resume_payload(state, resume_payload)
                return False
            state["needs_more_info"] = False
            state["final_response"] = f"ठीक है, मैंने आपकी मक्का की बुवाई की तारीख {extracted_sowing_date} सेव कर ली है।"
            return True
        else:
            state["needs_more_info"] = False
            state["final_response"] = f"ठीक है, मैंने आपकी मक्का की बुवाई की तारीख {extracted_sowing_date} सेव कर ली है।"
            return True
    elif should_interpret_relative_sowing_date(state.get("raw_input", ""), state):
        relative_sowing_date, needs_datetime_for_sowing = resolve_relative_sowing_date(
            state.get("raw_input", ""),
            tool_calls,
        )
        if relative_sowing_date:
            apply_sowing_date_to_state(state, relative_sowing_date)
            resume_payload = _build_resume_payload(
                state,
                resolved_fact_ack=f"ठीक है, मैंने आपकी मक्का की बुवाई की तारीख {relative_sowing_date} सेव कर ली है।",
                context_patch={"user_sowing_date": relative_sowing_date},
            )
            if resume_payload:
                _apply_resume_payload(state, resume_payload)
                return False
            else:
                state["needs_more_info"] = False
                state["final_response"] = f"ठीक है, मैंने आपकी मक्का की बुवाई की तारीख {relative_sowing_date} सेव कर ली है।"
                return True
        elif needs_datetime_for_sowing and auto_call_datetime_tool(
            state,
            tool_calls,
            loop_count,
            qdrant_client=qdrant_client,
            call_suffix="sowing",
        ):
            return True

    if sowing_reply:
        if auto_call_datetime_tool(
            state,
            tool_calls,
            loop_count,
            qdrant_client=qdrant_client,
            call_suffix="sowing_fallback",
        ):
            return True

        llm_resolved_sowing_date = _resolve_sowing_date_with_llm(llm, state, tool_calls)
        if llm_resolved_sowing_date:
            apply_sowing_date_to_state(state, llm_resolved_sowing_date)
            resume_payload = _build_resume_payload(
                state,
                resolved_fact_ack=f"ठीक है, मैंने आपकी मक्का की बुवाई की तारीख {llm_resolved_sowing_date} सेव कर ली है।",
                context_patch={"user_sowing_date": llm_resolved_sowing_date},
            )
            if resume_payload:
                _apply_resume_payload(state, resume_payload)
                return False
            else:
                state["needs_more_info"] = False
                state["final_response"] = f"ठीक है, मैंने आपकी मक्का की बुवाई की तारीख {llm_resolved_sowing_date} सेव कर ली है।"
                return True

        state["needs_more_info"] = False
        state["final_response"] = (
            "कृपया बुवाई की तारीख थोड़ी और साफ़ तरीके से बताइए, जैसे `2026-03-06`, "
            "`20 दिन पहले`, या `पिछले महीने 5 तारीख`."
        )
        return True

    if needs_maize_sowing_date(state.get("raw_input", ""), state):
        _set_pending_requirement(
            state,
            original_query=state.get("raw_input", ""),
            requirement=MAIZE_SOWING_DATE_REQUIREMENT,
            pending_context={"crop": "maize"},
        )
        state["needs_more_info"] = False
        state["final_response"] = (
            "मक्का की सही और स्टेज-आधारित सलाह देने के लिए पहले बुवाई की तारीख बताइए। "
            "कृपया तारीख `YYYY-MM-DD` में बताएं, जैसे `2025-07-10`."
        )
        return True

    if needs_datetime_tool(state.get("raw_input", "")) and auto_call_datetime_tool(
        state,
        tool_calls,
        loop_count,
        qdrant_client=qdrant_client,
        call_suffix="general",
    ):
        return True

    return False


def agent_node(state: AgentState, llm, qdrant_client=None) -> AgentState:
    """
    Main decision-making node.
    1. Translates Studio Chat UI `messages` → `raw_input` seamlessly.
    2. Runs deterministic prechecks.
    3. Builds context + prompts LLM.
    4. Handles tool calling vs final response.
    """
    tool_calls: List[Dict[str, Any]] = list(state.get("tool_calls") or [])
    loop_count: int = state.get("loop_count", 0)

    if loop_count >= MAX_LOOPS:
        state["needs_more_info"] = False
        state["final_response"] = (
            state.get("final_response")
            or "I've gathered information but reached the processing limit. Please refine your question."
        )
        return state

    if state.get("messages") and not state.get("raw_input"):
        last_msg = state["messages"][-1]
        if last_msg.type == "human":
            state["raw_input"] = last_msg.content

    print(f"\n[Agent Node] Processing input: {state.get('raw_input')}")

    if _run_prechecks(state, tool_calls, loop_count, llm, qdrant_client=qdrant_client):
        return state

    messages = build_messages(state)

    try:
        if state.get("tools_used", False):
            response = llm.invoke(messages)
            studio_msgs = list(state.get("messages", []))
            studio_msgs.append(AIMessage(content=response.content or ""))
            state["messages"] = studio_msgs
        else:
            llm_with_tools = llm.bind_tools(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": t.get("name"),
                            "description": t.get("description"),
                            "parameters": t.get("parameters", {}),
                        },
                    }
                    for t in TOOLS
                ]
            )
            response = llm_with_tools.invoke(messages)

            studio_msgs = list(state.get("messages", []))
            if response.content:
                studio_msgs.append(AIMessage(content=response.content))
            state["messages"] = studio_msgs

    except Exception as exc:
        log_llm_call(
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
            source="agent.error",
            request={"messages_count": len(messages), "loop_count": loop_count},
            error=str(exc),
        )
        state["needs_more_info"] = False
        state["final_response"] = "Sorry, I encountered an error. Please try again."
        return state

    log_llm_call(
        conversation_id=state.get("conversation_id"),
        user_id=state.get("user_id"),
        source="agent.invoke",
        request={
            "messages_count": len(messages),
            "loop_count": loop_count,
            "tools_mode": not state.get("tools_used", False),
        },
        response={
            "has_content": bool(getattr(response, "content", "")),
            "tool_calls_count": len((getattr(response, "tool_calls", None) or [])),
        },
    )

    raw_tool_calls = getattr(response, "tool_calls", None) or []
    if not raw_tool_calls:
        raw_tool_calls = response.additional_kwargs.get("tool_calls") or []

    if raw_tool_calls:
        for tc in raw_tool_calls:
            if isinstance(tc, dict):
                name = tc.get("name") or tc.get("function", {}).get("name", "")
                args_raw = tc.get("args") or tc.get("function", {}).get("arguments", "{}")
                call_id = tc.get("id", f"call_{name}_{loop_count}")
            else:
                name = tc.name
                args_raw = tc.args
                call_id = getattr(tc, "id", f"call_{name}_{loop_count}")

            if isinstance(args_raw, str):
                try:
                    params = json.loads(args_raw)
                except json.JSONDecodeError:
                    params = {}
            else:
                params = dict(args_raw)

            if name == "web_search":
                params.setdefault("state", state.get("user_state") or "Uttar Pradesh")
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

        state["tool_calls"] = tool_calls
        state["loop_count"] = loop_count + 1
        state["needs_more_info"] = True
        return state

    state["tool_calls"] = tool_calls
    state["loop_count"] = loop_count + 1
    state["needs_more_info"] = False
    state["final_response"] = response.content or ""
    state["resume_pending_intent"] = False
    state["resume_acknowledgment"] = None
    return state
