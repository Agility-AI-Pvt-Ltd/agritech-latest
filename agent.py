"""
agent.py  –  The single LangGraph agent node.

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

from state import AgentState
from tools import TOOLS, dispatch_tool

MAX_LOOPS = 5  # safety cap

SYSTEM_PROMPT = """You are an expert agricultural advisor for Indian farmers, specializing in Spring Corn (Zaid Maize) cultivation in Uttar Pradesh.

You have access to four tools:
- rag_search: Search the agricultural knowledge base for crop practices, fertilizers, pests & diseases.
- get_weather: Get current weather and 3-day forecast using latitude/longitude.
- web_search: Search the web for information not in the knowledge base.
- get_current_datetime: Get the current date, day of week, time and farming season.

Guidelines:
1. Always try rag_search first for crop/farming questions.
2. Use get_weather when the user asks about weather or needs weather context.
3. CRITICAL WEATHER RULE: DO NOT guess or approximate coordinates! If you do not know the user's exact latitude/longitude, you MUST call geocode_location with their city/village name to get the coordinates. If you don't even know their city name, politely ask them for it first.
4. Use web_search only as a fallback.
5. STRICT RAG RULE: If the retrieved information from rag_search is empty or insufficient, you MUST NOT answer from your own knowledge immediately. You MUST call rag_search AGAIN with a significantly refined or simplified query.
6. IF A TOOL RETURNS AN ERROR (e.g. Collection not found), DO NOT call the exact same tool again. Use web_search as fallback.
7. ALWAYS respond in proper Hindi using Devanagari script (e.g. "नमस्ते! कैसे मदद कर सकता हूँ?"). NEVER use Hinglish or English unless citing technical terms.
8. PROACTIVE ADVISOR RULE: During greetings or small talk, do not just ask "How are you?". Actively offer agricultural help by asking relevant questions, such as:
   - "आज/कल खेत में क्या काम करना चाहिए?"
   - "क्या आप अगले 7 दिनों का मौसम और उससे जुड़े जोखिम जानना चाहते हैं?"
   - "क्या आपकी फसल में कोई बीमारी के लक्षण दिख रहे हैं?"
   - "क्या आप खाद या कीटनाशक डालने के बारे में जानना चाहते हैं?"
   - "अपनी फसल को गर्मी से कैसे बचाएं?"
9. Be practical, concise, and farmer-friendly.
10. When you have enough information, give a direct, actionable response — do not call more tools."""


def _profile_block(profile: dict | None) -> str:
    """Build a human-readable profile context string for the system prompt."""
    if not profile:
        return ""
    lines = ["\n--- Known User Profile ---"]
    if profile.get("name"):            lines.append(f"Name: {profile['name']}")
    if profile.get("language"):        lines.append(f"Preferred language: {profile['language']}")
    if profile.get("location"):        lines.append(f"Location: {profile['location']}")
    if profile.get("latitude"):        lines.append(f"Coordinates: {profile['latitude']}, {profile['longitude']}")
    if profile.get("farm_size_acres"): lines.append(f"Farm size: {profile['farm_size_acres']} acres")
    if profile.get("soil_type"):       lines.append(f"Soil type: {profile['soil_type']}")
    if profile.get("crops"):           lines.append(f"Crops: {', '.join(profile['crops'])}")
    extra = profile.get("extra_facts") or {}
    for k, v in extra.items():         lines.append(f"{k}: {v}")
    lines.append("Use this profile to personalize your responses without asking again.")
    lines.append("--- End of Profile ---")
    return "\n".join(lines)


def _build_messages(state: AgentState) -> list:
    """Build the full message list for the LLM call."""
    profile_ctx = _profile_block(state.get("user_profile"))
    system_content = SYSTEM_PROMPT + profile_ctx
    messages = [SystemMessage(content=system_content)]

    # Rolling summary of older compressed messages (injected when window overflows)
    summary = state.get("conversation_summary")
    if summary:
        messages.append(SystemMessage(
            content=f"[Earlier conversation summary — treat as established context]\n{summary}"
        ))

    # System note: recency priority
    history = state.get("chat_history") or []
    if history:
        messages.append(SystemMessage(
            content="IMPORTANT: The conversation history below is ordered oldest→newest. "
                    "More recent messages carry HIGHER priority. When information conflicts, "
                    "always prefer the most recent statement."
        ))

    # Recent chat history with recency labels (last N messages verbatim)
    # Group into turns (user+assistant pairs) to assign Turn N of M labels
    pairs: list[tuple] = []
    i = 0
    while i < len(history):
        user_msg  = history[i] if i < len(history) else None
        asst_msg  = history[i + 1] if i + 1 < len(history) else None
        pairs.append((user_msg, asst_msg))
        i += 2

    total_turns = len(pairs)
    for turn_idx, (user_msg, asst_msg) in enumerate(pairs, start=1):
        # Recency label — makes it explicit to the LLM
        age = total_turns - turn_idx               # 0 = most recent
        if age == 0:
            label = f"[Turn {turn_idx}/{total_turns} — MOST RECENT ↑ highest priority]"
        elif age <= 1:
            label = f"[Turn {turn_idx}/{total_turns} — recent]"
        else:
            label = f"[Turn {turn_idx}/{total_turns} — older, lower priority]"

        if user_msg:
            content = user_msg.get("content", "")
            messages.append(HumanMessage(content=f"{label}\n{content}"))
        if asst_msg:
            content = asst_msg.get("content", "")
            messages.append(AIMessage(content=f"{label}\n{content}"))

    # Current user query
    messages.append(HumanMessage(content=state.get("raw_input", "")))

    # Inject tool call history as ToolMessages so LLM knows what was already tried
    for call in (state.get("tool_calls") or []):
        tool_name = call.get("tool")
        params = call.get("params", {})
        result = call.get("result", {})

        # Fake AIMessage with tool_use (so the LLM sees its own prior calls)
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_name,
                    "args": params,
                    "id": call.get("id", f"call_{tool_name}"),
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


def agent_node(state: AgentState, llm, qdrant_client=None) -> AgentState:
    """
    Main decision-making node.
    1. Translates Studio Chat UI `messages` -> `raw_input` seamlessly.
    2. Builds context + prompts LLM.
    3. Handles tool calling vs final response.
    """
    errors: List[str] = list(state.get("errors") or [])
    tool_calls: List[Dict[str, Any]] = list(state.get("tool_calls") or [])
    loop_count: int = state.get("loop_count", 0)

    # Safety cap
    if loop_count >= MAX_LOOPS:
        state["needs_more_info"] = False
        state["final_response"] = (
            state.get("final_response")
            or "I've gathered information but reached the processing limit. Please refine your question."
        )
        return state

    # [Studio Intercept] If invoked from Studio Chat UI, grab the latest Human message
    if state.get("messages") and not state.get("raw_input"):
        last_msg = state["messages"][-1]
        if last_msg.type == "human":
            state["raw_input"] = last_msg.content

    print(f"\n[Agent Node] Processing input: {state.get('raw_input')}")

    messages = _build_messages(state)

    try:
        if state.get("tools_used", False):
            # If we just used tools, force the LLM to give a final answer
            response = llm.invoke(messages)
            
            # [Studio Sync] Append to messages
            studio_msgs = list(state.get("messages", []))
            studio_msgs.append(AIMessage(content=response.content or ""))
            state["messages"] = studio_msgs

        else:
            # Otherwise, bind tools and let it decide
            from tools import TOOLS
            llm_with_tools = llm.bind_tools(
                [{
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description"),
                        "parameters": t.get("parameters", {}),
                    }
                } for t in TOOLS]
            )
            response = llm_with_tools.invoke(messages)

            # [Studio Sync] Append AIMessage to state messages
            studio_msgs = list(state.get("messages", []))
            if response.content:
                studio_msgs.append(AIMessage(content=response.content))
            state["messages"] = studio_msgs

    except Exception as e:
        state["needs_more_info"] = False
        state["final_response"] = "Sorry, I encountered an error. Please try again."
        return state

    # ── Check if LLM wants to call a tool ──────────────────────────────────
    raw_tool_calls = getattr(response, "tool_calls", None) or []
    if not raw_tool_calls:
        # Also check additional_kwargs for providers that put tool_calls there
        raw_tool_calls = response.additional_kwargs.get("tool_calls") or []

    if raw_tool_calls:
        # Execute each requested tool
        for tc in raw_tool_calls:
            # Normalize across providers
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

            # Execute tool
            result = dispatch_tool(
                name,
                params,
                qdrant_client=qdrant_client,
                chat_history=state.get("chat_history")
            )

            tool_calls.append({
                "id": call_id,
                "tool": name,
                "params": params,
                "result": result,
            })

            # Store RAG chunks separately for easy access
            if name == "rag_search" and "chunks" in result:
                existing = list(state.get("retrieved_chunks") or [])
                existing.extend(result["chunks"])
                state["retrieved_chunks"] = existing

        state["tool_calls"] = tool_calls
        state["loop_count"] = loop_count + 1
        state["needs_more_info"] = True  # loop back to agent
        return state

    # ── LLM gave a final answer ────────────────────────────────────────────
    state["tool_calls"] = tool_calls
    state["loop_count"] = loop_count + 1
    state["needs_more_info"] = False
    state["final_response"] = response.content or ""
    return state
