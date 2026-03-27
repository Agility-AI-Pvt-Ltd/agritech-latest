"""
pipeline/agent_context.py  –  Prompt/message assembly helpers for the agent node.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from pipeline.prompts.system_prompt import SYSTEM_PROMPT
from pipeline.state import AgentState


def profile_block(profile: dict | None) -> str:
    """Build a human-readable profile context string to append to the system prompt."""
    if not profile:
        return ""

    lines = ["\n--- Known User Profile ---"]
    if profile.get("name"):
        lines.append(f"Name: {profile['name']}")
    if profile.get("language"):
        lines.append(f"Preferred language: {profile['language']}")
    if profile.get("location"):
        lines.append(f"Location: {profile['location']}")
    if profile.get("state"):
        lines.append(f"State: {profile['state']}")
    if profile.get("country"):
        lines.append(f"Country: {profile['country']}")
    if profile.get("sowing_date"):
        lines.append(f"Maize sowing date: {profile['sowing_date']}")
    if profile.get("crop_stage"):
        lines.append(f"Current maize crop stage: {profile['crop_stage']}")
    if profile.get("latitude") is not None and profile.get("longitude") is not None:
        lines.append(f"Coordinates: {profile['latitude']}, {profile['longitude']}")
    if profile.get("farm_size_acres"):
        lines.append(f"Farm size: {profile['farm_size_acres']} acres")
    if profile.get("soil_type"):
        lines.append(f"Soil type: {profile['soil_type']}")
    if profile.get("crops"):
        lines.append(f"Crops: {', '.join(profile['crops'])}")

    extra = profile.get("extra_facts") or {}
    for key, value in extra.items():
        lines.append(f"{key}: {value}")

    lines.append("Use this profile to personalize your responses without asking again.")
    lines.append("--- End of Profile ---")
    return "\n".join(lines)


def build_messages(state: AgentState) -> list:
    """Construct the full message list for the LLM call."""
    profile_ctx = profile_block(state.get("user_profile"))
    messages = [SystemMessage(content=SYSTEM_PROMPT + profile_ctx)]

    summary = state.get("conversation_summary")
    if summary:
        messages.append(
            SystemMessage(
                content=f"[Earlier conversation summary — treat as established context]\n{summary}"
            )
        )

    if state.get("resume_pending_intent"):
        ack = state.get("resume_acknowledgment") or ""
        messages.append(
            SystemMessage(
                content=(
                    "The user has just provided a missing detail for an earlier unanswered request. "
                    f"Briefly acknowledge it using this exact fact if natural: {ack} "
                    "Then answer the original user question directly in the same response. "
                    "Do not ask again for the same missing detail."
                )
            )
        )

    tool_calls = state.get("tool_calls") or []
    tool_names = [call.get("tool") for call in tool_calls]
    if "faq_search_by_crop_stage" in tool_names and "rag_search" in tool_names:
        messages.append(
            SystemMessage(
                content=(
                    "When both faq_search_by_crop_stage and rag_search results are available, "
                    "present the FAQ guidance FIRST as the primary answer. "
                    "Then add supporting or supplementary points from the broader RAG/manual results. "
                    "Do not ignore the FAQ answer and do not reverse this order."
                )
            )
        )

    history = state.get("chat_history") or []
    if history:
        messages.append(
            SystemMessage(
                content=(
                    "IMPORTANT: The conversation history below is ordered oldest→newest. "
                    "More recent messages carry HIGHER priority. When information conflicts, "
                    "always prefer the most recent statement."
                )
            )
        )

    pairs: list[tuple] = []
    i = 0
    while i < len(history):
        user_msg = history[i] if i < len(history) else None
        asst_msg = history[i + 1] if i + 1 < len(history) else None
        pairs.append((user_msg, asst_msg))
        i += 2

    total_turns = len(pairs)
    for turn_idx, (user_msg, asst_msg) in enumerate(pairs, start=1):
        age = total_turns - turn_idx
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

    messages.append(HumanMessage(content=state.get("raw_input", "")))

    for call in tool_calls:
        tool_name = call.get("tool")
        params = call.get("params", {})
        result = call.get("result", {})
        call_id = call.get("id", f"call_{tool_name}")

        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": params, "id": call_id}],
            )
        )
        messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call_id))

    return messages
