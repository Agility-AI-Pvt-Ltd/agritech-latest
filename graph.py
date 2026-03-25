"""
graph.py  –  The LangGraph definition + conversation state persistence.

Context management strategy:
  - Keep last RECENT_MSGS_WINDOW messages (5) verbatim.
  - After >5 messages, summarize the older portion ONCE and store it in DB.
  - Summary is NOT regenerated on every turn — only when overflow occurs.
  - Each turn, the system prompt sees: [Summary if any] + [last 5 msgs] + [user query].
"""
from __future__ import annotations

import json
from functools import partial

from langgraph.graph import StateGraph, END

from state import AgentState
from agent import agent_node
from llm_logging import log_llm_call
import db

# ─────────────────────────────────────────────────────────────────────────────
# Context window setting
# ─────────────────────────────────────────────────────────────────────────────
# Keep last N individual messages verbatim; older -> summarize once into DB.
RECENT_MSGS_WINDOW = 10   # 10 messages = 5 user+assistant turns

# ─────────────────────────────────────────────────────────────────────────────
# Sliding window helpers
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARIZE_SYSTEM = """You are a concise summarizer.
Given chat messages from a farming advisory session, write a 5-6 sentence summary
capturing: user identity, farm details, and key questions/answers discussed so far.
Write in third person. No greetings or filler."""


def _summarize_history(
    llm,
    messages: list,
    existing_summary: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """Produce a rolling LLM summary of older messages."""
    from langchain_core.messages import SystemMessage, HumanMessage
    try:
        prefix = f"Previous summary:\n{existing_summary}\n\n" if existing_summary else ""
        turns  = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
        resp   = llm.invoke([
            SystemMessage(content=_SUMMARIZE_SYSTEM),
            HumanMessage(content=f"{prefix}New messages to incorporate:\n{turns}")
        ])
        log_llm_call(
            conversation_id=conversation_id,
            user_id=user_id,
            source="graph.summarize_history",
            request={"messages_count": len(messages)},
            response={"has_content": bool(getattr(resp, "content", ""))},
        )
        return resp.content.strip()
    except Exception as exc:
        log_llm_call(
            conversation_id=conversation_id,
            user_id=user_id,
            source="graph.summarize_history.error",
            request={"messages_count": len(messages)},
            error=str(exc),
        )
        return existing_summary or ""


def _apply_sliding_window(
    llm,
    chat_history: list,
    existing_summary: str | None,
    window: int = RECENT_MSGS_WINDOW,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> tuple:
    """
    Only runs when history exceeds `window` messages.
    Summarizes older messages and keeps only the most recent `window` messages.
    Returns (updated_summary, trimmed_history).
    NO summarization happens on every call — only when overflow occurs.
    """
    if len(chat_history) <= window:
        return existing_summary, chat_history    # Nothing to do yet

    older   = chat_history[:-window]   # messages to compress into summary
    recent  = chat_history[-window:]   # messages to keep verbatim
    print(f"[Context] Summarizing {len(older)} older messages (keeping last {window})...")
    new_summary = _summarize_history(
        llm,
        older,
        existing_summary,
        conversation_id=conversation_id,
        user_id=user_id,
    )
    return new_summary, recent


# ─────────────────────────────────────────────────────────────────────────────
# Graph routing
# ─────────────────────────────────────────────────────────────────────────────

def _should_loop(state: AgentState) -> str:
    if state.get("needs_more_info", False):
        return "agent"
    return END


def build_graph(llm, qdrant_client=None) -> StateGraph:
    from state import InputState
    _agent = partial(agent_node, llm=llm, qdrant_client=qdrant_client)
    g = StateGraph(state_schema=AgentState, input=InputState)
    g.add_node("agent", _agent)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", _should_loop, {"agent": "agent", END: END})
    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Profile extraction (runs after graph resolves)
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """You are a data extraction assistant.
Given a user message and the assistant reply, extract any personal or farm facts the user mentioned.
Return ONLY a valid JSON object (no markdown, no prose) with ONLY the keys that were explicitly mentioned.
Valid keys: name, language, location, latitude, longitude, farm_size_acres, soil_type, crops (list), extra_facts (dict).
CRITICAL: If the user provides ANY city, village, state, or address (even just replying to "Where are you?"), you MUST extract it as the `location` key!
Omit any key where no value was stated. Return {} if nothing new was mentioned."""


def _extract_profile_update(
    llm,
    user_msg: str,
    assistant_msg: str,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Ask the LLM to extract any user facts from a single turn. Returns a patch dict."""
    from langchain_core.messages import SystemMessage, HumanMessage
    try:
        combined = f"User: {user_msg}\nAssistant: {assistant_msg}"
        resp = llm.invoke([
            SystemMessage(content=_EXTRACT_SYSTEM),
            HumanMessage(content=combined),
        ])
        log_llm_call(
            conversation_id=conversation_id,
            user_id=user_id,
            source="graph.extract_profile",
            request={"user_message_length": len(user_msg)},
            response={"has_content": bool(getattr(resp, "content", ""))},
        )
        content = resp.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        patch = json.loads(content)
        return patch if isinstance(patch, dict) else {}
    except Exception as exc:
        log_llm_call(
            conversation_id=conversation_id,
            user_id=user_id,
            source="graph.extract_profile.error",
            request={"user_message_length": len(user_msg)},
            error=str(exc),
        )
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Public run() entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    query: str,
    llm,
    qdrant_client=None,
    chat_history: list | None = None,
    user_location: str | None = None,
    user_latitude: float | None = None,
    user_longitude: float | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> AgentState:
    """
    Run the agentic RAG graph with PostgreSQL state persistence.

    Context management:
      - Loads chat_history + conversation_summary from DB.
      - Applies sliding window: summary is generated ONLY when history >
        RECENT_MSGS_WINDOW (10 msgs / 5 turns); not on every call.
      - LLM receives: profile context + summary + last 10 msgs + current query.
    """
    graph = build_graph(llm=llm, qdrant_client=qdrant_client)

    # ── 1. Load persisted state ─────────────────────────────────────────────
    user_profile: dict | None = None
    conversation_summary: str | None = None

    if conversation_id:
        persisted = db.load_state(conversation_id)
        if persisted:
            n_msgs = len(persisted.get("chat_history", []))
            print(f"[DB] Loaded state for {conversation_id} ({n_msgs} msgs)")
            chat_history         = chat_history   or persisted["chat_history"]
            conversation_summary = persisted.get("conversation_summary")
            user_location        = user_location  or persisted["user_location"]
            user_latitude        = user_latitude  or persisted["user_latitude"]
            user_longitude       = user_longitude or persisted["user_longitude"]
            user_profile         = persisted.get("user_profile")
            user_id              = user_id or persisted.get("user_id")

    if user_id and user_profile is None:
        user_profile = db.load_user_profile(user_id)

    # ── 2. Sliding window  (only triggers when history > 10 messages) ───────
    conversation_summary, trimmed_history = _apply_sliding_window(
        llm,
        chat_history or [],
        conversation_summary,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    # ── 3. Build initial state ──────────────────────────────────────────────
    initial_state: AgentState = {
        "raw_input":           query,
        "conversation_id":     conversation_id,
        "user_id":             user_id,
        "chat_history":        trimmed_history,
        "conversation_summary": conversation_summary,
        "user_location":       user_location,
        "user_latitude":       user_latitude,
        "user_longitude":      user_longitude,
        "user_profile":        user_profile,
        "tool_calls":          [],
        "retrieved_chunks":    [],
        "loop_count":          0,
        "needs_more_info":     False,
        "errors":              [],
    }

    # ── 4. Run graph ────────────────────────────────────────────────────────
    result: AgentState = graph.invoke(initial_state)

    # ── 5. Extract & persist user profile facts FIRST ───────────────────────
    if conversation_id:
        assistant_reply = result.get("final_response", "")
        updated_history = list(trimmed_history)
        updated_history.append({"role": "user",      "content": query})
        updated_history.append({"role": "assistant", "content": assistant_reply})

        resolved_loc = result.get("user_location") or user_location
        resolved_lat = result.get("user_latitude")  or user_latitude
        resolved_lon = result.get("user_longitude") or user_longitude

        if user_id:
            patch = _extract_profile_update(
                llm,
                query,
                assistant_reply,
                conversation_id=conversation_id,
                user_id=user_id,
            )
            if patch:
                if resolved_lat and "latitude" not in patch:
                    patch["latitude"]  = resolved_lat
                    patch["longitude"] = resolved_lon
                db.upsert_user_profile(user_id, patch)
                print(f"[DB] Profile updated for {user_id}: {list(patch.keys())}")
                
                # Merge into active state profile so Redis cache is perfectly synced
                active_profile = dict(result.get("user_profile") or {})
                active_profile.update(patch)
                result["user_profile"] = active_profile

        # ── 6. Persist full state to DB & Redis ─────────────────────────────
        result["chat_history"] = updated_history
        result["conversation_summary"] = conversation_summary
        result["user_location"] = resolved_loc
        result["user_latitude"] = resolved_lat
        result["user_longitude"] = resolved_lon

        db.save_state(conversation_id, dict(result), user_id=user_id)
        print(f"[DB] Saved full AgentState for {conversation_id}")

    return result
