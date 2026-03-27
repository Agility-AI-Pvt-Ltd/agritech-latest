"""
pipeline/graph.py  –  LangGraph definition + conversation state persistence.

Context management strategy:
  - Keep last RECENT_MSGS_WINDOW messages (10) verbatim.
  - After > 10 messages, summarize the older portion ONCE and store it in DB.
  - Summary is NOT regenerated on every turn — only when overflow occurs.
  - Each turn, the system prompt sees: [Summary if any] + [last 10 msgs] + [user query].
"""
from __future__ import annotations

import json
from functools import partial

from langgraph.graph import StateGraph, END

from pipeline.state import AgentState
from pipeline.agent import agent_node
from pipeline.logging_utils import log_llm_call
from pipeline.prompts.summarize_prompt import SUMMARIZE_SYSTEM
from pipeline.prompts.profile_prompt import EXTRACT_SYSTEM
from pipeline.safety import NON_PERSISTED_SAFETY_REASONS, safety_node, should_route_from_safety
import pipeline.database as db

# ─────────────────────────────────────────────────────────────────────────────
# Context window setting
# ─────────────────────────────────────────────────────────────────────────────
# Keep last N individual messages verbatim; older → summarize once into DB.
RECENT_MSGS_WINDOW = 10   # 10 messages = 5 user+assistant turns


# ─────────────────────────────────────────────────────────────────────────────
# Sliding window helpers
# ─────────────────────────────────────────────────────────────────────────────

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
            SystemMessage(content=SUMMARIZE_SYSTEM),
            HumanMessage(content=f"{prefix}New messages to incorporate:\n{turns}"),
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
        return existing_summary, chat_history  # Nothing to do yet

    older  = chat_history[:-window]   # messages to compress into summary
    recent = chat_history[-window:]   # messages to keep verbatim
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


def build_graph(llm, qdrant_client=None, safety_llm=None) -> StateGraph:
    """Compile and return the LangGraph agent graph."""
    from pipeline.state import InputState
    _agent = partial(agent_node, llm=llm, qdrant_client=qdrant_client)
    _safety = partial(safety_node, safety_llm=safety_llm)
    g = StateGraph(state_schema=AgentState, input=InputState)
    g.add_node("safety_check", _safety)
    g.add_node("agent", _agent)
    g.set_entry_point("safety_check")
    g.add_conditional_edges("safety_check", should_route_from_safety, {"agent": "agent", END: END})
    g.add_conditional_edges("agent", _should_loop, {"agent": "agent", END: END})
    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Profile extraction (runs after graph resolves)
# ─────────────────────────────────────────────────────────────────────────────

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
            SystemMessage(content=EXTRACT_SYSTEM),
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
    safety_llm=None,
    qdrant_client=None,
    chat_history: list | None = None,
    user_location: str | None = None,
    user_sowing_date: str | None = None,
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
    graph = build_graph(llm=llm, qdrant_client=qdrant_client, safety_llm=safety_llm)

    # ── 1. Load persisted state ─────────────────────────────────────────────
    user_profile: dict | None = None
    conversation_summary: str | None = None

    if conversation_id:
        persisted = db.load_state(conversation_id)
        if persisted:
            n_msgs = len(persisted.get("chat_history", []))
            print(f"[DB] Loaded state for {conversation_id} ({n_msgs} msgs)")
            chat_history         = chat_history  or persisted["chat_history"]
            conversation_summary = persisted.get("conversation_summary")
            user_location        = user_location or persisted["user_location"]
            user_state           = persisted.get("user_state")
            user_country         = persisted.get("user_country")
            user_sowing_date     = user_sowing_date or persisted.get("user_sowing_date")
            pending_user_intent  = persisted.get("pending_user_intent")
            pending_requirement  = persisted.get("pending_requirement")
            pending_context      = persisted.get("pending_context")
            user_latitude        = user_latitude or persisted["user_latitude"]
            user_longitude       = user_longitude or persisted["user_longitude"]
            user_profile         = persisted.get("user_profile")
            user_id              = user_id or persisted.get("user_id")
        else:
            user_state = None
            user_country = None
            pending_user_intent = None
            pending_requirement = None
            pending_context = None
    else:
        user_state = None
        user_country = None
        pending_user_intent = None
        pending_requirement = None
        pending_context = None

    if user_id and user_profile is None:
        user_profile = db.load_user_profile(user_id)
        if user_profile:
            user_state = user_state or user_profile.get("state")
            user_country = user_country or user_profile.get("country")
            user_location = user_location or user_profile.get("location")
            user_sowing_date = user_sowing_date or user_profile.get("sowing_date")
            user_latitude = user_latitude or user_profile.get("latitude")
            user_longitude = user_longitude or user_profile.get("longitude")

    # ── 2. Sliding window (only triggers when history > 10 messages) ────────
    conversation_summary, trimmed_history = _apply_sliding_window(
        llm,
        chat_history or [],
        conversation_summary,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    # ── 3. Build initial state ──────────────────────────────────────────────
    initial_state: AgentState = {
        "raw_input":            query,
        "conversation_id":      conversation_id,
        "user_id":              user_id,
        "user_state":           user_state,
        "user_country":         user_country,
        "user_sowing_date":     user_sowing_date,
        "pending_user_intent":  pending_user_intent,
        "pending_requirement":  pending_requirement,
        "pending_context":      pending_context or {},
        "resume_pending_intent": False,
        "resume_acknowledgment": None,
        "chat_history":         trimmed_history,
        "conversation_summary": conversation_summary,
        "user_location":        user_location,
        "user_latitude":        user_latitude,
        "user_longitude":       user_longitude,
        "user_profile":         user_profile,
        "tool_calls":           [],
        "retrieved_chunks":     [],
        "loop_count":           0,
        "needs_more_info":      False,
        "safety_decision":      None,
        "safety_reason":        None,
        "errors":               [],
    }

    # ── 4. Run graph ────────────────────────────────────────────────────────
    result: AgentState = graph.invoke(initial_state)

    # ── 5. Extract & persist user profile facts FIRST ───────────────────────
    if conversation_id:
        if (
            result.get("safety_decision") == "block"
            and result.get("safety_reason") in _NON_PERSISTED_SAFETY_REASONS
        ):
            print(
                f"[Safety] Skipping state persistence for low-information blocked query: "
                f"{result.get('safety_reason')}"
            )
            return result

        assistant_reply  = result.get("final_response", "")
        updated_history  = list(trimmed_history)
        updated_history.append({"role": "user",      "content": query})
        updated_history.append({"role": "assistant", "content": assistant_reply})

        resolved_loc = result.get("user_location") or user_location
        resolved_state = result.get("user_state") or user_state
        resolved_country = result.get("user_country") or user_country
        resolved_sowing_date = result.get("user_sowing_date") or user_sowing_date
        resolved_pending_user_intent = result.get("pending_user_intent")
        resolved_pending_requirement = result.get("pending_requirement")
        resolved_pending_context = result.get("pending_context") or {}
        resolved_lat = result.get("user_latitude")  or user_latitude
        resolved_lon = result.get("user_longitude") or user_longitude

        if user_id and result.get("safety_decision") != "block":
            patch = _extract_profile_update(
                llm,
                query,
                assistant_reply,
                conversation_id=conversation_id,
                user_id=user_id,
            )
            if patch:
                if resolved_loc and "location" not in patch:
                    patch["location"] = resolved_loc
                if resolved_state and "state" not in patch:
                    patch["state"] = resolved_state
                if resolved_country and "country" not in patch:
                    patch["country"] = resolved_country
                if resolved_sowing_date and "sowing_date" not in patch:
                    patch["sowing_date"] = resolved_sowing_date
                if resolved_lat is not None and resolved_lon is not None and "latitude" not in patch:
                    patch["latitude"]  = resolved_lat
                    patch["longitude"] = resolved_lon
                db.upsert_user_profile(user_id, patch)
                print(f"[DB] Profile updated for {user_id}: {list(patch.keys())}")

                # Merge into active state profile so Redis cache is perfectly synced
                active_profile = dict(result.get("user_profile") or {})
                active_profile.update(patch)
                result["user_profile"] = active_profile

        # ── 6. Persist full state to DB & Redis ──────────────────────────────
        result["chat_history"]         = updated_history
        result["conversation_summary"] = conversation_summary
        result["user_location"]        = resolved_loc
        result["user_state"]           = resolved_state
        result["user_country"]         = resolved_country
        result["user_sowing_date"]     = resolved_sowing_date
        result["pending_user_intent"]  = resolved_pending_user_intent
        result["pending_requirement"]  = resolved_pending_requirement
        result["pending_context"]      = resolved_pending_context
        result["user_latitude"]        = resolved_lat
        result["user_longitude"]       = resolved_lon

        db.save_state(conversation_id, dict(result), user_id=user_id)
        print(f"[DB] Saved full AgentState for {conversation_id}")

    return result
