"""
pipeline/state.py  –  LangGraph state definitions for the Kisan Mitra agent.
"""
from typing import Optional, List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class InputState(TypedDict, total=False):
    """The subset of AgentState editable in LangGraph Studio."""
    messages: Annotated[list[AnyMessage], add_messages]
    raw_input: str
    conversation_id: str
    user_id: str
    user_location: str
    user_latitude: float
    user_longitude: float
    user_state: str
    user_country: str
    user_sowing_date: str
    user_crop_stage: str
    pending_user_intent: str
    pending_requirement: str
    pending_context: Dict[str, Any]
    resume_pending_intent: bool
    resume_acknowledgment: str
    user_profile: Dict[str, Any]
    chat_history: List[Dict[str, str]]
    conversation_summary: str


class AgentState(TypedDict, total=False):
    # ── Studio Chat Integration ────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]

    # ── Input ──────────────────────────────────────────────
    raw_input: str
    conversation_id: Optional[str]
    user_id: Optional[str]
    language: Optional[str]           # detected language (e.g. "hi", "en")
    user_location: Optional[str]
    user_latitude: Optional[float]
    user_longitude: Optional[float]
    user_state: Optional[str]
    user_country: Optional[str]
    user_sowing_date: Optional[str]
    user_crop_stage: Optional[str]
    pending_user_intent: Optional[str]
    pending_requirement: Optional[str]
    pending_context: Optional[Dict[str, Any]]
    resume_pending_intent: Optional[bool]
    resume_acknowledgment: Optional[str]

    # ── User Profile (loaded from user_profiles table) ─────
    user_profile: Optional[Dict[str, Any]]   # name, crops, farm_size, etc.

    # ── Memory / History ───────────────────────────────────
    chat_history:         List[Dict[str, str]]   # [{role, content}, ...] — last N msgs
    conversation_summary: Optional[str]          # LLM-generated rolling summary of older msgs

    # ── Agent Loop ─────────────────────────────────────────
    tool_calls: List[Dict[str, Any]]     # list of {tool, params, result}
    needs_more_info: bool                # agent wants another tool call
    loop_count: int                      # guard against infinite loops

    # ── RAG ────────────────────────────────────────────────
    retrieved_chunks: Optional[List[Dict]]   # raw vector search hits
    rag_context: Optional[str]               # fused context string

    # ── Tool Results ───────────────────────────────────────
    tool_result: Optional[Dict]          # latest tool result

    # ── Output ─────────────────────────────────────────────
    final_response: Optional[str]
    safety_decision: Optional[str]
    safety_reason: Optional[str]
    skip_profile_extraction: Optional[bool]

    # ── Meta ───────────────────────────────────────────────
    errors: List[str]
