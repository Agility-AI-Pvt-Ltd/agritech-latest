"""
db.py  –  PostgreSQL-backed state store for the Agritech RAG agent.

Two tables (auto-created on first use):

  user_profiles        — long-lived user facts extracted from conversation
  conversation_states  — per-session chat history & location, linked to user

All functions are synchronous (psycopg2).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
import redis
from dotenv import load_dotenv
from langchain_core.messages import messages_to_dict, messages_from_dict

load_dotenv()

# Strip asyncpg/psycopg2 dialect prefixes for plain psycopg2
_raw_url = os.getenv("DATABASE_URL", "postgresql://krishnakumar:krrish@localhost:5432/agritech")
DSN = (
    _raw_url
    .replace("postgresql+asyncpg://", "postgresql://")
    .replace("postgresql+psycopg2://", "postgresql://")
)

# Redis initialisation
_raw_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis_client = redis.Redis.from_url(_raw_redis_url, decode_responses=True)


# ─────────────────────────────────────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_conn():
    return psycopg2.connect(DSN)


# ─────────────────────────────────────────────────────────────────────────────
# Schema initialisation
# ─────────────────────────────────────────────────────────────────────────────

_DDL_USER_PROFILES = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id          TEXT        PRIMARY KEY,
    name             TEXT,
    language         TEXT,                        -- detected: 'en' | 'hi' | 'hinglish'
    location         TEXT,                        -- human-readable e.g. "Varanasi, UP"
    latitude         FLOAT,
    longitude        FLOAT,
    farm_size_acres  FLOAT,
    soil_type        TEXT,                        -- sandy | loamy | clayey | etc.
    crops            JSONB       NOT NULL DEFAULT '[]',   -- ["spring corn","wheat"]
    extra_facts      JSONB       NOT NULL DEFAULT '{}',   -- catch-all key-value bag
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_DDL_CONVERSATION_STATES = """
CREATE TABLE IF NOT EXISTS conversation_states (
    conversation_id       TEXT        PRIMARY KEY,
    user_id               TEXT        REFERENCES user_profiles(user_id) ON DELETE SET NULL,
    chat_history          JSONB       NOT NULL DEFAULT '[]',
    conversation_summary  TEXT,
    user_location         TEXT,
    user_latitude         FLOAT,
    user_longitude        FLOAT,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def init_db() -> None:
    """Create both tables and apply any pending column migrations."""
    try:
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(_DDL_USER_PROFILES)
                cur.execute(_DDL_CONVERSATION_STATES)
                # Migrations for pre-existing tables
                cur.execute("""
                    ALTER TABLE conversation_states
                    ADD COLUMN IF NOT EXISTS user_id TEXT
                    REFERENCES user_profiles(user_id) ON DELETE SET NULL;
                """)
                cur.execute("""
                    ALTER TABLE conversation_states
                    ADD COLUMN IF NOT EXISTS conversation_summary TEXT;
                """)
        conn.close()
        print("[DB] Tables ready: user_profiles, conversation_states")
    except Exception as e:
        print(f"[DB] init_db failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# User profile helpers
# ─────────────────────────────────────────────────────────────────────────────

def upsert_user_profile(user_id: str, patch: Dict[str, Any]) -> None:
    """
    Merge-update the user_profiles row for user_id.
    Only non-None values in `patch` are applied.
    Scalar fields: name, language, location, latitude, longitude,
                   farm_size_acres, soil_type
    List fields:   crops  (merged, deduplicated)
    Dict fields:   extra_facts (merged)
    """
    if not user_id or not patch:
        return

    try:
        conn = _get_conn()
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Ensure row exists
                cur.execute(
                    "INSERT INTO user_profiles(user_id) VALUES (%s) ON CONFLICT DO NOTHING;",
                    (user_id,)
                )
                # Fetch current values
                cur.execute(
                    "SELECT crops, extra_facts FROM user_profiles WHERE user_id = %s;",
                    (user_id,)
                )
                row = cur.fetchone()
                existing_crops = list(row["crops"] or [])
                existing_extra = dict(row["extra_facts"] or {})

                # Merge crops (list)
                new_crops = patch.get("crops")
                if new_crops:
                    merged_crops = list(dict.fromkeys(existing_crops + list(new_crops)))
                else:
                    merged_crops = existing_crops

                # Merge extra_facts (dict)
                new_extra = patch.get("extra_facts")
                if new_extra:
                    existing_extra.update(new_extra)

                cur.execute(
                    """
                    UPDATE user_profiles SET
                        name            = COALESCE(%s, name),
                        language        = COALESCE(%s, language),
                        location        = COALESCE(%s, location),
                        latitude        = COALESCE(%s, latitude),
                        longitude       = COALESCE(%s, longitude),
                        farm_size_acres = COALESCE(%s, farm_size_acres),
                        soil_type       = COALESCE(%s, soil_type),
                        crops           = %s::jsonb,
                        extra_facts     = %s::jsonb,
                        updated_at      = now()
                    WHERE user_id = %s;
                    """,
                    (
                        patch.get("name"),
                        patch.get("language"),
                        patch.get("location"),
                        patch.get("latitude"),
                        patch.get("longitude"),
                        patch.get("farm_size_acres"),
                        patch.get("soil_type"),
                        json.dumps(merged_crops),
                        json.dumps(existing_extra),
                        user_id,
                    )
                )
        conn.close()
    except Exception as e:
        print(f"[DB] upsert_user_profile failed for {user_id}: {e}")


def load_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Return the full user profile dict, or None if not found."""
    if not user_id:
        return None
    try:
        conn = _get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT user_id, name, language, location, latitude, longitude,
                       farm_size_acres, soil_type, crops, extra_facts, updated_at
                FROM   user_profiles
                WHERE  user_id = %s;
                """,
                (user_id,)
            )
            row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "user_id":         row["user_id"],
            "name":            row["name"],
            "language":        row["language"],
            "location":        row["location"],
            "latitude":        row["latitude"],
            "longitude":       row["longitude"],
            "farm_size_acres": row["farm_size_acres"],
            "soil_type":       row["soil_type"],
            "crops":           list(row["crops"] or []),
            "extra_facts":     dict(row["extra_facts"] or {}),
        }
    except Exception as e:
        print(f"[DB] load_user_profile failed for {user_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Conversation state helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_state(conversation_id: str, state: Dict[str, Any], user_id: str | None = None) -> None:
    """
    Upsert the conversation state.
    Persists: chat_history, conversation_summary, user_location, user_latitude, user_longitude, user_id.
    """
    if not conversation_id:
        return

    # 1. SAVE FAST CACHE TO REDIS
    redis_key = f"agri:state:{conversation_id}"
    try:
        redis_state = dict(state)
        # LangChain messages must be serialized to standard dictionaries using messages_to_dict
        if "messages" in redis_state and redis_state["messages"]:
            redis_state["messages"] = messages_to_dict(redis_state["messages"])
        
        # Save exact current runtime state for 24 hours
        _redis_client.setex(redis_key, 86400, json.dumps(redis_state))
        print(f"[Redis] Saved active AgentState -> {redis_key}")
    except Exception as e:
        print(f"[Redis] Error saving state: {e}")

    # 2. SAVE PERMANENT LEDGER TO POSTGRES

    # Ensure user_profiles row exists so FK doesn't fail
    if user_id:
        try:
            conn = _get_conn()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO user_profiles(user_id) VALUES (%s) ON CONFLICT DO NOTHING;",
                        (user_id,)
                    )
            conn.close()
        except Exception:
            pass

    sql = """
    INSERT INTO conversation_states
        (conversation_id, user_id, chat_history, conversation_summary,
         user_location, user_latitude, user_longitude, updated_at)
    VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s, now())
    ON CONFLICT (conversation_id) DO UPDATE SET
        user_id               = COALESCE(EXCLUDED.user_id, conversation_states.user_id),
        chat_history          = EXCLUDED.chat_history,
        conversation_summary  = COALESCE(EXCLUDED.conversation_summary, conversation_states.conversation_summary),
        user_location  = COALESCE(EXCLUDED.user_location,  conversation_states.user_location),
        user_latitude  = COALESCE(EXCLUDED.user_latitude,  conversation_states.user_latitude),
        user_longitude = COALESCE(EXCLUDED.user_longitude, conversation_states.user_longitude),
        updated_at     = now();
    """
    try:
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    conversation_id,
                    user_id,
                    json.dumps(state.get("chat_history") or []),
                    state.get("conversation_summary"),
                    state.get("user_location"),
                    state.get("user_latitude"),
                    state.get("user_longitude"),
                ))
        conn.close()
    except Exception as e:
        print(f"[DB] save_state failed for {conversation_id}: {e}")


def load_state(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Load persisted conversation state.
    Returns merged dict with all profile + session keys, or None.
    """
    if not conversation_id:
        return None

    # 1. TRY FAST CACHE FROM REDIS
    redis_key = f"agri:state:{conversation_id}"
    try:
        cached = _redis_client.get(redis_key)
        if cached:
            print(f"[Redis] Fast load hit -> {redis_key}")
            data = json.loads(cached)
            # Reconstruct LangGraph-compatible AnyMessage objects
            if "messages" in data and data["messages"]:
                data["messages"] = messages_from_dict(data["messages"])
            return data
    except Exception as e:
        print(f"[Redis] Error loading state: {e}")

    print(f"[Postgres] Cache miss. Fetching permanent ledger -> {conversation_id}")
    # 2. FALLBACK TO POSTGRES
    sql = """
    SELECT
        cs.user_id,
        cs.chat_history,
        cs.conversation_summary,
        cs.user_location,
        cs.user_latitude,
        cs.user_longitude,
        up.name,
        up.language,
        up.location        AS profile_location,
        up.latitude        AS profile_latitude,
        up.longitude       AS profile_longitude,
        up.farm_size_acres,
        up.soil_type,
        up.crops,
        up.extra_facts
    FROM   conversation_states cs
    LEFT JOIN user_profiles    up ON up.user_id = cs.user_id
    WHERE  cs.conversation_id = %s;
    """
    try:
        conn = _get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (conversation_id,))
            row = cur.fetchone()
        conn.close()
        if row is None:
            return None

        # Session location takes priority over profile location
        loc  = row["user_location"]  or row["profile_location"]
        lat  = row["user_latitude"]  or row["profile_latitude"]
        lon  = row["user_longitude"] or row["profile_longitude"]

        return {
            # Session
            "user_id":              row["user_id"],
            "chat_history":         list(row["chat_history"] or []),
            "conversation_summary": row["conversation_summary"],
            "user_location":        loc,
            "user_latitude":        lat,
            "user_longitude":       lon,
            # Profile
            "user_profile": {
                "name":            row["name"],
                "language":        row["language"],
                "location":        loc,
                "latitude":        lat,
                "longitude":       lon,
                "farm_size_acres": row["farm_size_acres"],
                "soil_type":       row["soil_type"],
                "crops":           list(row["crops"] or []),
                "extra_facts":     dict(row["extra_facts"] or {}),
            }
        }
    except Exception as e:
        print(f"[DB] load_state failed for {conversation_id}: {e}")
        return None


def delete_state(conversation_id: str) -> None:
    """Remove a conversation state record."""
    if not conversation_id:
        return
    try:
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_states WHERE conversation_id = %s;",
                    (conversation_id,)
                )
        conn.close()
    except Exception as e:
        print(f"[DB] delete_state failed for {conversation_id}: {e}")
