import json
from typing import Any, Dict, Optional
import psycopg2.extras
from langchain_core.messages import messages_to_dict, messages_from_dict

from pipeline.database.connection import get_db_cursor, redis_client


def save_state(conversation_id: str, state: Dict[str, Any], user_id: str | None = None) -> None:
    """
    Upsert the conversation state.
    Persists: chat_history, conversation_summary, user_location, user_state,
    user_country, user_latitude, user_longitude, user_id.
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
        redis_client.setex(redis_key, 86400, json.dumps(redis_state))
        print(f"[Redis] Saved active AgentState -> {redis_key}")
    except Exception as e:
        print(f"[Redis] Error saving state: {e}")

    # 2. SAVE PERMANENT LEDGER TO POSTGRES
    
    # Ensure user_profiles row exists so FK doesn't fail
    if user_id:
        try:
            with get_db_cursor() as cur:
                cur.execute(
                    "INSERT INTO user_profiles(user_id) VALUES (%s) ON CONFLICT DO NOTHING;",
                    (user_id,)
                )
        except Exception:
            pass

    sql = """
    INSERT INTO conversation_states
        (conversation_id, user_id, chat_history, conversation_summary,
         user_location, user_state, user_country, user_latitude, user_longitude, updated_at)
    VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, now())
    ON CONFLICT (conversation_id) DO UPDATE SET
        user_id               = COALESCE(EXCLUDED.user_id, conversation_states.user_id),
        chat_history          = EXCLUDED.chat_history,
        conversation_summary  = COALESCE(EXCLUDED.conversation_summary, conversation_states.conversation_summary),
        user_location  = COALESCE(EXCLUDED.user_location,  conversation_states.user_location),
        user_state     = COALESCE(EXCLUDED.user_state,     conversation_states.user_state),
        user_country   = COALESCE(EXCLUDED.user_country,   conversation_states.user_country),
        user_latitude  = COALESCE(EXCLUDED.user_latitude,  conversation_states.user_latitude),
        user_longitude = COALESCE(EXCLUDED.user_longitude, conversation_states.user_longitude),
        updated_at     = now();
    """
    try:
        with get_db_cursor() as cur:
            cur.execute(sql, (
                conversation_id,
                user_id,
                json.dumps(state.get("chat_history") or []),
                state.get("conversation_summary"),
                state.get("user_location"),
                state.get("user_state"),
                state.get("user_country"),
                state.get("user_latitude"),
                state.get("user_longitude"),
            ))
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
        cached = redis_client.get(redis_key)
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
        cs.user_state,
        cs.user_country,
        cs.user_latitude,
        cs.user_longitude,
        up.name,
        up.language,
        up.location        AS profile_location,
        up.state           AS profile_state,
        up.country         AS profile_country,
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
        with get_db_cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (conversation_id,))
            row = cur.fetchone()

        if row is None:
            return None

        # Session location takes priority over profile location
        loc  = row["user_location"]  or row["profile_location"]
        state_name = row["user_state"] or row["profile_state"]
        country_name = row["user_country"] or row["profile_country"]
        lat  = row["user_latitude"]  or row["profile_latitude"]
        lon  = row["user_longitude"] or row["profile_longitude"]

        return {
            # Session
            "user_id":              row["user_id"],
            "chat_history":         list(row["chat_history"] or []),
            "conversation_summary": row["conversation_summary"],
            "user_location":        loc,
            "user_state":           state_name,
            "user_country":         country_name,
            "user_latitude":        lat,
            "user_longitude":       lon,
            # Profile
            "user_profile": {
                "name":            row["name"],
                "language":        row["language"],
                "location":        loc,
                "state":           state_name,
                "country":         country_name,
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
        with get_db_cursor() as cur:
            cur.execute(
                "DELETE FROM conversation_states WHERE conversation_id = %s;",
                (conversation_id,)
            )
    except Exception as e:
        print(f"[DB] delete_state failed for {conversation_id}: {e}")
