from sqlalchemy import text

from pipeline.database.connection import get_async_db_session, run_async_db

_DDL_USER_PROFILES = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id          TEXT        PRIMARY KEY,
    name             TEXT,
    language         TEXT,
    location         TEXT,
    state            TEXT,
    country          TEXT,
    sowing_date      TEXT,
    crop_stage       TEXT,
    latitude         FLOAT,
    longitude        FLOAT,
    farm_size_acres  FLOAT,
    soil_type        TEXT,
    crops            JSONB       NOT NULL DEFAULT '[]',
    extra_facts      JSONB       NOT NULL DEFAULT '{}',
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
    user_state            TEXT,
    user_country          TEXT,
    user_sowing_date      TEXT,
    user_crop_stage       TEXT,
    pending_user_intent   TEXT,
    pending_requirement   TEXT,
    pending_context       JSONB       NOT NULL DEFAULT '{}',
    pending_maize_query   TEXT,
    user_latitude         FLOAT,
    user_longitude        FLOAT,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_MIGRATIONS = [
    """
    ALTER TABLE conversation_states
    ADD COLUMN IF NOT EXISTS user_id TEXT
    REFERENCES user_profiles(user_id) ON DELETE SET NULL;
    """,
    "ALTER TABLE conversation_states ADD COLUMN IF NOT EXISTS conversation_summary TEXT;",
    "ALTER TABLE conversation_states ADD COLUMN IF NOT EXISTS user_location TEXT;",
    "ALTER TABLE conversation_states ADD COLUMN IF NOT EXISTS user_state TEXT;",
    "ALTER TABLE conversation_states ADD COLUMN IF NOT EXISTS user_country TEXT;",
    "ALTER TABLE conversation_states ADD COLUMN IF NOT EXISTS user_sowing_date TEXT;",
    "ALTER TABLE conversation_states ADD COLUMN IF NOT EXISTS user_crop_stage TEXT;",
    "ALTER TABLE conversation_states ADD COLUMN IF NOT EXISTS pending_user_intent TEXT;",
    "ALTER TABLE conversation_states ADD COLUMN IF NOT EXISTS pending_requirement TEXT;",
    "ALTER TABLE conversation_states ADD COLUMN IF NOT EXISTS pending_context JSONB NOT NULL DEFAULT '{}';",
    "ALTER TABLE conversation_states ADD COLUMN IF NOT EXISTS pending_maize_query TEXT;",
    "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS state TEXT;",
    "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS country TEXT;",
    "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS sowing_date TEXT;",
    "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS crop_stage TEXT;",
]


async def init_db_async() -> None:
    """Create both tables and apply any pending column migrations using asyncpg."""
    try:
        async with get_async_db_session() as session:
            await session.execute(text(_DDL_USER_PROFILES))
            await session.execute(text(_DDL_CONVERSATION_STATES))
            for migration in _MIGRATIONS:
                await session.execute(text(migration))
        print("[DB] Tables ready: user_profiles, conversation_states (async pool)")
    except Exception as e:
        print(f"[DB] init_db failed: {e}")


def init_db() -> None:
    """Sync compatibility wrapper for scripts/tests outside an event loop."""
    return run_async_db(init_db_async())
