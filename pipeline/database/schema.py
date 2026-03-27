from pipeline.database.connection import get_db_cursor

_DDL_USER_PROFILES = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id          TEXT        PRIMARY KEY,
    name             TEXT,
    language         TEXT,                        -- detected: 'en' | 'hi' | 'hinglish'
    location         TEXT,                        -- human-readable e.g. "Varanasi, UP"
    state            TEXT,
    country          TEXT,
    sowing_date      TEXT,                        -- e.g. "2025-07-10" for maize sowing
    crop_stage       TEXT,
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


def init_db() -> None:
    """Create both tables and apply any pending column migrations."""
    try:
        with get_db_cursor() as cur:
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
            cur.execute("""
                ALTER TABLE conversation_states
                ADD COLUMN IF NOT EXISTS user_location TEXT;
            """)
            cur.execute("""
                ALTER TABLE conversation_states
                ADD COLUMN IF NOT EXISTS user_state TEXT;
            """)
            cur.execute("""
                ALTER TABLE conversation_states
                ADD COLUMN IF NOT EXISTS user_country TEXT;
            """)
            cur.execute("""
                ALTER TABLE conversation_states
                ADD COLUMN IF NOT EXISTS user_sowing_date TEXT;
            """)
            cur.execute("""
                ALTER TABLE conversation_states
                ADD COLUMN IF NOT EXISTS user_crop_stage TEXT;
            """)
            cur.execute("""
                ALTER TABLE conversation_states
                ADD COLUMN IF NOT EXISTS pending_user_intent TEXT;
            """)
            cur.execute("""
                ALTER TABLE conversation_states
                ADD COLUMN IF NOT EXISTS pending_requirement TEXT;
            """)
            cur.execute("""
                ALTER TABLE conversation_states
                ADD COLUMN IF NOT EXISTS pending_context JSONB NOT NULL DEFAULT '{}';
            """)
            cur.execute("""
                ALTER TABLE conversation_states
                ADD COLUMN IF NOT EXISTS pending_maize_query TEXT;
            """)
            cur.execute("""
                ALTER TABLE user_profiles
                ADD COLUMN IF NOT EXISTS state TEXT;
            """)
            cur.execute("""
                ALTER TABLE user_profiles
                ADD COLUMN IF NOT EXISTS country TEXT;
            """)
            cur.execute("""
                ALTER TABLE user_profiles
                ADD COLUMN IF NOT EXISTS sowing_date TEXT;
            """)
            cur.execute("""
                ALTER TABLE user_profiles
                ADD COLUMN IF NOT EXISTS crop_stage TEXT;
            """)
        print("[DB] Tables ready: user_profiles, conversation_states (Connection Pool)")
    except Exception as e:
        print(f"[DB] init_db failed: {e}")
