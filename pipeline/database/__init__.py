"""
pipeline/database  –  PostgreSQL/Redis-backed state store for the Agritech RAG agent.

Modules:
- connection.py : shared asyncpg SQLAlchemy pool + Redis instance
- schema.py     : DDL and schema init
- profiles.py   : User profile persistent logic
- states.py     : Active conversation state cache and SQL ledger logic
"""

from .schema import init_db, init_db_async
from .profiles import (
    load_user_profile,
    load_user_profile_async,
    upsert_user_profile,
    upsert_user_profile_async,
)
from .states import (
    delete_state,
    delete_state_async,
    load_state,
    load_state_async,
    save_state,
    save_state_async,
)

__all__ = [
    "init_db",
    "init_db_async",
    "upsert_user_profile",
    "upsert_user_profile_async",
    "load_user_profile",
    "load_user_profile_async",
    "save_state",
    "save_state_async",
    "load_state",
    "load_state_async",
    "delete_state",
    "delete_state_async",
]
