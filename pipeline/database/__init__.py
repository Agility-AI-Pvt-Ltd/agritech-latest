"""
pipeline/database  –  PostgreSQL/Redis-backed state store for the Agritech RAG agent.

Modules:
- connection.py : ThreadedConnectionPool for safe multithreaded PSQL + Redis instance
- schema.py     : DDL and schema init
- profiles.py   : User profile persistent logic
- states.py     : Active conversation state cache and SQL ledger logic
"""

from .schema import init_db
from .profiles import upsert_user_profile, load_user_profile
from .states import save_state, load_state, delete_state

__all__ = [
    "init_db",
    "upsert_user_profile",
    "load_user_profile",
    "save_state",
    "load_state",
    "delete_state"
]
