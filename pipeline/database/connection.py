from contextlib import contextmanager

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import redis
from core.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL Connection Pool
# ─────────────────────────────────────────────────────────────────────────────
DSN = settings.sync_database_url

# Initialize a thread-safe connection pool for PostgreSQL
# minconn=1, maxconn=10 (adjust based on production needs)
try:
    pg_pool = ThreadedConnectionPool(1, 10, DSN)
except Exception as e:
    print(f"[DB Pool] Initialization error: {e}")
    pg_pool = None


@contextmanager
def get_db_connection():
    """Context manager to safely checkout and release a database connection."""
    if not pg_pool:
        raise RuntimeError("Database connection pool is not initialized.")
    
    conn = pg_pool.getconn()
    try:
        yield conn
    finally:
        # Always return the connection to the pool
        pg_pool.putconn(conn)


@contextmanager
def get_db_cursor(cursor_factory=None):
    """Context manager to lease a connection, provide a cursor, and commit transactions automatically."""
    with get_db_connection() as conn:
        # The `with conn` block automatically commits if no exceptions occur, or rolls back.
        with conn:
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                yield cur


# ─────────────────────────────────────────────────────────────────────────────
# Redis Client Configuration
# ─────────────────────────────────────────────────────────────────────────────
_raw_redis_url = settings.redis_url

# Redis connection pooling is built-in to the redis-py library automatically
redis_client = redis.Redis.from_url(_raw_redis_url, decode_responses=True)
