from contextlib import contextmanager

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import redis
from core.config import settings


DSN = settings.sync_database_url


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
        with conn:
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                yield cur


_raw_redis_url = settings.redis_url
redis_client = redis.Redis.from_url(_raw_redis_url, decode_responses=True)
