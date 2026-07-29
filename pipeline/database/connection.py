from contextlib import asynccontextmanager
import asyncio

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.database import db_manager


async def ensure_async_db_ready() -> None:
    """Initialize the shared async SQLAlchemy pool when used outside FastAPI startup."""
    if db_manager.session_factory is None:
        await db_manager.init(auto_create_tables=False)


@asynccontextmanager
async def get_async_db_session() -> AsyncSession:
    """Lease an async DB session from the shared FastAPI asyncpg pool."""
    await ensure_async_db_ready()
    if db_manager.session_factory is None:
        raise RuntimeError("Database session factory is not initialized.")

    async with db_manager.session_factory() as session:
        async with session.begin():
            yield session


def run_async_db(coro):
    """Compatibility helper for old sync scripts/tests. Backend code should await directly."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Cannot run async DB helper from a running event loop; await the async function instead.")


_raw_redis_url = settings.redis_url
redis_client = redis.Redis.from_url(_raw_redis_url, decode_responses=True)
