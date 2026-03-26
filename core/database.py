import asyncio
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings


class DatabaseManager:
    """Singleton-style manager for async SQLAlchemy engine + session factory."""

    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def init(self, auto_create_tables: bool = True) -> None:
        """Initialize engine + pool only once for the app lifecycle."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            database_url = settings.async_database_url

            engine_kwargs = {
                "echo": settings.database_echo,
                "pool_pre_ping": settings.database_pool_pre_ping,
            }

            if not database_url.startswith("sqlite"):
                engine_kwargs.update(
                    {
                        "pool_size": settings.database_pool_size,
                        "max_overflow": settings.database_max_overflow,
                        "pool_timeout": settings.database_pool_timeout,
                        "pool_recycle": settings.database_pool_recycle,
                    }
                )

            self.engine = create_async_engine(database_url, **engine_kwargs)
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False,
            )

            if auto_create_tables:
                await self.create_all_tables()

            self._initialized = True

    async def create_all_tables(self) -> None:
        """Create all mapped tables if they do not exist."""
        if self.engine is None:
            raise RuntimeError("Database engine is not initialized")

        # Ensure model modules are imported so metadata contains all tables.
        import models  # noqa: F401
        from models.base import Base

        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Request-scoped session generator."""
        if self.session_factory is None:
            raise RuntimeError("DatabaseManager is not initialized. Call init() at startup.")

        async with self.session_factory() as session:
            yield session

    async def dispose(self) -> None:
        """Dispose engine and release pooled connections."""
        if self.engine is not None:
            await self.engine.dispose()

        self.engine = None
        self.session_factory = None
        self._initialized = False


db_manager = DatabaseManager()
