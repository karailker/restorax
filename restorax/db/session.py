from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from restorax.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    echo=settings.app_env == "development",
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables() -> None:
    """Create all tables for development and tests.

    Production must run `alembic upgrade head` instead: `create_all` only creates
    missing tables, so it silently leaves an existing schema un-migrated.
    """
    if settings.app_env == "production":
        logger.warning(
            "app_env=production: skipping create_all — run 'alembic upgrade head' to migrate"
        )
        return

    from restorax.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
