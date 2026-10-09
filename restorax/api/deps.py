from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from restorax.db.session import get_db as _get_db


# Re-export for FastAPI Depends() use
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in _get_db():
        yield session
