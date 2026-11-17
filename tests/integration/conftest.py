"""
Shared fixtures for integration tests.

Environment variables (test database, Redis, device, storage root) and the per-session
database reset live in `tests/conftest.py`, before any restorax import.
"""

from __future__ import annotations

import pytest_asyncio


@pytest_asyncio.fixture
async def async_client():
    from httpx import ASGITransport, AsyncClient

    from restorax.api.app import create_app

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
