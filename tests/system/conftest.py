"""Shared setup for system tests — reuses the same env as integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ["RESTORAX_DATABASE_URL"] = "sqlite+aiosqlite:///./test_restorax.db"
os.environ["RESTORAX_REDIS_URL"] = "redis://localhost:6379/0"
os.environ["RESTORAX_DEVICE"] = "cpu"
os.environ["RESTORAX_STORAGE_LOCAL_ROOT"] = "/tmp/restorax_test_data"
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "1"


@pytest.fixture(scope="session", autouse=True)
def _cleanup_system_db():
    yield
    Path("test_restorax.db").unlink(missing_ok=True)
