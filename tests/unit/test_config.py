"""Unit tests for Settings (pydantic-settings config)."""

from __future__ import annotations

import pytest

from restorax.config import Settings


class TestSettingsDefaults:
    """Test settings default values when no env vars are set."""

    KEYS = [
        "RESTORAX_DATABASE_URL",
        "RESTORAX_DEVICE",
        "RESTORAX_STORAGE_BACKEND",
        "RESTORAX_REGISTRY_MAX_LOADED",
        "RESTORAX_LOG_LEVEL",
    ]

    @pytest.fixture(autouse=True)
    def clear_restorax_env(self, monkeypatch):
        for key in self.KEYS:
            monkeypatch.delenv(key, raising=False)

    def test_default_database_url_is_sqlite(self):
        s = Settings()
        assert "sqlite" in s.database_url

    def test_default_device_is_cuda(self):
        s = Settings()
        assert s.device == "cuda"

    def test_default_storage_backend(self):
        s = Settings()
        assert s.storage_backend == "local"

    def test_default_registry_max_loaded(self):
        s = Settings()
        assert s.registry_max_loaded == 2

    def test_default_log_level(self):
        s = Settings()
        assert s.log_level == "INFO"


class TestSettingsEnvOverride:
    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("RESTORAX_DEVICE", "cpu")
        s = Settings()
        assert s.device == "cpu"

    def test_redis_url_override(self, monkeypatch):
        monkeypatch.setenv("RESTORAX_REDIS_URL", "redis://myhost:6380/1")
        s = Settings()
        assert "myhost" in s.redis_url

    def test_registry_max_loaded_override(self, monkeypatch):
        monkeypatch.setenv("RESTORAX_REGISTRY_MAX_LOADED", "4")
        s = Settings()
        assert s.registry_max_loaded == 4

    def test_s3_bucket_override(self, monkeypatch):
        monkeypatch.setenv("RESTORAX_S3_BUCKET", "my-bucket")
        s = Settings()
        assert s.s3_bucket == "my-bucket"

    def test_extra_env_vars_ignored(self, monkeypatch):
        monkeypatch.setenv("RESTORAX_UNKNOWN_KEY", "value")
        s = Settings()  # should not raise
        assert s is not None
