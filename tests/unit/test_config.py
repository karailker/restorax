"""Unit tests for Settings (pydantic-settings config)."""

from __future__ import annotations

import pytest

from restorax.config import Settings


def _fresh_settings() -> Settings:
    """Settings with defaults only — the repo `.env` must not leak into unit tests."""
    return Settings(_env_file=None)


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
        s = _fresh_settings()
        assert "sqlite" in s.database_url

    def test_default_device_is_cuda(self):
        s = _fresh_settings()
        assert s.device == "cuda"

    def test_default_storage_backend(self):
        s = _fresh_settings()
        assert s.storage_backend == "local"

    def test_default_registry_max_loaded(self):
        s = _fresh_settings()
        assert s.registry_max_loaded == 2

    def test_default_log_level(self):
        s = _fresh_settings()
        assert s.log_level == "INFO"


class TestSettingsEnvOverride:
    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("RESTORAX_DEVICE", "cpu")
        s = _fresh_settings()
        assert s.device == "cpu"

    def test_redis_url_override(self, monkeypatch):
        monkeypatch.setenv("RESTORAX_REDIS_URL", "redis://myhost:6380/1")
        s = _fresh_settings()
        assert "myhost" in s.redis_url

    def test_registry_max_loaded_override(self, monkeypatch):
        monkeypatch.setenv("RESTORAX_REGISTRY_MAX_LOADED", "4")
        s = _fresh_settings()
        assert s.registry_max_loaded == 4

    def test_s3_bucket_override(self, monkeypatch):
        monkeypatch.setenv("RESTORAX_S3_BUCKET", "my-bucket")
        s = _fresh_settings()
        assert s.s3_bucket == "my-bucket"

    def test_s3_secret_key_reads_prefixed_env(self, monkeypatch):
        """The documented RESTORAX_S3_SECRET_KEY must be honoured (a validation_alias
        once bypassed env_prefix, so production silently ran with an empty secret)."""
        monkeypatch.setenv("RESTORAX_S3_SECRET_KEY", "supersecret")
        s = _fresh_settings()
        assert s.s3_secret_key == "supersecret"

    def test_s3_secret_key_defaults_to_empty(self):
        assert _fresh_settings().s3_secret_key == ""

    def test_extra_env_vars_ignored(self, monkeypatch):
        monkeypatch.setenv("RESTORAX_UNKNOWN_KEY", "value")
        s = _fresh_settings()  # should not raise
        assert s is not None
