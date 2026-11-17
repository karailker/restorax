from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RESTORAX_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = "sqlite+aiosqlite:///./restorax.db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Storage
    storage_backend: str = "local"  # "local" | "s3"
    storage_local_root: str = "./data"
    # Reject uploads above this size while streaming them to disk (0 = no limit).
    max_upload_mb: int = 4096

    # S3 / MinIO
    s3_endpoint_url: str = "http://localhost:9000"
    s3_bucket: str = "restorax"
    s3_access_key: str = "minioadmin"
    # No default secret: `RESTORAX_S3_SECRET_KEY` must be set for storage_backend="s3".
    # (A validation_alias here once bypassed env_prefix, so the documented
    # RESTORAX_S3_SECRET_KEY was ignored and deployments ran with an empty secret.)
    s3_secret_key: str = Field(default="", description="S3 secret key (required for s3)")

    # ML
    device: str = "cuda"
    model_dir: str = "./models"
    registry_max_loaded: int = 2

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    # Allowed CORS origins for the browser frontend. Dev defaults to the Vite dev
    # server; set explicit origins in production (wildcard is invalid with credentials).
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Observability
    otel_service_name: str = "restorax"
    otel_exporter_otlp_endpoint: str | None = None  # e.g. "http://localhost:4317"
    sentry_dsn: str | None = None


settings = Settings()
