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

    # S3 / MinIO
    s3_endpoint_url: str = "http://localhost:9000"
    s3_bucket: str = "restorax"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"  # noqa: S105 - local MinIO dev default; override in production

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
