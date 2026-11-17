# ── API service image (CPU-only base, no CUDA needed) ─────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: ffmpeg for PyAV, libgl for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies ──────────────────────────────────────────────────────────────
# The API does not run inference, but it must be able to import the restorer
# catalog: `restorax.api.routers.models` imports every restorer, and those import
# torch / opencv. CPU torch keeps the image small enough to boot.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install \
    fastapi \
    "uvicorn[standard]" \
    celery \
    redis \
    "sqlalchemy[asyncio]" \
    asyncpg \
    aiosqlite \
    alembic \
    pydantic \
    pydantic-settings \
    av \
    numpy \
    pillow \
    pyyaml \
    click \
    rich \
    opencv-python-headless \
    structlog \
    python-multipart \
    prometheus-client \
    opentelemetry-sdk \
    opentelemetry-api \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-celery \
    opentelemetry-instrumentation-sqlalchemy \
    opentelemetry-instrumentation-redis \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-exporter-prometheus

# ── App source ────────────────────────────────────────────────────────────────
COPY restorax/ restorax/
COPY configs/ configs/
COPY alembic/ alembic/
COPY alembic.ini .

EXPOSE 8000

CMD ["uvicorn", "restorax.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
