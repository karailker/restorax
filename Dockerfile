# ── API service image (CPU-only base, no CUDA needed) ─────────────────────────
FROM python:3.14-slim AS base

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
COPY pyproject.toml .
# Install without the heavy torch (API service doesn't run inference)
RUN pip install --no-deps -e ".[dev]" || true
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
    "redis[asyncio]"

# ── App source ────────────────────────────────────────────────────────────────
COPY restorax/ restorax/
COPY configs/ configs/
COPY alembic/ alembic/
COPY alembic.ini .

EXPOSE 8000

CMD ["uvicorn", "restorax.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
