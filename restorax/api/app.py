from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from restorax.api.middleware import RequestIDMiddleware, TimingMiddleware
from restorax.api.routers import jobs, models, pipelines, ws
from restorax.api.routers.health import router as health_router
from restorax.config import settings
from restorax.logging import configure_logging
from restorax.telemetry import configure_telemetry


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from restorax.db.session import create_tables
    await create_tables()
    yield


def create_app() -> FastAPI:
    configure_logging(app_env=settings.app_env, log_level=settings.log_level)
    configure_telemetry(settings)

    app = FastAPI(
        title="RestoraX",
        description="Modern AI video restoration platform",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(jobs.router)
    app.include_router(models.router)
    app.include_router(pipelines.router)
    app.include_router(ws.router)

    # ── Prometheus /metrics ───────────────────────────────────────────────────
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
