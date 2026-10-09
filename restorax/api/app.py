from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

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
        allow_origins=settings.cors_origins,
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

    # ── Exception handlers ────────────────────────────────────────────────────
    from restorax.core.exceptions import (
        JobNotFoundError,
        PipelineConfigError,
        RestoraXError,
        RestorerLoadError,
        RestorerNotFoundError,
    )

    @app.exception_handler(RestorerLoadError)
    async def _handle_restorer_load_error(request: Request, exc: RestorerLoadError) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"error": "restorer_load_error", "message": str(exc)}
        )

    @app.exception_handler(RestorerNotFoundError)
    async def _handle_restorer_not_found(
        request: Request, exc: RestorerNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404, content={"error": "restorer_not_found", "message": str(exc)}
        )

    @app.exception_handler(JobNotFoundError)
    async def _handle_job_not_found(request: Request, exc: JobNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404, content={"error": "job_not_found", "message": str(exc)}
        )

    @app.exception_handler(PipelineConfigError)
    async def _handle_pipeline_config_error(
        request: Request, exc: PipelineConfigError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"error": "pipeline_config_error", "message": str(exc)}
        )

    @app.exception_handler(RestoraXError)
    async def _handle_restorax_error(request: Request, exc: RestoraXError) -> JSONResponse:
        return JSONResponse(
            status_code=500, content={"error": "internal_error", "message": str(exc)}
        )

    # ── Prometheus /metrics ───────────────────────────────────────────────────
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
