"""FastAPI application factory.

Startup sequence (per the Phase 8 spec):

    load configuration -> init logging -> init dataset manager -> validate
    datasets -> load model -> warm up inference -> build spatial index ->
    register routes -> health checks -> ready.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

from app.api.router import build_api_router
from app.core.app_container import ApplicationContainer
from app.modules.health.router import router as health_router
from app.core.config import Settings, load_settings
from app.core.handlers import register_exception_handlers
from app.core.logging import setup_logging, get_logger, PerformanceTimer
from app.core.tracing import setup_tracing
from app.middleware import register_middleware
from app.services.prometheus import metrics as prometheus_metrics
from app.workers.tasks import dataset_refresh, model_warmup

logger = get_logger("main")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the CropFusion backend application."""
    settings = settings or load_settings()
    setup_logging(settings.log)

    container = ApplicationContainer(settings)
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "CropFusion — location-based multi-crop recommendation and yield "
            "prediction. A domain-driven modular monolith exposing the "
            "Phase 2–7 stack."
        ),
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=_lifespan(container, settings),
    )

    container.wire(app)
    register_middleware(app, settings, container)
    register_exception_handlers(app)
    setup_tracing(app, settings)
    app.include_router(health_router)  # /health, /ready, /live at the root
    app.include_router(build_api_router(settings.api_prefix))

    if settings.monitoring.prometheus_enabled:
        @app.get("/metrics", tags=["meta"], include_in_schema=False)
        async def metrics() -> PlainTextResponse:
            """Prometheus text exposition for scraping."""
            return PlainTextResponse(
                prometheus_metrics.render(),
                media_type=prometheus_metrics.content_type(),
            )

    @app.get("/", tags=["meta"])
    async def root() -> dict[str, Any]:
        return {
            "name": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "openapi": settings.api_prefix + "/openapi.json" if False else "/openapi.json",
        }

    return app


def _lifespan(container: ApplicationContainer, settings: Settings):
    @asynccontextmanager
    async def _ctx(app: FastAPI):
        # 1-2. Configuration + logging are already initialised.
        # 3-7. Build heavy components, load model, warm up, spatial index.
        with PerformanceTimer("startup"):
            await container.initialize()
        engine = container.model.resolve("inference_engine")
        engine.start()
        # Background warmup / dataset refresh (non-blocking).
        if settings.model.warmup:
            await model_warmup(container)
        if settings.dataset.validate_on_startup:
            try:
                await dataset_refresh(container)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("startup dataset refresh failed ({})", exc)
        logger.info("application ready", version=settings.version)
        yield
        # Shutdown.
        await container.shutdown()

    return _ctx


#: Module-level app instance (for uvicorn "app.main:app").
app = create_app()
