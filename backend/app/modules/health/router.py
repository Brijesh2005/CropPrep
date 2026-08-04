"""Health routes: /health, /ready, /live."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.dependencies.container import get_container
from app.modules.health.schemas import HealthResponse
from app.modules.health.service import HealthService

router = APIRouter(tags=["health"])


def _service(request: Request) -> HealthService:
    container = get_container(request)
    return container.services.resolve("health_service")


@router.get("/health", response_model=HealthResponse, summary="Full health")
async def health(request: Request) -> HealthResponse:
    return _service(request).full()


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
async def ready(request: Request) -> HealthResponse:
    return await _service(request).ready()


@router.get("/live", response_model=HealthResponse, summary="Liveness probe")
async def live(request: Request) -> HealthResponse:
    return _service(request).live()
