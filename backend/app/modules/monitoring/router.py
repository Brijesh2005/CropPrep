"""Monitoring routes: performance metrics."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.core.security import ROLE_ADMIN
from app.dependencies.security import require_role
from app.modules.monitoring.dependencies import get_monitoring_service
from app.modules.monitoring.schemas import MetricsResponse
from app.modules.monitoring.service import MonitoringService

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Runtime metrics",
    description="Request counts, latency and per-path performance (admin).",
)
async def metrics(
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: MonitoringService = Depends(get_monitoring_service),
) -> MetricsResponse:
    return MetricsResponse(metrics=service.metrics())
