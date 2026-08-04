"""Admin routes: dashboard, statistics, retraining."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.core.security import ROLE_ADMIN
from app.dependencies.security import require_role
from app.modules.admin.dependencies import get_admin_service
from app.modules.admin.schemas import Dashboard, RetrainResponse, Statistics
from app.modules.admin.service import AdminService

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/dashboard",
    response_model=Dashboard,
    summary="Admin dashboard",
    description="System status overview (model, dataset, counts, queue).",
)
async def dashboard(
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: AdminService = Depends(get_admin_service),
) -> Dashboard:
    return await service.dashboard()


@router.get(
    "/statistics",
    response_model=Statistics,
    summary="Prediction statistics",
    description="Aggregate prediction counts, confidence and latency.",
)
async def statistics(
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: AdminService = Depends(get_admin_service),
) -> Statistics:
    return await service.statistics()


@router.post(
    "/retrain",
    response_model=RetrainResponse,
    summary="Trigger retraining",
    description="Enqueue a Phase 6 retraining job (admin only).",
)
async def retrain(
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: AdminService = Depends(get_admin_service),
) -> RetrainResponse:
    return RetrainResponse(**await service.retrain())
