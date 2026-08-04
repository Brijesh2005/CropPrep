"""Dataset routes: status, summary, reload."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.dependencies.security import require_role
from app.core.security import ROLE_ADMIN, ROLE_ANALYST, ROLE_USER
from app.modules.dataset.dependencies import get_dataset_service
from app.modules.dataset.schemas import DatasetStatus, DatasetSummary, ReloadResponse
from app.modules.dataset.service import DatasetService

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get(
    "/status",
    response_model=DatasetStatus,
    summary="Dataset status",
    description="Whether the configured catalog is present and the manager is ready.",
)
async def dataset_status(
    _: Any = Depends(require_role(ROLE_USER)),
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetStatus:
    return service.status()


@router.get(
    "/summary",
    response_model=DatasetSummary,
    summary="Dataset summary",
    description="File counts, years and vegetation-index types in the catalog.",
)
async def dataset_summary(
    _: Any = Depends(require_role(ROLE_USER)),
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetSummary:
    return service.summary()


@router.post(
    "/reload",
    response_model=ReloadResponse,
    summary="Reload the dataset metadata",
    description="Force the Dataset Manager to regenerate metadata.",
)
async def dataset_reload(
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: DatasetService = Depends(get_dataset_service),
) -> ReloadResponse:
    return ReloadResponse(**service.reload())
