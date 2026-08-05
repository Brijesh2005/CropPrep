"""Catalog routes: crops and seasons."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.core.security import ROLE_DATASET_MANAGER
from app.dependencies.enterprise import get_catalog_service
from app.dependencies.security import get_current_user_optional, require_role
from database.api.schemas import CropCreate, SeasonCreate
from database.services.catalog_service import CatalogService

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get(
    "/crops",
    summary="List crops",
    description="Active crop catalog (public).",
)
async def list_crops(
    search: str | None = Query(default=None),
    _: Any = Depends(get_current_user_optional),
    service: CatalogService = Depends(get_catalog_service),
) -> dict[str, Any]:
    return await service.list_crops(search=search)


@router.post(
    "/crops",
    summary="Create a crop (dataset manager or higher)",
)
async def create_crop(
    body: CropCreate,
    _: Any = Depends(require_role(ROLE_DATASET_MANAGER)),
    service: CatalogService = Depends(get_catalog_service),
) -> dict[str, Any]:
    crop = await service.create_crop(
        code=body.code,
        name=body.name,
        scientific_name=body.scientific_name,
        category=body.category,
        description=body.description,
        metadata_=body.metadata,
    )
    return {"id": crop.id, "code": crop.code, "name": crop.name}


@router.get(
    "/seasons",
    summary="List seasons",
    description="Active season catalog (public).",
)
async def list_seasons(
    _: Any = Depends(get_current_user_optional),
    service: CatalogService = Depends(get_catalog_service),
) -> dict[str, Any]:
    return await service.list_seasons()


@router.post(
    "/seasons",
    summary="Create a season (dataset manager or higher)",
)
async def create_season(
    body: SeasonCreate,
    _: Any = Depends(require_role(ROLE_DATASET_MANAGER)),
    service: CatalogService = Depends(get_catalog_service),
) -> dict[str, Any]:
    season = await service.create_season(
        code=body.code,
        name=body.name,
        start_month=body.start_month,
        end_month=body.end_month,
        region=body.region,
        description=body.description,
    )
    return {"id": season.id, "code": season.code, "name": season.name}
