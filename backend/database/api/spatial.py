"""Spatial routes: boundary resolution, locations, boundary catalogs.

Complement the Phase 8 ``/gis`` routes (dataset-location discovery) with the
Phase 10 database-backed spatial layer.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.core.security import ROLE_DATASET_MANAGER
from app.dependencies.enterprise import get_spatial_service
from app.dependencies.security import get_current_user, get_current_user_optional, require_role
from database.api.schemas import ResolveRequest, SpatialLocationCreate
from database.services.spatial_service import SpatialService

router = APIRouter(prefix="/spatial", tags=["spatial"])


@router.get(
    "/resolve",
    summary="Resolve coordinates to an administrative region",
    description="Return the village / taluk / district chain for a coordinate "
    "(cached in Redis).",
)
async def resolve(
    lon: float = Query(ge=-180, le=180),
    lat: float = Query(ge=-90, le=90),
    _: Any = Depends(get_current_user_optional),
    service: SpatialService = Depends(get_spatial_service),
) -> dict[str, Any] | None:
    return await service.resolve_admin_region(lon, lat)


@router.get(
    "/boundaries",
    summary="List administrative boundaries",
)
async def list_boundaries(
    level: str = Query(default="district"),
    parent_id: int | None = Query(default=None),
    _: Any = Depends(get_current_user_optional),
    service: SpatialService = Depends(get_spatial_service),
) -> list[dict[str, Any]]:
    return await service.list_boundaries(level, parent_id=parent_id)


@router.get(
    "/boundaries/counts",
    summary="Boundary counts by level",
)
async def boundary_counts(
    _: Any = Depends(get_current_user_optional),
    service: SpatialService = Depends(get_spatial_service),
) -> dict[str, int]:
    return await service.boundary_counts()


@router.get(
    "/locations",
    summary="List spatial locations",
)
async def list_locations(
    lon: float | None = Query(default=None, ge=-180, le=180),
    lat: float | None = Query(default=None, ge=-90, le=90),
    radius_km: float = Query(default=50.0, ge=1, le=1000),
    limit: int = Query(default=10, ge=1, le=100),
    _: Any = Depends(get_current_user_optional),
    service: SpatialService = Depends(get_spatial_service),
) -> list[dict[str, Any]]:
    if lon is None or lat is None:
        return []
    return await service.nearest_locations(lon, lat, limit=limit, radius_km=radius_km)


@router.post(
    "/locations",
    summary="Create a spatial location (dataset manager or higher)",
)
async def create_location(
    body: SpatialLocationCreate,
    _: Any = Depends(require_role(ROLE_DATASET_MANAGER)),
    service: SpatialService = Depends(get_spatial_service),
) -> dict[str, Any]:
    return await service.create_location(
        name=body.name,
        lon=body.lon,
        lat=body.lat,
        location_type=body.location_type,
        properties=body.properties,
        source=body.source,
    )


@router.post(
    "/resolve/admin",
    summary="Resolve a coordinate payload to an admin region",
)
async def resolve_admin(
    body: ResolveRequest,
    _: Any = Depends(get_current_user),
    service: SpatialService = Depends(get_spatial_service),
) -> dict[str, Any] | None:
    return await service.resolve_admin_region(body.lon, body.lat)
