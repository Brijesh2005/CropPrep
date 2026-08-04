"""GIS routes: locations, nearest, search, boundaries."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.core.security import ROLE_USER
from app.dependencies.security import get_current_user_optional
from app.modules.gis.dependencies import get_gis_service
from app.modules.gis.schemas import BoundaryResponse, LocationResponse, NearestRequest, SearchRequest
from app.modules.gis.service import GISService

router = APIRouter(prefix="/gis", tags=["gis"])


@router.get(
    "/locations",
    response_model=list[LocationResponse],
    summary="List dataset locations",
    description="Return only the locations present in the dataset.",
)
async def list_locations(
    limit: int = Query(default=100, ge=1, le=500),
    _: Any = Depends(get_current_user_optional),
    service: GISService = Depends(get_gis_service),
) -> list[LocationResponse]:
    return service.list(limit=limit)


@router.get(
    "/location/{location_id}",
    response_model=LocationResponse,
    summary="Get one location",
    description="Return a single dataset location by id.",
)
async def get_location(
    location_id: str,
    _: Any = Depends(get_current_user_optional),
    service: GISService = Depends(get_gis_service),
) -> LocationResponse:
    return service.get(location_id)


@router.post(
    "/search",
    response_model=list[LocationResponse],
    summary="Search locations",
    description="Search dataset locations by name / village / district.",
)
async def search_locations(
    body: SearchRequest,
    _: Any = Depends(get_current_user_optional),
    service: GISService = Depends(get_gis_service),
) -> list[LocationResponse]:
    return service.search(body.query, body.limit)


@router.get(
    "/boundaries",
    response_model=list[BoundaryResponse],
    summary="Administrative boundaries",
    description="Summaries of the administrative boundary geometries.",
)
async def boundaries(
    _: Any = Depends(get_current_user_optional),
    service: GISService = Depends(get_gis_service),
) -> list[BoundaryResponse]:
    return service.boundaries()
