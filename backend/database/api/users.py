"""Enterprise user routes: preferences and saved locations.

Complement the Phase 8 ``/users`` routes (``/users/me``, ``/users/profile``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Path

from app.dependencies.enterprise import get_profile_service
from app.dependencies.security import get_current_user
from database.api.schemas import PreferencesUpdate, UserLocationCreate
from database.services.profile import ProfileService

router = APIRouter(prefix="/users", tags=["users-enterprise"])


@router.get(
    "/preferences",
    summary="Get the current user's preferences",
)
async def get_preferences(
    user: Any = Depends(get_current_user),
    service: ProfileService = Depends(get_profile_service),
) -> dict[str, Any]:
    return await service.get_preferences(user.id)


@router.put(
    "/preferences",
    summary="Update the current user's preferences",
)
async def update_preferences(
    body: PreferencesUpdate,
    user: Any = Depends(get_current_user),
    service: ProfileService = Depends(get_profile_service),
) -> dict[str, Any]:
    updates: dict[str, Any] = {"metadata_": {}}
    if body.language is not None:
        updates["preferred_language"] = body.language
    if body.theme is not None:
        updates["theme"] = body.theme
    if body.notification_preferences is not None:
        updates["notifications"] = body.notification_preferences
    if body.units is not None:
        updates["metadata_"]["units"] = body.units
    if body.extra:
        updates["metadata_"].update(body.extra)
    return await service.update_preferences(user.id, updates=updates)


@router.get(
    "/locations",
    summary="List the current user's saved locations",
)
async def list_locations(
    user: Any = Depends(get_current_user),
    service: ProfileService = Depends(get_profile_service),
) -> list[dict[str, Any]]:
    return await service.list_locations(user.id)


@router.post(
    "/locations",
    summary="Save a location",
)
async def add_location(
    body: UserLocationCreate,
    user: Any = Depends(get_current_user),
    service: ProfileService = Depends(get_profile_service),
) -> dict[str, Any]:
    return await service.add_location(
        user.id,
        name=body.name,
        lon=body.lon,
        lat=body.lat,
        is_primary=body.is_default,
        properties=body.properties,
    )


@router.put(
    "/locations/{location_id}/primary",
    summary="Mark a saved location as primary",
)
async def set_primary_location(
    location_id: int = Path(ge=1),
    user: Any = Depends(get_current_user),
    service: ProfileService = Depends(get_profile_service),
) -> dict[str, Any]:
    ok = await service.set_primary_location(user.id, location_id)
    return {"ok": ok}


@router.delete(
    "/locations/{location_id}",
    summary="Delete a saved location",
)
async def delete_location(
    location_id: int = Path(ge=1),
    user: Any = Depends(get_current_user),
    service: ProfileService = Depends(get_profile_service),
) -> dict[str, Any]:
    ok = await service.delete_location(user.id, location_id)
    return {"deleted": ok}
