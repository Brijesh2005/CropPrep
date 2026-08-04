"""Profile service: profile updates, preferences, saved locations."""

from __future__ import annotations

from typing import Any

from app.models.user import User
from database.repositories import UserLocationRepository, UserPreferenceRepository, UserRepository
from database.services.geo import validate_coordinates


class ProfileService:
    """Manage the enterprise user profile (profile + preferences + locations)."""

    def __init__(
        self,
        users: UserRepository,
        preferences: UserPreferenceRepository,
        locations: UserLocationRepository,
    ) -> None:
        self._users = users
        self._preferences = preferences
        self._locations = locations

    async def update_profile(self, user: User, *, full_name: str | None = None, phone: str | None = None) -> User:
        return await self._users.update_profile(user, full_name=full_name, phone=phone)

    # ------------------------------------------------------------------ #
    # Preferences
    # ------------------------------------------------------------------ #
    async def get_preferences(self, user_id: int) -> dict[str, Any]:
        pref = await self._preferences.get_or_create(user_id)
        return {
            "preferred_language": pref.preferred_language,
            "theme": pref.theme,
            "timezone": pref.timezone,
            "notifications": pref.notifications,
            "metadata": pref.metadata_,
        }

    async def update_preferences(self, user_id: int, *, updates: dict[str, Any]) -> dict[str, Any]:
        pref = await self._preferences.get_or_create(user_id)
        allowed = {"preferred_language", "theme", "timezone", "notifications", "metadata_"}
        for key, value in updates.items():
            if key == "metadata":
                key = "metadata_"
            if key in allowed:
                setattr(pref, key, value)
        await self._preferences.add(pref)
        await self._preferences.commit()
        return await self.get_preferences(user_id)

    # ------------------------------------------------------------------ #
    # Saved locations
    # ------------------------------------------------------------------ #
    async def list_locations(self, user_id: int) -> list[dict[str, Any]]:
        return [
            {
                "id": loc.id,
                "name": loc.name,
                "lon": loc.lon,
                "lat": loc.lat,
                "is_primary": loc.is_primary,
                "properties": loc.properties,
            }
            for loc in await self._locations.list_for_user(user_id)
        ]

    async def add_location(
        self, user_id: int, *, name: str, lon: float, lat: float,
        is_primary: bool = False, properties: dict | None = None,
    ) -> dict[str, Any]:
        lon, lat = validate_coordinates(lon, lat)
        if is_primary:
            await self._locations.clear_primary(user_id)
        loc = await self._locations.save(
            self._locations.model(
                user_id=user_id, name=name, lon=lon, lat=lat,
                is_primary=is_primary, properties=properties or {},
            )
        )
        return {
            "id": loc.id, "name": loc.name, "lon": loc.lon, "lat": loc.lat,
            "is_primary": loc.is_primary, "properties": loc.properties,
        }

    async def delete_location(self, user_id: int, location_id: int) -> bool:
        loc = await self._locations.get_for_user(user_id, location_id)
        if loc is None:
            return False
        await self._locations.delete(loc)
        await self._locations.commit()
        return True

    async def set_primary_location(self, user_id: int, location_id: int) -> bool:
        loc = await self._locations.get_for_user(user_id, location_id)
        if loc is None:
            return False
        await self._locations.clear_primary(user_id, keep_id=location_id)
        loc.is_primary = True
        await self._locations.commit()
        return True
