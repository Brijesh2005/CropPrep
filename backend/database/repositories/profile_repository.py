"""User profile repositories: preferences and saved locations."""

from __future__ import annotations

from sqlalchemy import select

from database.models.profile import UserLocation, UserPreference
from database.repositories.base import DataRepository


class UserPreferenceRepository(DataRepository[UserPreference]):
    model = UserPreference

    async def get_for_user(self, user_id: int) -> UserPreference | None:
        result = await self.session.execute(
            select(UserPreference).where(UserPreference.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_or_create(self, user_id: int) -> UserPreference:
        existing = await self.get_for_user(user_id)
        if existing is not None:
            return existing
        return UserPreference(user_id=user_id)


class UserLocationRepository(DataRepository[UserLocation]):
    model = UserLocation

    async def list_for_user(self, user_id: int) -> list[UserLocation]:
        result = await self.session.execute(
            select(UserLocation).where(UserLocation.user_id == user_id).order_by(UserLocation.id)
        )
        return list(result.scalars().all())

    async def get_for_user(self, user_id: int, location_id: int) -> UserLocation | None:
        result = await self.session.execute(
            select(UserLocation).where(
                UserLocation.id == location_id, UserLocation.user_id == user_id
            )
        )
        return result.scalar_one_or_none()

    async def clear_primary(self, user_id: int, *, keep_id: int | None = None) -> None:
        result = await self.session.execute(
            select(UserLocation).where(UserLocation.user_id == user_id)
        )
        for loc in result.scalars().all():
            if keep_id is None or loc.id != keep_id:
                loc.is_primary = False
        await self.session.flush()
