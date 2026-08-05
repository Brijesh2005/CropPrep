"""Crop and season catalog repositories."""

from __future__ import annotations

from sqlalchemy import select

from database.models.catalog import Crop, Season
from database.repositories.base import DataRepository


class CropRepository(DataRepository[Crop]):
    model = Crop

    async def get_by_code(self, code: str) -> Crop | None:
        result = await self.session.execute(select(Crop).where(Crop.code == code))
        return result.scalar_one_or_none()

    async def list_active(self) -> list[Crop]:
        result = await self.session.execute(
            select(Crop).where(Crop.is_active.is_(True)).order_by(Crop.name)
        )
        return list(result.scalars().all())

    async def search(self, query: str, *, limit: int = 20) -> list[Crop]:
        like = f"%{query.lower()}%"
        result = await self.session.execute(
            select(Crop)
            .where(Crop.is_active.is_(True))
            .where(Crop.name.ilike(like) | Crop.code.ilike(like))
            .order_by(Crop.name)
            .limit(limit)
        )
        return list(result.scalars().all())


class SeasonRepository(DataRepository[Season]):
    model = Season

    async def get_by_code(self, code: str) -> Season | None:
        result = await self.session.execute(select(Season).where(Season.code == code))
        return result.scalar_one_or_none()

    async def list_active(self) -> list[Season]:
        result = await self.session.execute(
            select(Season).where(Season.is_active.is_(True)).order_by(Season.name)
        )
        return list(result.scalars().all())
