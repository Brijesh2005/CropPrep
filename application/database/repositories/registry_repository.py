"""Model and dataset registry repositories."""

from __future__ import annotations

from sqlalchemy import select

from database.models.registry import DatasetVersion, ModelVersion
from database.repositories.base import DataRepository


class ModelVersionRepository(DataRepository[ModelVersion]):
    model = ModelVersion

    async def get_by_name_version(self, name: str, version: str) -> ModelVersion | None:
        result = await self.session.execute(
            select(ModelVersion).where(
                ModelVersion.name == name, ModelVersion.version == version
            )
        )
        return result.scalar_one_or_none()

    async def get_active(self, name: str | None = None) -> ModelVersion | None:
        stmt = select(ModelVersion).where(
            ModelVersion.is_active.is_(True), ModelVersion.status == "active"
        )
        if name:
            stmt = stmt.where(ModelVersion.name == name)
        result = await self.session.execute(stmt.order_by(ModelVersion.training_date.desc()))
        return result.scalars().first()

    async def list_versions(self, name: str | None = None) -> list[ModelVersion]:
        stmt = select(ModelVersion).order_by(ModelVersion.training_date.desc())
        if name:
            stmt = stmt.where(ModelVersion.name == name)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def deactivate_all(self, name: str, *, except_id: int | None = None) -> int:
        result = await self.session.execute(select(ModelVersion).where(ModelVersion.name == name))
        updated = 0
        for row in result.scalars().all():
            if except_id is None or row.id != except_id:
                row.is_active = False
                row.status = "archived"
                updated += 1
        await self.session.flush()
        return updated


class DatasetVersionRepository(DataRepository[DatasetVersion]):
    model = DatasetVersion

    async def get_by_name_version(self, name: str, version: str) -> DatasetVersion | None:
        result = await self.session.execute(
            select(DatasetVersion).where(
                DatasetVersion.name == name, DatasetVersion.version == version
            )
        )
        return result.scalar_one_or_none()

    async def get_active(self, name: str | None = None) -> DatasetVersion | None:
        stmt = select(DatasetVersion).where(DatasetVersion.is_active.is_(True))
        if name:
            stmt = stmt.where(DatasetVersion.name == name)
        result = await self.session.execute(stmt.order_by(DatasetVersion.downloaded_date.desc()))
        return result.scalars().first()

    async def list_versions(self, name: str | None = None) -> list[DatasetVersion]:
        stmt = select(DatasetVersion).order_by(DatasetVersion.downloaded_date.desc())
        if name:
            stmt = stmt.where(DatasetVersion.name == name)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
