"""Research experiment and app-configuration repositories."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from database.models.configuration import AppConfiguration
from database.models.experiments import ResearchExperiment
from database.repositories.base import DataRepository


class ResearchExperimentRepository(DataRepository[ResearchExperiment]):
    model = ResearchExperiment

    async def list_filtered(
        self,
        *,
        status: str | None = None,
        created_by: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ResearchExperiment], int]:
        stmt = select(ResearchExperiment)
        if status:
            stmt = stmt.where(ResearchExperiment.status == status)
        if created_by is not None:
            stmt = stmt.where(ResearchExperiment.created_by == created_by)
        stmt = stmt.order_by(ResearchExperiment.created_at.desc())
        return await self.paginate(stmt, limit=limit, offset=offset)

    async def mark_started(self, experiment: ResearchExperiment) -> None:
        experiment.status = "running"
        experiment.started_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def mark_finished(self, experiment: ResearchExperiment, *, metrics: dict | None = None) -> None:
        experiment.status = "completed"
        experiment.finished_at = datetime.now(timezone.utc)
        if metrics:
            experiment.metrics = {**(experiment.metrics or {}), **metrics}
        await self.session.flush()

    async def mark_failed(self, experiment: ResearchExperiment, *, error: str) -> None:
        experiment.status = "failed"
        experiment.finished_at = datetime.now(timezone.utc)
        experiment.metrics = {**(experiment.metrics or {}), "error": error}
        await self.session.flush()


class AppConfigurationRepository(DataRepository[AppConfiguration]):
    model = AppConfiguration

    async def get_by_key(self, key: str) -> AppConfiguration | None:
        result = await self.session.execute(
            select(AppConfiguration).where(AppConfiguration.key == key)
        )
        return result.scalar_one_or_none()

    async def list_by_category(self, category: str | None = None) -> list[AppConfiguration]:
        stmt = select(AppConfiguration).order_by(AppConfiguration.category, AppConfiguration.key)
        if category:
            stmt = stmt.where(AppConfiguration.category == category)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def set_value(
        self, *, key: str, value: dict | None, category: str | None = None,
        description: str | None = None, is_secret: bool = False, updated_by: int | None = None,
    ) -> AppConfiguration:
        config = await self.get_by_key(key)
        if config is None:
            config = AppConfiguration(key=key, version=1)
            self.session.add(config)
        else:
            config.version += 1
        config.value = value
        config.category = category or config.category
        config.description = description or config.description
        config.is_secret = is_secret
        config.updated_by = updated_by
        await self.session.flush()
        return config
