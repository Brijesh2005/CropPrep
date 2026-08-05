"""Research experiment service."""

from __future__ import annotations

from typing import Any

from database.models.experiments import ResearchExperiment
from database.repositories import ResearchExperimentRepository


class ExperimentService:
    """Create and track research experiments (Phase 6 integration)."""

    def __init__(self, repository: ResearchExperimentRepository) -> None:
        self._repo = repository

    async def create(
        self,
        *,
        name: str,
        config: dict | None = None,
        created_by: int | None = None,
        dataset_version_id: int | None = None,
        model_version_id: int | None = None,
        description: str | None = None,
    ) -> ResearchExperiment:
        return await self._repo.save(
            ResearchExperiment(
                name=name, description=description, config=config or {},
                created_by=created_by, dataset_version_id=dataset_version_id,
                model_version_id=model_version_id, status="queued", metrics={},
            )
        )

    async def list(
        self, *, status: str | None = None, created_by: int | None = None,
        limit: int = 50, offset: int = 0,
    ) -> dict[str, Any]:
        rows, total = await self._repo.list_filtered(
            status=status, created_by=created_by, limit=limit, offset=offset
        )
        return {
            "items": [
                {
                    "id": e.id, "name": e.name, "description": e.description,
                    "config": e.config, "status": e.status, "metrics": e.metrics,
                    "dataset_version_id": e.dataset_version_id,
                    "model_version_id": e.model_version_id,
                    "created_by": e.created_by,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                    "started_at": e.started_at.isoformat() if e.started_at else None,
                    "finished_at": e.finished_at.isoformat() if e.finished_at else None,
                }
                for e in rows
            ],
            "total": total,
        }

    async def mark_started(self, experiment_id: int) -> ResearchExperiment | None:
        experiment = await self._repo.get(experiment_id)
        if experiment is None:
            return None
        await self._repo.mark_started(experiment)
        await self._repo.commit()
        return experiment

    async def mark_finished(self, experiment_id: int, *, metrics: dict | None = None) -> ResearchExperiment | None:
        experiment = await self._repo.get(experiment_id)
        if experiment is None:
            return None
        await self._repo.mark_finished(experiment, metrics=metrics)
        await self._repo.commit()
        return experiment

    async def mark_failed(self, experiment_id: int, *, error: str) -> ResearchExperiment | None:
        experiment = await self._repo.get(experiment_id)
        if experiment is None:
            return None
        await self._repo.mark_failed(experiment, error=error)
        await self._repo.commit()
        return experiment
