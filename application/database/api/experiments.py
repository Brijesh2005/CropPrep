"""Research experiment routes (admin / analyst)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.core.security import ROLE_ADMIN
from app.dependencies.enterprise import get_experiment_service
from app.dependencies.security import get_current_user, require_role
from database.api.schemas import ExperimentCreate, ExperimentTransition
from database.services.experiment_service import ExperimentService

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post(
    "",
    summary="Create a research experiment (admin)",
)
async def create_experiment(
    body: ExperimentCreate,
    user: Any = Depends(require_role(ROLE_ADMIN)),
    service: ExperimentService = Depends(get_experiment_service),
) -> dict[str, Any]:
    experiment = await service.create(
        name=body.name,
        description=body.description,
        config=body.config,
        created_by=user.id,
        dataset_version_id=body.dataset_version_id,
        model_version_id=body.model_version_id,
    )
    return {"id": experiment.id, "name": experiment.name, "status": experiment.status}


@router.get(
    "",
    summary="List research experiments",
)
async def list_experiments(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _: Any = Depends(get_current_user),
    service: ExperimentService = Depends(get_experiment_service),
) -> dict[str, Any]:
    return await service.list(status=status, limit=limit, offset=offset)


@router.post(
    "/{experiment_id}/start",
    summary="Mark an experiment as started (admin)",
)
async def start_experiment(
    experiment_id: int,
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: ExperimentService = Depends(get_experiment_service),
) -> dict[str, Any]:
    experiment = await service.mark_started(experiment_id)
    if experiment is None:
        return {"ok": False}
    return {"ok": True, "status": experiment.status}


@router.post(
    "/{experiment_id}/finish",
    summary="Mark an experiment as finished (admin)",
)
async def finish_experiment(
    body: ExperimentTransition,
    experiment_id: int,
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: ExperimentService = Depends(get_experiment_service),
) -> dict[str, Any]:
    experiment = await service.mark_finished(experiment_id, metrics=body.metrics)
    if experiment is None:
        return {"ok": False}
    return {"ok": True, "status": experiment.status}


@router.post(
    "/{experiment_id}/fail",
    summary="Mark an experiment as failed (admin)",
)
async def fail_experiment(
    body: ExperimentTransition,
    experiment_id: int,
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: ExperimentService = Depends(get_experiment_service),
) -> dict[str, Any]:
    experiment = await service.mark_failed(experiment_id, error=body.error or "unknown")
    if experiment is None:
        return {"ok": False}
    return {"ok": True, "status": experiment.status}
