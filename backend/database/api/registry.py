"""Model + dataset registry routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.core.security import ROLE_DATASET_MANAGER
from app.dependencies.enterprise import get_registry_service
from app.dependencies.security import get_current_user, require_role
from database.api.schemas import DatasetVersionCreate, ModelVersionCreate
from database.services.registry_service import RegistryService

router = APIRouter(prefix="/registry", tags=["registry"])


@router.post(
    "/models",
    summary="Register a model version",
    description="Dataset manager or higher.",
)
async def register_model(
    body: ModelVersionCreate,
    user: Any = Depends(require_role(ROLE_DATASET_MANAGER)),
    service: RegistryService = Depends(get_registry_service),
) -> dict[str, Any]:
    model = await service.register_model(
        name=body.name,
        version=body.version,
        created_by=user.id,
        checkpoint_path=body.checkpoint_path,
        accuracy=body.accuracy,
        loss=body.loss,
        hyperparameters=body.hyperparameters,
        notes=body.notes,
        activate=body.is_active,
    )
    return {"id": model.id, "name": model.name, "version": model.version}


@router.get(
    "/models",
    summary="List model versions",
)
async def list_models(
    name: str | None = Query(default=None),
    _: Any = Depends(get_current_user),
    service: RegistryService = Depends(get_registry_service),
) -> dict[str, Any]:
    return await service.list_models(name)


@router.post(
    "/models/activate",
    summary="Activate a model version",
)
async def activate_model(
    name: str = Query(...),
    version: str = Query(...),
    user: Any = Depends(require_role(ROLE_DATASET_MANAGER)),
    service: RegistryService = Depends(get_registry_service),
) -> dict[str, Any]:
    model = await service.activate_model(name, version)
    if model is None:
        return {"ok": False}
    return {"ok": True, "id": model.id, "is_active": model.is_active}


@router.post(
    "/datasets",
    summary="Register a dataset version",
)
async def register_dataset(
    body: DatasetVersionCreate,
    user: Any = Depends(require_role(ROLE_DATASET_MANAGER)),
    service: RegistryService = Depends(get_registry_service),
) -> dict[str, Any]:
    dataset = await service.register_dataset(
        name=body.name,
        version=body.version,
        source=body.source,
        description=body.description,
        metadata_=body.metadata,
        checksum=body.checksum,
        activate=body.is_active,
    )
    return {"id": dataset.id, "name": dataset.name, "version": dataset.version}


@router.post(
    "/datasets/validate",
    summary="Mark a dataset version as validated",
)
async def validate_dataset(
    name: str = Query(...),
    version: str = Query(...),
    status: str = Query(default="valid"),
    user: Any = Depends(require_role(ROLE_DATASET_MANAGER)),
    service: RegistryService = Depends(get_registry_service),
) -> dict[str, Any]:
    dataset = await service.mark_dataset_validated(name, version, status=status)
    if dataset is None:
        return {"ok": False}
    return {"ok": True, "validation_status": dataset.validation_status}


@router.get(
    "/datasets",
    summary="List dataset versions",
)
async def list_datasets(
    name: str | None = Query(default=None),
    _: Any = Depends(get_current_user),
    service: RegistryService = Depends(get_registry_service),
) -> dict[str, Any]:
    return await service.list_datasets(name)
