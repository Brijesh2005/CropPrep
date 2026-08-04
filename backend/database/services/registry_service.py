"""Model and dataset registry service."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from database.models.registry import DatasetVersion, ModelVersion
from database.repositories import DatasetVersionRepository, ModelVersionRepository


class RegistryService:
    """Manage model versions and dataset versions."""

    def __init__(
        self, models: ModelVersionRepository, datasets: DatasetVersionRepository
    ) -> None:
        self._models = models
        self._datasets = datasets

    # ------------------------------------------------------------------ #
    # Model versions
    # ------------------------------------------------------------------ #
    async def register_model(
        self,
        *,
        name: str,
        version: str,
        created_by: int | None = None,
        checkpoint_path: str | None = None,
        accuracy: float | None = None,
        loss: float | None = None,
        hyperparameters: dict | None = None,
        metrics: dict | None = None,
        notes: str | None = None,
        activate: bool = False,
    ) -> ModelVersion:
        existing = await self._models.get_by_name_version(name, version)
        if existing is not None:
            raise ValueError(f"model {name}@{version} is already registered")
        if activate:
            await self._models.deactivate_all(name)
        model = await self._models.save(
            ModelVersion(
                name=name, version=version, created_by=created_by,
                checkpoint_path=checkpoint_path, accuracy=accuracy, loss=loss,
                hyperparameters=hyperparameters or {},
                metrics=metrics or {}, notes=notes,
                status="active" if activate else "draft",
                is_active=activate, training_date=datetime.now(),
            )
        )
        return model

    async def activate_model(self, name: str, version: str) -> ModelVersion | None:
        model = await self._models.get_by_name_version(name, version)
        if model is None:
            return None
        await self._models.deactivate_all(name, except_id=model.id)
        model.is_active = True
        model.status = "active"
        await self._models.commit()
        return model

    async def list_models(self, name: str | None = None) -> dict[str, Any]:
        rows = await self._models.list_versions(name)
        return {
            "items": [
                {
                    "id": m.id, "name": m.name, "version": m.version,
                    "training_date": m.training_date.isoformat() if m.training_date else None,
                    "accuracy": m.accuracy, "loss": m.loss,
                    "status": m.status, "is_active": m.is_active,
                    "checkpoint_path": m.checkpoint_path,
                }
                for m in rows
            ]
        }

    async def get_active_model(self, name: str | None = None) -> ModelVersion | None:
        return await self._models.get_active(name)

    # ------------------------------------------------------------------ #
    # Dataset versions
    # ------------------------------------------------------------------ #
    async def register_dataset(
        self,
        *,
        name: str,
        version: str,
        source: str | None = None,
        description: str | None = None,
        metadata_: dict | None = None,
        checksum: str | None = None,
        checksum_algorithm: str | None = None,
        file_count: int | None = None,
        size_bytes: int | None = None,
        schema: dict | None = None,
        activate: bool = False,
    ) -> DatasetVersion:
        existing = await self._datasets.get_by_name_version(name, version)
        if existing is not None:
            raise ValueError(f"dataset {name}@{version} is already registered")
        if activate:
            await self._set_single_active(name, except_id=None)
        dataset = await self._datasets.save(
            DatasetVersion(
                name=name, version=version, source=source, description=description,
                metadata_=metadata_ or {}, validation_status="pending",
                checksum=checksum, checksum_algorithm=checksum_algorithm,
                file_count=file_count, size_bytes=size_bytes,
                schema=schema or {}, is_active=activate,
                downloaded_date=datetime.now(),
            )
        )
        return dataset

    async def _set_single_active(self, name: str, *, except_id: int | None) -> None:
        for row in await self._datasets.list_versions(name):
            if except_id is None or row.id != except_id:
                row.is_active = False
        await self._datasets.commit()

    async def mark_dataset_validated(
        self, name: str, version: str, *, status: str, metadata_: dict | None = None
    ) -> DatasetVersion | None:
        dataset = await self._datasets.get_by_name_version(name, version)
        if dataset is None:
            return None
        dataset.validation_status = status
        if metadata_:
            dataset.metadata_ = {**(dataset.metadata_ or {}), **metadata_}
        await self._datasets.commit()
        return dataset

    async def list_datasets(self, name: str | None = None) -> dict[str, Any]:
        rows = await self._datasets.list_versions(name)
        return {
            "items": [
                {
                    "id": d.id, "name": d.name, "version": d.version,
                    "source": d.source, "validation_status": d.validation_status,
                    "is_active": d.is_active, "checksum": d.checksum,
                    "file_count": d.file_count, "size_bytes": d.size_bytes,
                    "downloaded_date": d.downloaded_date.isoformat() if d.downloaded_date else None,
                }
                for d in rows
            ]
        }

    async def get_active_dataset(self, name: str | None = None) -> DatasetVersion | None:
        return await self._datasets.get_active(name)
