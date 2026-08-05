"""Dataset module dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

from app.dependencies.container import get_model_container
from app.modules.dataset.service import DatasetService


def get_dataset_service(model_container: Any = Depends(get_model_container)) -> DatasetService:
    """Build the dataset service from the model container's dataset manager."""
    return model_container.resolve("dataset_service")
