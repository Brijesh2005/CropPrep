"""Explainability module dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

from app.dependencies.container import get_model_container
from app.modules.explainability.service import ExplainabilityService


def get_explainability_service(
    model_container: Any = Depends(get_model_container),
) -> ExplainabilityService:
    return model_container.resolve("explainability_service")
