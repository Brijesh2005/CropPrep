"""Predictions module dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.container import get_model_container
from app.dependencies.database import get_session
from app.modules.predictions.service import PredictionService


def get_prediction_service(
    session: AsyncSession = Depends(get_session),
    model_container: Any = Depends(get_model_container),
) -> PredictionService:
    return PredictionService(
        model_container.resolve("inference_engine"),
        model_container.resolve("explainability_service"),
        session,
    )
