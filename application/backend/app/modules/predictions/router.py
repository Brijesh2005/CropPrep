"""Predictions routes: /predict, /predict/location, /predict/map."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.core.security import ROLE_USER
from app.dependencies.security import get_current_user, get_current_user_optional, require_role
from app.modules.predictions.dependencies import get_prediction_service
from app.modules.predictions.schemas import (
    MapPredictionRequest,
    PredictionRequest,
    PredictionResponse,
)
from app.modules.predictions.service import PredictionService

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict for a location",
    description="Run the full inference pipeline (STAM → preprocessing → model) "
                "for a point and store the prediction.",
)
async def predict(
    body: PredictionRequest,
    user: Any = Depends(get_current_user_optional),
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    return await service.predict(body, user_id=user.id if user else None)


@router.post(
    "/location",
    response_model=PredictionResponse,
    summary="Predict for a location (alias)",
    description="Alias of POST /predict for a single point.",
)
async def predict_location(
    body: PredictionRequest,
    user: Any = Depends(get_current_user_optional),
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    return await service.predict(body, user_id=user.id if user else None)


@router.post(
    "/map",
    response_model=list[PredictionResponse],
    summary="Predict for many locations",
    description="Batch predictions for up to 500 points.",
)
async def predict_map(
    body: MapPredictionRequest,
    user: Any = Depends(require_role(ROLE_USER)),
    service: PredictionService = Depends(get_prediction_service),
) -> list[PredictionResponse]:
    return await service.predict_map(body, user_id=user.id)
