"""Explainability routes: explain a location, fetch a stored explanation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import ROLE_USER
from app.dependencies.database import get_session
from app.dependencies.security import get_current_user_optional, require_role
from app.modules.explainability.dependencies import get_explainability_service
from app.modules.explainability.repository import ExplanationRepository
from app.modules.explainability.schemas import ExplainRequest, ExplanationResponse
from app.modules.explainability.service import ExplainabilityService

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.post(
    "",
    response_model=ExplanationResponse,
    summary="Explain a location",
    description="Run the full multimodal explanation for a location.",
)
async def explain(
    body: ExplainRequest,
    _: Any = Depends(require_role(ROLE_USER)),
    service: ExplainabilityService = Depends(get_explainability_service),
) -> ExplanationResponse:
    return service.explain_location(body.lon, body.lat, year=body.year, season=body.season)


@router.get(
    "/{prediction_id}",
    response_model=ExplanationResponse,
    summary="Stored explanation for a prediction",
    description="Return the explanation summary stored for a prediction.",
)
async def get_explanation(
    prediction_id: int,
    _: Any = Depends(get_current_user_optional),
    session: AsyncSession = Depends(get_session),
    service: ExplainabilityService = Depends(get_explainability_service),
) -> ExplanationResponse:
    record = await ExplanationRepository(session).get_by_prediction(prediction_id)
    if record is None:
        return ExplanationResponse(crop="", limitations=["no stored explanation"])
    return ExplanationResponse(**record.summary)
