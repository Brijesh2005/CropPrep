"""Enterprise prediction-history routes: filtered search + export markers.

``/predictions/history`` (Phase 8) returns the plain paginated history; this
router adds the Phase 10 filtered search without duplicating that path.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query

from app.dependencies.enterprise import get_prediction_history_service
from app.dependencies.security import get_current_user
from database.services.prediction_service import PredictionHistoryService

router = APIRouter(prefix="/predictions", tags=["history-enterprise"])


@router.get(
    "/history/search",
    summary="Search prediction history",
    description="Filtered, paginated history search by crop, season, year, "
    "admin region, date range and minimum confidence.",
)
async def search_history(
    crop: str | None = Query(default=None),
    season: str | None = Query(default=None),
    year: int | None = Query(default=None),
    district: str | None = Query(default=None),
    taluk: str | None = Query(default=None),
    village: str | None = Query(default=None),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    min_confidence: float | None = Query(default=None, ge=0, le=1),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: Any = Depends(get_current_user),
    service: PredictionHistoryService = Depends(get_prediction_history_service),
) -> dict[str, Any]:
    return await service.search(
        user.id,
        crop=crop,
        season=season,
        year=year,
        district=district,
        taluk=taluk,
        village=village,
        date_from=date_from,
        date_to=date_to,
        min_confidence=min_confidence,
        limit=limit,
        offset=offset,
    )
