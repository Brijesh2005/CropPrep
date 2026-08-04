"""History routes: paginated past predictions for the current user."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.dependencies.security import get_current_user
from app.modules.history.dependencies import get_history_service
from app.modules.history.schemas import HistoryPage
from app.modules.history.service import HistoryService

router = APIRouter(prefix="/predictions", tags=["history"])


@router.get(
    "/history",
    response_model=HistoryPage,
    summary="Prediction history",
    description="Paginated past predictions for the authenticated user.",
)
async def history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: Any = Depends(get_current_user),
    service: HistoryService = Depends(get_history_service),
) -> HistoryPage:
    return await service.list_for_user(user.id, limit=limit, offset=offset)
