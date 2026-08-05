"""History module dependencies."""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.database import get_session
from app.modules.history.service import HistoryService


def get_history_service(session: AsyncSession = Depends(get_session)) -> HistoryService:
    return HistoryService(session)
