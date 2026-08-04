"""History module service."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.prediction import Prediction
from app.modules.history.schemas import HistoryItem, HistoryPage
from app.modules.history.repository import PredictionRepository


class HistoryService:
    """Reads past predictions for a user."""

    def __init__(self, session: AsyncSession) -> None:
        self._repository = PredictionRepository(session)

    async def list_for_user(
        self, user_id: int, *, limit: int = 50, offset: int = 0
    ) -> HistoryPage:
        records = await self._repository.list_for_user(
            user_id, limit=limit, offset=offset
        )
        total = await self._repository.count(user_id=user_id)
        items = [self._to_item(record) for record in records]
        return HistoryPage(items=items, total=total, limit=limit, offset=offset)

    @staticmethod
    def _to_item(record: Prediction) -> HistoryItem:
        return HistoryItem(
            prediction_id=record.id,
            location={
                "lon": record.location_lon,
                "lat": record.location_lat,
                "name": record.location_name,
            },
            recommended_crop=record.crop,
            expected_yield=record.yield_prediction,
            confidence=record.confidence,
            model_version=record.model_version,
            created_at=record.created_at,
        )
