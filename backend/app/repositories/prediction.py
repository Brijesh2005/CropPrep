"""Prediction + explanation-record repositories."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.models.prediction import ExplanationRecord, Prediction
from app.repositories.base import BaseRepository


class PredictionRepository(BaseRepository[Prediction]):
    """Async data access for :class:`Prediction`."""

    model = Prediction

    async def list_for_user(
        self, user_id: int, *, limit: int = 50, offset: int = 0
    ) -> list[Prediction]:
        result = await self.session.execute(
            select(Prediction)
            .where(Prediction.user_id == user_id)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def count(self, **filters: Any) -> int:
        statement = select(Prediction.id)
        for key, value in filters.items():
            if value is not None:
                statement = statement.where(getattr(Prediction, key) == value)
        result = await self.session.execute(statement)
        return len(result.scalars().all())

    async def crop_counts(self, *, limit: int = 20) -> list[tuple[str, int]]:
        from sqlalchemy import func

        result = await self.session.execute(
            select(Prediction.crop, func.count(Prediction.id))
            .group_by(Prediction.crop)
            .order_by(func.count(Prediction.id).desc())
            .limit(limit)
        )
        return list(result.all())


class ExplanationRepository(BaseRepository[ExplanationRecord]):
    """Async data access for :class:`ExplanationRecord`."""

    model = ExplanationRecord

    async def get_by_prediction(self, prediction_id: int) -> ExplanationRecord | None:
        result = await self.session.execute(
            select(ExplanationRecord).where(
                ExplanationRecord.prediction_id == prediction_id
            )
        )
        return result.scalar_one_or_none()
