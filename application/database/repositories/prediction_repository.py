"""Enterprise prediction repository: history filters + analytics aggregates."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select

from app.models.prediction import Prediction
from app.repositories.prediction import PredictionRepository as BasePredictionRepository
from database.repositories.base import DataRepository


class PredictionRepository(BasePredictionRepository, DataRepository[Prediction]):
    """History querying with the Phase 10 geography/season filters."""

    model = Prediction

    async def search_history(
        self,
        user_id: int,
        *,
        crop: str | None = None,
        season: str | None = None,
        year: int | None = None,
        district: str | None = None,
        taluk: str | None = None,
        village: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        min_confidence: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Prediction], int]:
        stmt = select(Prediction).where(Prediction.user_id == user_id)
        if crop:
            stmt = stmt.where(Prediction.crop == crop)
        if season:
            stmt = stmt.where(Prediction.season == season)
        if year:
            stmt = stmt.where(Prediction.year == year)
        if district:
            stmt = stmt.where(Prediction.district == district)
        if taluk:
            stmt = stmt.where(Prediction.taluk == taluk)
        if village:
            stmt = stmt.where(Prediction.village == village)
        if date_from:
            stmt = stmt.where(Prediction.created_at >= date_from)
        if date_to:
            stmt = stmt.where(Prediction.created_at <= date_to)
        if min_confidence is not None:
            stmt = stmt.where(Prediction.confidence >= min_confidence)
        stmt = stmt.order_by(Prediction.created_at.desc())
        return await self.paginate(stmt, limit=limit, offset=offset)

    async def analytics_by_region(
        self, *, district: str | None = None, season: str | None = None, year: int | None = None
    ) -> list[dict[str, Any]]:
        """Aggregate avg confidence / avg yield / prediction count per region."""
        cols = [Prediction.district, func.avg(Prediction.confidence), func.avg(Prediction.yield_prediction), func.count(Prediction.id)]
        stmt = (
            select(*cols)
            .where(Prediction.district.isnot(None))
            .group_by(Prediction.district)
        )
        if season:
            stmt = stmt.where(Prediction.season == season)
        if year:
            stmt = stmt.where(Prediction.year == year)
        if district:
            stmt = stmt.where(Prediction.district == district)
        rows = await self._execute_all(stmt)
        return [
            {
                "district": r[0],
                "avg_confidence": round(float(r[1]), 4) if r[1] is not None else None,
                "avg_yield": round(float(r[2]), 4) if r[2] is not None else None,
                "predictions": int(r[3]),
            }
            for r in rows
        ]

    async def analytics_by_crop(
        self, *, season: str | None = None, year: int | None = None
    ) -> list[dict[str, Any]]:
        cols = [
            Prediction.crop,
            func.avg(Prediction.confidence),
            func.avg(Prediction.yield_prediction),
            func.count(Prediction.id),
        ]
        stmt = select(*cols).group_by(Prediction.crop)
        if season:
            stmt = stmt.where(Prediction.season == season)
        if year:
            stmt = stmt.where(Prediction.year == year)
        rows = await self._execute_all(stmt)
        return [
            {
                "crop": r[0],
                "avg_confidence": round(float(r[1]), 4) if r[1] is not None else None,
                "avg_yield": round(float(r[2]), 4) if r[2] is not None else None,
                "predictions": int(r[3]),
            }
            for r in rows
        ]

    async def confidence_distribution(self) -> list[dict[str, Any]]:
        bucket = func.round(Prediction.confidence * 10).label("bucket")
        stmt = select(bucket, func.count(Prediction.id)).group_by(bucket)
        rows = await self._execute_all(stmt)
        return [{"bucket": int(r[0]), "count": int(r[1])} for r in rows]

    async def _execute_all(self, statement):
        result = await self.session.execute(statement)
        return list(result.all())
