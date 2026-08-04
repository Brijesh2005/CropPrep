"""Prediction metadata repository."""

from __future__ import annotations

from sqlalchemy import select

from database.models.metadata import PredictionMetadata
from database.repositories.base import DataRepository


class PredictionMetadataRepository(DataRepository[PredictionMetadata]):
    model = PredictionMetadata

    async def get_by_prediction(self, prediction_id: int) -> PredictionMetadata | None:
        result = await self.session.execute(
            select(PredictionMetadata).where(PredictionMetadata.prediction_id == prediction_id)
        )
        return result.scalar_one_or_none()

    async def get_or_create(self, prediction_id: int) -> PredictionMetadata:
        existing = await self.get_by_prediction(prediction_id)
        if existing is not None:
            return existing
        return PredictionMetadata(prediction_id=prediction_id)
