"""Prediction history service: filtered history + metadata + export markers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.models.prediction import Prediction
from database.repositories import PredictionMetadataRepository, PredictionRepository


class PredictionHistoryService:
    """History search and metadata enrichment for predictions."""

    def __init__(
        self,
        predictions: PredictionRepository,
        metadata_repo: PredictionMetadataRepository,
    ) -> None:
        self._predictions = predictions
        self._metadata = metadata_repo

    async def search(
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
    ) -> dict[str, Any]:
        rows, total = await self._predictions.search_history(
            user_id,
            crop=crop, season=season, year=year, district=district,
            taluk=taluk, village=village, date_from=date_from, date_to=date_to,
            min_confidence=min_confidence, limit=limit, offset=offset,
        )
        return {
            "items": [p.history_dict() for p in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def get_metadata(self, prediction_id: int) -> dict[str, Any] | None:
        meta = await self._metadata.get_by_prediction(prediction_id)
        if meta is None:
            return None
        return {
            "prediction_id": meta.prediction_id,
            "inputs": meta.inputs,
            "feature_snapshot": meta.feature_snapshot,
            "weather": meta.weather,
            "soil": meta.soil,
            "tags": meta.tags,
            "client_context": meta.client_context,
        }

    async def attach_metadata(self, prediction: Prediction, *, metadata: dict[str, Any]) -> None:
        entry = await self._metadata.get_or_create(prediction.id)
        for key, value in metadata.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        await self._metadata.add(entry)
        await self._metadata.commit()

    async def mark_exported(self, prediction: Prediction) -> Prediction:
        prediction.is_exported = True
        await self._predictions.commit()
        return prediction
