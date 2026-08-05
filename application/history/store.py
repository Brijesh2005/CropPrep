"""Prediction history store port (architecture contract).

Persists and queries prediction results against the existing enterprise
schema. R1.4 is a port only — the SQLAlchemy implementation already exists in
``application/backend/app/repositories/prediction.py`` and
``application/database/repositories/prediction_repository.py``; a future
phase binds this port to them (or to a lightweight inference-only store) so
the services layer stays storage-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..inference.models import PredictionContext, PredictionResult
from .models import HistoryFilters, HistoryPage, HistoryRecord


class PredictionHistoryStore(ABC):
    """Port for saving and querying prediction history."""

    @abstractmethod
    async def save(
        self,
        result: PredictionResult,
        context: PredictionContext | None = None,
        *,
        user_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HistoryRecord:
        """Persist a prediction and return the stored record."""

    @abstractmethod
    async def search(self, filters: HistoryFilters) -> HistoryPage:
        """Search stored predictions with the given filters."""

    @abstractmethod
    async def get_metadata(self, prediction_id: int) -> dict[str, Any] | None:
        """Return the stored metadata for a prediction, or None."""


__all__ = ["PredictionHistoryStore"]
