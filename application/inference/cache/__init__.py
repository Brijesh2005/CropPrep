"""Prediction cache port (architecture contract).

Caches :class:`PredictionResult` keyed by location + date bucket so repeated
requests for the same point are served without re-running the engine. The
existing backend cache service is the reference implementation; this port
documents the future inference-only contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from ..models import PredictionResult


class PredictionCache(ABC):
    """Port for a keyed prediction cache."""

    @abstractmethod
    async def get(self, lon: float, lat: float, day: date) -> PredictionResult | None:
        """Return a cached result for the location/day, or None."""

    @abstractmethod
    async def set(
        self,
        lon: float,
        lat: float,
        day: date,
        result: PredictionResult,
        *,
        ttl_seconds: int,
    ) -> None:
        """Store ``result`` under the location/day key."""

    @abstractmethod
    async def clear(self, lon: float | None = None, lat: float | None = None) -> None:
        """Evict one point (or everything when coordinates are omitted)."""


__all__ = ["PredictionCache"]
