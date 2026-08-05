"""Historical context resolver port (architecture contract).

Resolves the seasonal / climatological context for a point + date from the
exported ``historical_context.parquet`` and ``location_index.parquet``. This
is what lets the farmer-mode ``POST /predict`` accept latitude + longitude
ONLY: season, year and context are derived here rather than sent by the client.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from ..models import AdminContext, GeoPoint, HistoricalContext


class HistoricalContextResolver(ABC):
    """Port for point + date -> historical/seasonal context."""

    @abstractmethod
    def resolve(
        self,
        point: GeoPoint,
        day: date,
        admin: AdminContext | None = None,
    ) -> HistoricalContext:
        """Return season + climatology + target year for the point/date."""


__all__ = ["HistoricalContextResolver"]
