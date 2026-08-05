"""Reverse geocoding port (architecture contract).

Maps a raw :class:`~application.gis.models.GeoPoint` to the nearest known
place using the exported ``location_index.parquet`` (and the boundary
shapefiles as fallback). R1.4 is a port only; the implementation will run in
a later phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import GeoPoint, ResolvedPlace


class ReverseGeocoder(ABC):
    """Port for point -> place resolution."""

    @abstractmethod
    def resolve(self, point: GeoPoint) -> ResolvedPlace:
        """Return the nearest known place for ``point``."""


__all__ = ["ReverseGeocoder"]
