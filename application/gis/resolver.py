"""Location resolver facade (architecture contract).

Composes the GIS chain for a request date:

    GeoPoint
      -> ReverseGeocoder
      -> SpatialResolver
      -> HistoricalContextResolver
      -> GeoContext

A future phase wires this facade into ``application.inference.services`` so
the inference engine receives a fully resolved context. R1.4 defines the
orchestration contract only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from .historical_context import HistoricalContextResolver
from .models import GeoContext, GeoPoint
from .reverse_geocoding import ReverseGeocoder
from .spatial_resolver import SpatialResolver


class LocationResolver(ABC):
    """Port for the full point -> context resolution chain."""

    #: Reference to the default composition; subclasses may narrow it.
    reverse_geocoder: ReverseGeocoder
    spatial_resolver: SpatialResolver
    historical_resolver: HistoricalContextResolver

    @abstractmethod
    def resolve(self, point: GeoPoint, day: date | None = None) -> GeoContext:
        """Resolve ``point`` into a complete :class:`GeoContext`."""


__all__ = ["LocationResolver"]
