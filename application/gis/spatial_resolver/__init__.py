"""Spatial resolver port (architecture contract).

Maps a resolved place to the administrative hierarchy (district / taluk /
village) using the boundary data already shipped in ``application/gis``
(District/, Taluk/, Dakshina_Kannada/ shapefiles, kml/). R1.4 is a port only;
the point-in-polygon implementation is deferred.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import AdminContext, GeoPoint, ResolvedPlace


class SpatialResolver(ABC):
    """Port for place -> administrative-context resolution."""

    @abstractmethod
    def resolve(self, point: GeoPoint, place: ResolvedPlace | None = None) -> AdminContext:
        """Return the administrative context containing ``point``."""


__all__ = ["SpatialResolver"]
