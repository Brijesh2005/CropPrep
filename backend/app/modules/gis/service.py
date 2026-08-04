"""GIS module service — nearest location, spatial search, boundaries.

Only dataset locations are returned. Coordinates are validated before any
spatial lookup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from app.core.exceptions import GISError, NotFoundError
from app.modules.gis.schemas import BoundaryResponse, LocationResponse


@dataclass
class Location:
    """A known dataset location (village / observation point)."""

    id: str
    lon: float
    lat: float
    name: str = ""
    admin: dict = field(default_factory=dict)


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class GISService:
    """Spatial queries over the dataset locations."""

    def __init__(
        self,
        locations: Sequence[Location] | None = None,
        boundaries: Any = None,
    ) -> None:
        self._locations = {loc.id: loc for loc in (locations or [])}
        self._boundaries = boundaries

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_coordinates(lon: float, lat: float) -> None:
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise GISError(
                "invalid coordinates", detail={"lon": lon, "lat": lat}
            )

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def nearest(self, lon: float, lat: float, k: int = 1) -> list[LocationResponse]:
        self.validate_coordinates(lon, lat)
        if not self._locations:
            raise GISError("no dataset locations available")
        ranked = sorted(
            self._locations.values(),
            key=lambda loc: _haversine_km(lon, lat, loc.lon, loc.lat),
        )
        return [
            LocationResponse(
                id=loc.id,
                lon=loc.lon,
                lat=loc.lat,
                name=loc.name,
                admin=loc.admin,
                distance_km=round(_haversine_km(lon, lat, loc.lon, loc.lat), 3),
            )
            for loc in ranked[:k]
        ]

    def get(self, location_id: str) -> LocationResponse:
        loc = self._locations.get(location_id)
        if loc is None:
            raise NotFoundError("location not found", detail=location_id)
        return LocationResponse(id=loc.id, lon=loc.lon, lat=loc.lat, name=loc.name, admin=loc.admin)

    def search(self, query: str, limit: int = 10) -> list[LocationResponse]:
        q = query.lower().strip()
        matches = []
        for loc in self._locations.values():
            haystack = f"{loc.name} {loc.admin.get('village', '')} {loc.admin.get('district', '')}".lower()
            if q in haystack:
                matches.append(
                    LocationResponse(id=loc.id, lon=loc.lon, lat=loc.lat, name=loc.name, admin=loc.admin)
                )
        return matches[:limit]

    def list(self, limit: int = 100) -> list[LocationResponse]:
        return [
            LocationResponse(id=loc.id, lon=loc.lon, lat=loc.lat, name=loc.name, admin=loc.admin)
            for loc in list(self._locations.values())[:limit]
        ]

    # ------------------------------------------------------------------ #
    # Boundaries
    # ------------------------------------------------------------------ #

    def boundaries(self) -> list[BoundaryResponse]:
        if self._boundaries is None:
            return []
        try:
            return self._summarize_boundaries(self._boundaries)
        except Exception as exc:
            raise GISError("failed to read administrative boundaries", detail=str(exc)) from exc

    @staticmethod
    def _summarize_boundaries(gdf: Any) -> list[BoundaryResponse]:
        result: list[BoundaryResponse] = []
        for _, row in gdf.iterrows():
            geometry = row.get("geometry")
            if geometry is None:
                continue
            name = str(row.get("name", row.get("NAME_2", "boundary")))
            result.append(
                BoundaryResponse(
                    name=name,
                    geometry_type=geometry.geom_type,
                    bbox=list(geometry.bounds),
                    features=1,
                )
            )
        return result
