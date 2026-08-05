"""GIS module repository — spatial index over dataset locations."""

from __future__ import annotations

from typing import Sequence

from app.modules.gis.service import Location


class LocationRepository:
    """Holds the dataset locations and a small in-memory spatial index."""

    def __init__(self, locations: Sequence[Location] | None = None) -> None:
        self._locations = list(locations or [])
        self._by_id = {loc.id: loc for loc in self._locations}

    def all(self) -> list[Location]:
        return list(self._locations)

    def get(self, location_id: str) -> Location | None:
        return self._by_id.get(location_id)

    def add(self, location: Location) -> None:
        self._locations.append(location)
        self._by_id[location.id] = location

    def replace(self, locations: Sequence[Location]) -> None:
        self._locations = list(locations)
        self._by_id = {loc.id: loc for loc in self._locations}
