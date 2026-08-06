"""Spatial index over named locations (villages / districts / taluks / ...).

:class:`SpatialIndex` turns tabular location data (a CSV with village names,
latitudes and longitudes) into a queryable in-memory index:

* **Name lookups** — villages / districts by (case-insensitive) name.
* **Nearest neighbour** — KD-tree over WGS84 coordinates.
* **Coordinate search** — exact point match within a tolerance.
* **Bounding box / radius** — rectangular and circular spatial filters.

The index is deliberately coordinate-based (longitude / latitude degrees) so
it stays dependency-light: a scipy KD-tree when available, a linear scan
fallback otherwise. It feeds the manager's ``get_location``, the spatial
report and spatial validation.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .interfaces import SpatialIndex
from .logger import get_logger
from .models import SpatialMetadata, SpatialRecord

logger = get_logger("spatial_index")


class SpatialIndexImpl(SpatialIndex):
    """Concrete :class:`SpatialIndex` implementation.

    Args:
        records: Initial :class:`SpatialRecord` list (optional; use
            :meth:`build` to (re)index later).
    """

    def __init__(self, records: Iterable[SpatialRecord] | None = None) -> None:
        self._records: list[SpatialRecord] = []
        self._by_village: dict[str, list[int]] = {}
        self._by_district: dict[str, list[int]] = {}
        self._tree: Any = None
        self._coords: np.ndarray | None = None
        if records:
            self.build(list(records))

    # -- Index management ------------------------------------------------------ #

    def build(self, records: list[SpatialRecord]) -> int:
        """(Re)build the index from ``records``; returns the indexed count."""
        self._records = list(records)
        self._by_village = {}
        self._by_district = {}

        coords: list[tuple[float, float]] = []
        for index, record in enumerate(self._records):
            key = record.name.strip().lower()
            if record.kind == "village":
                self._by_village.setdefault(key, []).append(index)
            elif record.kind == "district":
                self._by_district.setdefault(key, []).append(index)
            if record.district:
                district_key = record.district.strip().lower()
                self._by_district.setdefault(district_key, []).append(index)
            if record.latitude is not None and record.longitude is not None:
                coords.append((float(record.latitude), float(record.longitude)))

        self._coords = np.asarray(coords, dtype=float).reshape(-1, 2) if coords else None
        self._tree = None
        if self._coords is not None and len(self._coords):
            try:
                from scipy.spatial import cKDTree

                self._tree = cKDTree(self._coords)
            except ImportError:  # pragma: no cover - linear fallback
                self._tree = None
        logger.info("Spatial index built", extra={"records": len(self._records)})
        return len(self._records)

    def records(self) -> list[SpatialRecord]:
        return list(self._records)

    # -- Name lookups ---------------------------------------------------------- #

    def lookup_village(self, name: str) -> list[SpatialRecord]:
        """Villages whose name matches ``name`` (case-insensitive)."""
        return self._by_key(self._by_village, name)

    def lookup_district(self, name: str) -> list[SpatialRecord]:
        """Districts whose name matches ``name`` (case-insensitive)."""
        return [
            r for r in self._by_key(self._by_district, name) if r.kind == "district"
        ]

    def _by_key(self, mapping: dict[str, list[int]], name: str) -> list[SpatialRecord]:
        key = str(name).strip().lower()
        if key in mapping:
            return [self._records[i] for i in sorted(mapping[key])]
        prefix = [i for k, idxs in mapping.items() if k.startswith(key) for i in idxs]
        return [self._records[i] for i in sorted(set(prefix))]

    # -- Coordinate queries ---------------------------------------------------- #

    def nearest(
        self, latitude: float, longitude: float, k: int = 1
    ) -> list[tuple[SpatialRecord, float]]:
        """``k`` nearest records to a point as ``(record, distance_deg)``.

        Distance is measured in degrees on the (latitude, longitude) axes —
        sufficient for ranking. Use :meth:`within_radius` for kilometres.
        """
        if not self._records:
            return []
        point = np.asarray([float(latitude), float(longitude)], dtype=float)
        if self._tree is not None:
            distances, indices = self._tree.query(point, k=min(k, len(self._records)))
            if k == 1:
                indices = [int(indices)]
                distances = [float(distances)]
            return [
                (self._records[int(i)], float(d))
                for i, d in zip(indices, distances)
                if int(i) < len(self._records)
            ]
        # Linear fallback.
        scored = sorted(
            (
                (self._records[i], self._distance_deg(latitude, longitude, r))
                for i, r in enumerate(self._records)
            ),
            key=lambda pair: pair[1],
        )
        return scored[:k]

    def search_coordinates(
        self, latitude: float, longitude: float, tolerance: float = 0.01
    ) -> list[SpatialRecord]:
        """Records within ``tolerance`` degrees of the point (exact match)."""
        return [
            r
            for r in self._records
            if abs(r.latitude - latitude) <= tolerance
            and abs(r.longitude - longitude) <= tolerance
        ]

    def within_bbox(
        self, min_lon: float, min_lat: float, max_lon: float, max_lat: float
    ) -> list[SpatialRecord]:
        """Records whose coordinates fall inside the longitude/latitude box."""
        return [
            r
            for r in self._records
            if min_lon <= r.longitude <= max_lon and min_lat <= r.latitude <= max_lat
        ]

    def within_radius(
        self, latitude: float, longitude: float, radius_km: float
    ) -> list[SpatialRecord]:
        """Records within ``radius_km`` kilometres of the point (haversine)."""
        return [
            r
            for r in self._records
            if _haversine_km(latitude, longitude, r.latitude, r.longitude) <= radius_km
        ]

    # -- Metadata -------------------------------------------------------------- #

    def metadata(self) -> SpatialMetadata:
        latitudes = [r.latitude for r in self._records]
        longitudes = [r.longitude for r in self._records]
        bounds = None
        if latitudes and longitudes:
            bounds = (
                min(longitudes), min(latitudes), max(longitudes), max(latitudes)
            )
        return SpatialMetadata(
            count=len(self._records),
            villages=sum(1 for r in self._records if r.kind == "village"),
            districts=sum(1 for r in self._records if r.kind == "district"),
            bounds=bounds,
        )

    # -- Helpers --------------------------------------------------------------- #

    @staticmethod
    def _distance_deg(
        lat1: float, lon1: float, record: SpatialRecord
    ) -> float:
        return abs(record.latitude - lat1) + abs(record.longitude - lon1)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two WGS84 points."""
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def build_records_from_frame(
    frame: Any,
    *,
    name_col: str,
    lat_col: str,
    lon_col: str,
    kind: str = "village",
    district_col: str | None = None,
    extra_cols: Iterable[str] | None = None,
) -> list[SpatialRecord]:
    """Build :class:`SpatialRecord` objects from a pandas DataFrame.

    Args:
        frame: Data frame with name / latitude / longitude columns.
        name_col: Column holding the location name.
        lat_col / lon_col: WGS84 coordinate columns.
        kind: Record kind (``village`` / ``district`` / ...).
        district_col: Optional parent district column.
        extra_cols: Optional extra columns copied into record metadata.

    Returns:
        A list of :class:`SpatialRecord` (rows with unusable coordinates are
        skipped).
    """
    from .utils import safe_float

    records: list[SpatialRecord] = []
    extra_cols = list(extra_cols or [])
    for _, row in frame.iterrows():
        lat = safe_float(row.get(lat_col))
        lon = safe_float(row.get(lon_col))
        if lat is None or lon is None:
            continue
        name = str(row.get(name_col, "")).strip()
        if not name:
            continue
        metadata = {col: str(row.get(col)) for col in extra_cols if col in row}
        district = None
        if district_col and district_col in row and row.get(district_col):
            district = str(row.get(district_col))
        records.append(
            SpatialRecord(
                name=name,
                kind=kind,
                latitude=lat,
                longitude=lon,
                district=district,
                metadata=metadata,
            )
        )
    return records
