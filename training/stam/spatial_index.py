"""Spatial indexing for STAM: nearest dataset point + admin boundary lookup.

Two indexes are used:

* :class:`KDTreeSpatialIndex` — nearest-neighbour search over dataset
  locations (villages / image centroids) using :class:`scipy.spatial.cKDTree`
  for candidate pruning, refined with the haversine formula for exact
  kilometre distances.
* :class:`BoundaryIndex` — point-in-polygon resolution over administrative
  boundaries using a Shapely STRtree (R-tree) accelerated query.

**Why KDTree (over BallTree / RTree) for nearest points?** Nearest-point
search in 2D is a classic KDTree problem: ``cKDTree`` offers O(log n) queries,
a tiny memory footprint and zero extra C dependencies (scipy is already a
project dependency). BallTree is designed for high-dimensional / non-Euclidean
metrics where its better partitioning pays off — overkill for (lon, lat).
RTree shines for *bounding-box* queries and polygon containment, which is
exactly what :class:`BoundaryIndex` uses Shapely's STRtree for. Both concerns
are served by the tool designed for them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree

from .coordinate_transform import normalise_crs
from .exceptions import LocationNotFoundError
from .logger import get_logger

logger = get_logger("spatial_index")

# Mean Earth radius (km).
_EARTH_RADIUS_KM = 6371.0088


@dataclass(slots=True)
class LocationPoint:
    """A candidate dataset location (village centroid / image centroid)."""

    id: str
    name: str
    lon: float
    lat: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NearestMatch:
    """Result of a nearest-location query."""

    point: LocationPoint
    distance_km: float


@dataclass(slots=True)
class BoundaryHit:
    """A boundary feature containing a queried point."""

    name: str
    level: str | None
    attributes: dict[str, Any]
    index: int


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometres between two lon/lat points."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


class KDTreeSpatialIndex:
    """KDTree-backed nearest-neighbour index over dataset locations.

    Args:
        max_radius_km: Queries beyond this radius return no matches.
        duplicate_tolerance_m: Locations closer than this (metres) collapse
            into one entry when the index is built.
        leaf_size: KDTree leaf size (query performance tuning).
    """

    def __init__(
        self,
        *,
        max_radius_km: float = 5.0,
        duplicate_tolerance_m: float = 50.0,
        leaf_size: int = 40,
    ) -> None:
        self.max_radius_km = max_radius_km
        self.duplicate_tolerance_deg = duplicate_tolerance_m / 111_320.0  # approx deg
        self.leaf_size = leaf_size
        self.points: list[LocationPoint] = []
        self._coords: np.ndarray | None = None
        self._tree = None
        self._candidate_k = 16

    # -- Construction --------------------------------------------------------- #

    def build(self, points: Iterable[LocationPoint]) -> "KDTreeSpatialIndex":
        """Rebuild the index from a set of location points (deduplicated)."""
        from scipy.spatial import cKDTree

        tolerance_km = self.duplicate_tolerance_deg * 111_320.0 / 1000.0
        deduped: list[LocationPoint] = []
        for point in points:
            if not math.isfinite(point.lon) or not math.isfinite(point.lat):
                continue
            if any(
                haversine_km(p.lon, p.lat, point.lon, point.lat) < tolerance_km
                for p in deduped
            ):
                continue
            deduped.append(point)

        self.points = deduped
        if deduped:
            self._coords = np.asarray(
                [[p.lon, p.lat] for p in deduped], dtype="float64"
            )
            self._tree = cKDTree(self._coords, leafsize=self.leaf_size)
        else:
            self._coords = None
            self._tree = None
        logger.info("Spatial index built", extra={"points": len(deduped)})
        return self

    def __len__(self) -> int:
        return len(self.points)

    @property
    def is_built(self) -> bool:
        return self._tree is not None

    # -- Queries -------------------------------------------------------------- #

    def nearest(
        self, lon: float, lat: float, *, k: int = 1
    ) -> list[NearestMatch]:
        """Return up to ``k`` nearest matches, sorted by distance (km).

        Candidate pruning uses the KDTree (Euclidean on lon/lat degrees);
        exact ordering and radius filtering use the haversine formula.

        Raises:
            LocationNotFoundError: When the index is empty or nothing is
                within ``max_radius_km``.
        """
        if self._tree is None or len(self.points) == 0:
            raise LocationNotFoundError(
                "Spatial index is empty; call initialize() first"
            )
        # Query a generous candidate pool, then refine by haversine.
        pool = max(k * 4, self._candidate_k)
        pool = min(pool, len(self.points))
        _dist, idxs = self._tree.query(
            np.asarray([lon, lat], dtype="float64"), k=pool
        )
        idxs = np.atleast_1d(idxs)
        matches: list[NearestMatch] = []
        for i in idxs:
            point = self.points[int(i)]
            distance = haversine_km(lon, lat, point.lon, point.lat)
            if distance <= self.max_radius_km:
                matches.append(NearestMatch(point=point, distance_km=distance))
        matches.sort(key=lambda m: m.distance_km)
        if not matches:
            raise LocationNotFoundError(
                f"No dataset location within {self.max_radius_km:.1f} km",
                detail={"lon": lon, "lat": lat},
            )
        return matches[:k]

    def nearest_one(self, lon: float, lat: float) -> NearestMatch:
        """Shortcut for the single best match."""
        return self.nearest(lon, lat, k=1)[0]


class BoundaryIndex:
    """R-tree (STRtree) index over administrative boundary polygons.

    Args:
        name_column: Attribute holding the feature name.
        level_column: Optional attribute holding the admin level.
    """

    def __init__(self, *, name_column: str = "name", level_column: str = "level") -> None:
        self.name_column = name_column
        self.level_column = level_column
        self._geoms: list[Any] = []
        self._attrs: list[dict[str, Any]] = []
        self._tree: STRtree | None = None

    # -- Construction --------------------------------------------------------- #

    def build(self, gdf: Any) -> "BoundaryIndex":
        """Index a GeoDataFrame (reprojected to EPSG:4326 in-place copy)."""
        if gdf is None or len(gdf) == 0:
            return self
        work = gdf.to_crs(4326) if gdf.crs is not None else gdf
        geoms = [g for g in work.geometry if g is not None]
        if not geoms:
            return self
        self._geoms = geoms
        self._attrs = [dict(row) for _, row in work.iterrows()]
        self._tree = STRtree(geoms)
        logger.info("Boundary index built", extra={"features": len(geoms)})
        return self

    def __len__(self) -> int:
        return len(self._geoms)

    @property
    def is_built(self) -> bool:
        return self._tree is not None

    # -- Queries -------------------------------------------------------------- #

    def _query_geometries(self, point: Point) -> list[Any]:
        """STRtree candidate polygons whose envelope intersects ``point``.

        Shapely 2.x ``query`` returns an index array; 1.x returned geometry
        objects. Normalise to a list of geometry objects.
        """
        if self._tree is None:
            return []
        result = self._tree.query(point)
        if hasattr(result, "dtype") and result.dtype.kind in "iu":  # index array
            return [self._geoms[int(i)] for i in result]
        return list(result)

    def find_containing(self, lon: float, lat: float) -> BoundaryHit | None:
        """Return the finest boundary containing ``(lon, lat)``, or None.

        The STRtree returns all candidate polygons whose envelope intersects
        the point; the first one that truly contains it wins.
        """
        point = Point(lon, lat)
        for geom in self._query_geometries(point):
            if geom.contains(point) or geom.covers(point):
                index = self._geoms.index(geom)
                attrs = self._attrs[index]
                return BoundaryHit(
                    name=str(attrs.get(self.name_column, "")),
                    level=str(attrs.get(self.level_column)) if self.level_column in attrs else None,
                    attributes=attrs,
                    index=index,
                )
        return None

    def find_containing_all(self, lon: float, lat: float) -> list[BoundaryHit]:
        """Return every boundary containing the point (village, taluk, ...)."""
        point = Point(lon, lat)
        hits: list[BoundaryHit] = []
        for geom in self._query_geometries(point):
            if geom.contains(point) or geom.covers(point):
                index = self._geoms.index(geom)
                attrs = self._attrs[index]
                hits.append(
                    BoundaryHit(
                        name=str(attrs.get(self.name_column, "")),
                        level=str(attrs.get(self.level_column))
                        if self.level_column in attrs
                        else None,
                        attributes=attrs,
                        index=index,
                    )
                )
        return hits
