"""Unit tests for the spatial index (KDTree + boundary STRtree)."""

from __future__ import annotations

import geopandas as gpd
import pytest
import shapely.geometry
from shapely.geometry import Polygon

from training.stam.exceptions import LocationNotFoundError
from training.stam.spatial_index import (
    BoundaryIndex,
    KDTreeSpatialIndex,
    LocationPoint,
    haversine_km,
)


def _points():
    return [
        LocationPoint("p1", "A", 74.80, 13.10),
        LocationPoint("p2", "B", 74.82, 13.12),
        LocationPoint("p3", "C", 74.85, 13.05),
    ]


def test_haversine_sanity():
    # ~111 km per degree of latitude.
    assert 100 < haversine_km(74.8, 13.0, 74.8, 14.0) < 120


def test_nearest_one():
    index = KDTreeSpatialIndex(max_radius_km=100).build(_points())
    match = index.nearest_one(74.8005, 13.1005)
    assert match.point.id == "p1"
    assert match.distance_km < 0.1


def test_nearest_k_returns_sorted():
    index = KDTreeSpatialIndex(max_radius_km=100).build(_points())
    matches = index.nearest(74.80, 13.10, k=3)
    assert [m.point.id for m in matches] == ["p1", "p2", "p3"]
    assert matches[0].distance_km <= matches[-1].distance_km


def test_radius_filter():
    index = KDTreeSpatialIndex(max_radius_km=1.0).build(_points())
    with pytest.raises(LocationNotFoundError):
        index.nearest_one(74.95, 13.5)  # far away


def test_empty_index_raises():
    index = KDTreeSpatialIndex()
    with pytest.raises(LocationNotFoundError):
        index.nearest_one(74.8, 13.1)


def test_duplicate_tolerance_dedupes():
    close = [
        LocationPoint("a", "A", 74.80, 13.10),
        LocationPoint("b", "A-dup", 74.800001, 13.100001),  # ~0.15 m apart
    ]
    index = KDTreeSpatialIndex(duplicate_tolerance_m=5.0).build(close)
    assert len(index) == 1


def test_invalid_points_skipped():
    bad = [LocationPoint("x", "X", float("nan"), 13.1)]
    index = KDTreeSpatialIndex().build(bad)
    assert len(index) == 0


def _boundary_gdf():
    polys = {
        "name": ["A", "T1", "DK"],
        "level": ["village", "taluk", "district"],
        "geometry": [
            Polygon([(74.79, 13.08), (74.81, 13.08), (74.81, 13.11), (74.79, 13.11)]),
            Polygon([(74.78, 13.07), (74.85, 13.07), (74.85, 13.12), (74.78, 13.12)]),
            Polygon([(74.75, 13.05), (74.90, 13.05), (74.90, 13.20), (74.75, 13.20)]),
        ],
    }
    return gpd.GeoDataFrame(polys, crs="EPSG:4326")


def test_boundary_find_containing():
    index = BoundaryIndex().build(_boundary_gdf())
    hit = index.find_containing(74.80, 13.10)
    assert hit is not None
    assert hit.name == "A"
    assert hit.level == "village"


def test_boundary_find_containing_all():
    index = BoundaryIndex().build(_boundary_gdf())
    hits = index.find_containing_all(74.80, 13.10)
    names = sorted(h.name for h in hits)
    assert names == ["A", "DK", "T1"]


def test_boundary_outside_returns_none():
    index = BoundaryIndex().build(_boundary_gdf())
    assert index.find_containing(75.0, 14.0) is None
