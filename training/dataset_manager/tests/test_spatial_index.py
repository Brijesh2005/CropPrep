"""Tests for :class:`SpatialIndexImpl` — the R2.2 spatial index.

Covers building the index from :class:`SpatialRecord` objects and from tabular
DataFrames (:func:`build_records_from_frame`), plus name lookups, nearest
neighbour, coordinate matching, bounding-box and radius queries.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import pytest

from training.dataset_manager.models import SpatialRecord
from training.dataset_manager.spatial_index import (
    SpatialIndexImpl,
    build_records_from_frame,
)

_RECORDS = [
    SpatialRecord("Moodabidri", "village", 13.08, 74.89, district="Dakshina Kannada"),
    SpatialRecord("Bantwal", "village", 12.90, 75.00, district="Dakshina Kannada"),
    SpatialRecord("Sullia", "village", 12.56, 75.35, district="Dakshina Kannada"),
    SpatialRecord("Dakshina Kannada", "district", 12.85, 75.15),
]


@pytest.fixture
def index() -> SpatialIndexImpl:
    return SpatialIndexImpl(_RECORDS)


def test_build_and_records(index: SpatialIndexImpl):
    assert len(index.records()) == 4
    assert index.metadata().count == 4


def test_lookup_village(index: SpatialIndexImpl):
    matches = index.lookup_village("moodabidri")
    assert len(matches) == 1
    assert matches[0].name == "Moodabidri"
    assert matches[0].district == "Dakshina Kannada"


def test_lookup_village_prefix(index: SpatialIndexImpl):
    matches = index.lookup_village("sull")
    assert [m.name for m in matches] == ["Sullia"]


def test_lookup_district(index: SpatialIndexImpl):
    matches = index.lookup_district("dakshina kannada")
    assert len(matches) == 1
    assert matches[0].kind == "district"


def test_lookup_unknown_returns_empty(index: SpatialIndexImpl):
    assert index.lookup_village("nowhere") == []


def test_nearest(index: SpatialIndexImpl):
    nearest = index.nearest(13.09, 74.88, k=1)
    assert len(nearest) == 1
    record, distance = nearest[0]
    assert record.name == "Moodabidri"
    assert distance >= 0.0


def test_nearest_k(index: SpatialIndexImpl):
    nearest = index.nearest(13.09, 74.88, k=3)
    assert len(nearest) == 3
    distances = [d for _r, d in nearest]
    assert distances == sorted(distances)


def test_search_coordinates(index: SpatialIndexImpl):
    matches = index.search_coordinates(13.081, 74.891, tolerance=0.01)
    assert [m.name for m in matches] == ["Moodabidri"]


def test_search_coordinates_no_match(index: SpatialIndexImpl):
    assert index.search_coordinates(60.0, 10.0) == []


def test_within_bbox(index: SpatialIndexImpl):
    matches = index.within_bbox(74.8, 12.8, 75.1, 13.2)
    names = {m.name for m in matches}
    assert "Moodabidri" in names and "Bantwal" in names
    assert "Sullia" not in names


def test_within_radius_km(index: SpatialIndexImpl):
    matches = index.within_radius(13.08, 74.89, radius_km=20)
    assert [m.name for m in matches] == ["Moodabidri"]


def test_metadata_counts(index: SpatialIndexImpl):
    metadata = index.metadata()
    assert metadata.villages == 3
    assert metadata.districts == 1
    assert metadata.bounds is not None
    min_lon, min_lat, max_lon, max_lat = metadata.bounds
    assert min_lon == 74.89 and max_lon == 75.35
    assert min_lat == 12.56 and max_lat == 13.08


def test_build_replaces_records():
    index = SpatialIndexImpl([_RECORDS[0]])
    count = index.build(_RECORDS)
    assert count == 4
    assert len(index.records()) == 4


def test_empty_index_queries():
    index = SpatialIndexImpl([])
    assert index.nearest(13.0, 75.0) == []
    assert index.within_radius(13.0, 75.0, 10) == []
    assert index.metadata().count == 0


def test_build_records_from_frame():
    frame = pd.DataFrame(
        {
            "village": ["A", "B"],
            "district": ["D1", "D1"],
            "latitude": [10.0, 11.0],
            "longitude": [20.0, 21.0],
            "crop": ["rice", "rice"],
        }
    )
    records = build_records_from_frame(
        frame,
        name_col="village",
        lat_col="latitude",
        lon_col="longitude",
        kind="village",
        district_col="district",
        extra_cols=["crop"],
    )
    assert len(records) == 2
    assert records[0].name == "A"
    assert records[0].district == "D1"
    assert records[0].metadata["crop"] == "rice"


def test_build_records_from_frame_skips_unusable_rows():
    frame = pd.DataFrame(
        {
            "village": ["ok", "bad_lat", "bad_lon"],
            "latitude": [10.0, "n/a", 12.0],
            "longitude": [20.0, 22.0, "n/a"],
        }
    )
    records = build_records_from_frame(
        frame, name_col="village", lat_col="latitude", lon_col="longitude"
    )
    assert len(records) == 1
    assert records[0].name == "ok"
