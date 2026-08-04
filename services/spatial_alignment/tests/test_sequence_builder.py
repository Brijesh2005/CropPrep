"""Unit tests for the image-pair builder and sequence builder."""

from __future__ import annotations

from datetime import date

import pytest

from services.spatial_alignment.exceptions import PairingError
from services.spatial_alignment.observation import ImageRecordRef
from services.spatial_alignment.sequence_builder import (
    ImagePairBuilder,
    ObservationSequenceBuilder,
)


def _ref(index: str, day: date, **overrides) -> ImageRecordRef:
    defaults = {
        "path": f"/data/{index}_{day}.tif",
        "relative_path": f"{index}_{day}.tif",
        "index_type": index,
        "resolution": "R10m",
        "observation_date": day,
        "year": day.year,
        "crs": "EPSG:32643",
        "pixel_size": (0.0001, 0.0001),
        "bounds": (74.8, 13.0, 74.9, 13.1),
    }
    defaults.update(overrides)
    return ImageRecordRef(**defaults)


@pytest.fixture
def ndvi_records():
    return [_ref("NDVI", date(2020, 7, 1)), _ref("NDVI", date(2020, 8, 1))]


@pytest.fixture
def evi_records():
    return [_ref("EVI", date(2020, 7, 1)), _ref("EVI", date(2020, 8, 1))]


def test_pairing_by_date(ndvi_records, evi_records):
    pairs, issues, dups = ImagePairBuilder().build(ndvi_records, evi_records)
    assert len(pairs) == 2
    assert all(p.ndvi and p.evi for p in pairs)
    assert all(p.quality["paired"] for p in pairs)
    assert dups == 0 and not issues


def test_pairing_sorts_out_of_order(ndvi_records, evi_records):
    pairs, _, _ = ImagePairBuilder().build(
        list(reversed(ndvi_records)), list(reversed(evi_records))
    )
    assert [p.date for p in pairs] == [date(2020, 7, 1), date(2020, 8, 1)]


def test_duplicate_dates_dropped():
    dup = [
        _ref("NDVI", date(2020, 7, 1)),
        _ref("NDVI", date(2020, 7, 1), path="/data/other.tif"),
    ]
    pairs, issues, dups = ImagePairBuilder(require_pairs=False).build(dup, [])
    assert dups == 1
    assert any(i.code == "ST-IMAGE-003" for i in issues)


def test_missing_side_flags_warning():
    ndvi = [_ref("NDVI", date(2020, 7, 1))]
    pairs, issues, _ = ImagePairBuilder(require_pairs=False).build(ndvi, [])
    assert len(pairs) == 1
    assert pairs[0].quality["missing"] == ["EVI"]
    assert pairs[0].evi is None


def test_require_pairs_raises():
    ndvi = [_ref("NDVI", date(2020, 7, 1))]
    with pytest.raises(PairingError):
        ImagePairBuilder(require_pairs=True).build(ndvi, [])


def test_resolution_mismatch_flagged():
    ndvi = [_ref("NDVI", date(2020, 7, 1))]
    evi = [_ref("EVI", date(2020, 7, 1), resolution="R20m")]
    pairs, issues, _ = ImagePairBuilder(require_pairs=False).build(ndvi, evi)
    assert any(i.code == "ST-PAIR-002" for i in issues)
    assert pairs[0].quality["invalid"] is True


def test_crs_mismatch_flagged():
    ndvi = [_ref("NDVI", date(2020, 7, 1))]
    evi = [_ref("EVI", date(2020, 7, 1), crs="EPSG:4326")]
    pairs, issues, _ = ImagePairBuilder(require_pairs=False).build(ndvi, evi)
    assert any(i.code == "ST-PAIR-003" for i in issues)


def test_bbox_mismatch_flagged():
    ndvi = [_ref("NDVI", date(2020, 7, 1))]
    evi = [_ref("EVI", date(2020, 7, 1), bounds=(74.8, 13.0, 74.95, 13.1))]
    pairs, issues, _ = ImagePairBuilder(require_pairs=False).build(ndvi, evi)
    assert any(i.code == "ST-PAIR-004" for i in issues)


def test_sequence_builder_ordered_and_gaps(ndvi_records, evi_records):
    builder = ObservationSequenceBuilder(require_pairs=False, max_gap_days=60)
    result = builder.build(ndvi_records, evi_records)
    assert result.observation_count == 2
    assert result.paired_count == 2
    assert result.sequence.sorted_dates == [date(2020, 7, 1), date(2020, 8, 1)]
    assert result.sequence.gap_days == [31.0]


def test_sequence_builder_flags_large_gap():
    ndvi = [_ref("NDVI", date(2020, 6, 1)), _ref("NDVI", date(2020, 9, 1))]
    evi = [_ref("EVI", date(2020, 6, 1)), _ref("EVI", date(2020, 9, 1))]
    result = ObservationSequenceBuilder(max_gap_days=30).build(ndvi, evi)
    assert any(i.code == "ST-TEMP-002" for i in result.issues)


def test_sequence_builder_missing_date_issue():
    ndvi = [_ref("NDVI", date(2020, 7, 1))]
    result = ObservationSequenceBuilder(require_pairs=False).build(ndvi, [])
    assert result.evi_count == 0
    assert result.sequence.evi_paths == []
