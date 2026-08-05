"""Unit tests for the quality-control pass."""

from __future__ import annotations

from datetime import date

from training.stam.config import QualityConfig
from training.stam.observation import (
    ImagePairRef,
    ImageRecordRef,
    LocationInfo,
    SequenceInfo,
    TabularFeatures,
    TemporalInfo,
)
from training.stam.validators import assess_quality


def _base_parts():
    return {
        "config": QualityConfig(),
        "location": LocationInfo(lon=74.8, lat=13.1, distance_km=0.1),
        "temporal": TemporalInfo(year=2020, season="Kharif", tolerance_days=15),
        "tabular": TabularFeatures(crop="Rice", yield_value=5200.0,
                                   fields={"rainfall": 2100}, matched_level="village"),
        "sequence": SequenceInfo(),
    }


def _pair(day: date, index: str):
    return ImageRecordRef(
        path=f"/data/{index}_{day}.tif", relative_path=f"{index}_{day}.tif",
        index_type=index, resolution="R10m", observation_date=day,
        crs="EPSG:4326", pixel_size=(0.0001, 0.0001),
        bounds=(74.8, 13.0, 74.9, 13.1), width=20, height=20,
    )


def test_healthy_observation_passes():
    parts = _base_parts()
    ndvi = _pair(date(2020, 7, 1), "NDVI")
    parts["sequence"] = SequenceInfo(
        pairs=[ImagePairRef(date=date(2020, 7, 1), ndvi=ndvi,
                            evi=_pair(date(2020, 7, 1), "EVI"),
                            resolution="R10m", crs="EPSG:4326",
                            quality={"paired": True})],
        sorted_dates=[date(2020, 7, 1)],
        resolution="R10m", crs="EPSG:4326",
    )
    report = assess_quality(**parts)
    assert report.passed is True
    assert report.overall_score > 90


def test_invalid_coordinates_critical():
    parts = _base_parts()
    parts["location"] = LocationInfo(lon=500.0, lat=13.1, distance_km=0.1)
    report = assess_quality(**parts)
    assert report.passed is False
    assert any(i.code == "ST-Q-COORD-001" for i in report.issues)


def test_far_location_warns():
    parts = _base_parts()
    parts["location"] = LocationInfo(lon=74.8, lat=13.1, distance_km=8.0)
    # Give the observation a valid image pair so the only issue is the
    # low-confidence-location warning.
    ndvi = _pair(date(2020, 7, 1), "NDVI")
    parts["sequence"] = SequenceInfo(
        pairs=[ImagePairRef(date=date(2020, 7, 1), ndvi=ndvi,
                            evi=_pair(date(2020, 7, 1), "EVI"),
                            resolution="R10m", crs="EPSG:4326",
                            quality={"paired": True})],
        sorted_dates=[date(2020, 7, 1)], resolution="R10m", crs="EPSG:4326",
    )
    report = assess_quality(**parts, distance_threshold_km=5.0)
    assert any(i.code == "ST-Q-LOC-001" for i in report.issues)
    # Warning alone does not fail the report.
    assert report.passed is True


def test_no_tabular_record_fails():
    parts = _base_parts()
    parts["tabular"] = TabularFeatures(crop=None, yield_value=None, fields={},
                                       matched_level="none")
    report = assess_quality(**parts)
    assert any(i.code == "ST-Q-TAB-001" for i in report.issues)
    assert report.passed is False


def test_no_images_fails():
    parts = _base_parts()
    parts["sequence"] = SequenceInfo(resolution="R10m", crs="EPSG:4326")
    report = assess_quality(**parts)
    assert any(i.code == "ST-Q-IMG-001" for i in report.issues)


def test_missing_side_warns():
    parts = _base_parts()
    ndvi = _pair(date(2020, 7, 1), "NDVI")
    parts["sequence"] = SequenceInfo(
        pairs=[ImagePairRef(date=date(2020, 7, 1), ndvi=ndvi, evi=None,
                            resolution="R10m", crs="EPSG:4326",
                            quality={"paired": False, "missing": ["EVI"]})],
        sorted_dates=[date(2020, 7, 1)], resolution="R10m", crs="EPSG:4326",
    )
    report = assess_quality(**parts)
    assert any(i.code == "ST-Q-IMG-002" for i in report.issues)


def test_temporal_gap_warns():
    parts = _base_parts()
    a = _pair(date(2020, 6, 1), "NDVI")
    b = _pair(date(2020, 9, 1), "NDVI")
    parts["sequence"] = SequenceInfo(
        pairs=[
            ImagePairRef(date=date(2020, 6, 1), ndvi=a, evi=_pair(date(2020, 6, 1), "EVI"),
                         resolution="R10m", crs="EPSG:4326", quality={"paired": True}),
            ImagePairRef(date=date(2020, 9, 1), ndvi=b, evi=_pair(date(2020, 9, 1), "EVI"),
                         resolution="R10m", crs="EPSG:4326", quality={"paired": True}),
        ],
        sorted_dates=[date(2020, 6, 1), date(2020, 9, 1)],
        gap_days=[92.0], resolution="R10m", crs="EPSG:4326",
    )
    report = assess_quality(**parts)
    assert any(i.code == "ST-Q-TEMP-001" for i in report.issues)


def test_pair_invalid_flags_error():
    parts = _base_parts()
    ndvi = _pair(date(2020, 7, 1), "NDVI")
    parts["sequence"] = SequenceInfo(
        pairs=[ImagePairRef(date=date(2020, 7, 1), ndvi=ndvi,
                            evi=_pair(date(2020, 7, 1), "EVI"),
                            resolution="R10m", crs="EPSG:4326",
                            quality={"paired": False, "invalid": True})],
        sorted_dates=[date(2020, 7, 1)], resolution="R10m", crs="EPSG:4326",
    )
    report = assess_quality(**parts)
    assert any(i.code == "ST-Q-PAIR-001" for i in report.issues)
    assert report.passed is False
