"""Tests for the R2.2 validator extensions.

Covers the temporal (duplicate records / missing years), spatial (coordinate
ranges / duplicate locations), CRS-consistency and provider-availability
checks added to :class:`DatasetValidator`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import pytest

from training.dataset_manager.tests.helpers import make_tiff


def _write_csv(catalog: Path, rows: list[dict], name: str = "crop_yield.csv") -> Path:
    pd.DataFrame(rows).to_csv(catalog / name, index=False)
    return catalog / name


@pytest.fixture
def duplicate_dataset(tmp_path: Path) -> Path:
    """Two NDVI rasters sharing index/resolution/year/observation date."""
    root = tmp_path / "datasets"
    catalog = root / "raw" / "kaggle-crop-yield"
    (catalog / "2019_images" / "R10m").mkdir(parents=True)
    make_tiff(catalog / "2019_images" / "R10m" / "S2_NDVI_20190701.tif", seed=1)
    make_tiff(catalog / "2019_images" / "R10m" / "S2_NDVI_20190701_dup.tif", seed=1)
    _write_csv(catalog, [{"village": "A", "yield_kg": 1}])
    return root


@pytest.fixture
def spatial_bad_dataset(tmp_path: Path) -> Path:
    """A village with an out-of-range latitude."""
    root = tmp_path / "datasets"
    catalog = root / "raw" / "kaggle-crop-yield"
    (catalog / "2019_images" / "R10m").mkdir(parents=True)
    make_tiff(catalog / "2019_images" / "R10m" / "S2_NDVI_20190701.tif", seed=1)
    _write_csv(
        catalog,
        [
            {"village": "Bad", "latitude": 95.0, "longitude": 74.89, "year": 2019},
            {"village": "Good", "latitude": 13.0, "longitude": 75.0, "year": 2019},
        ],
    )
    return root


@pytest.fixture
def mixed_crs_dataset(tmp_path: Path) -> Path:
    """Two rasters using different coordinate systems."""
    root = tmp_path / "datasets"
    catalog = root / "raw" / "kaggle-crop-yield"
    (catalog / "2019_images" / "R10m").mkdir(parents=True)
    (catalog / "2020_images" / "R10m").mkdir(parents=True)
    make_tiff(catalog / "2019_images" / "R10m" / "S2_NDVI_20190701.tif", seed=1, crs="EPSG:32643")
    make_tiff(catalog / "2020_images" / "R10m" / "S2_NDVI_20200701.tif", seed=2, crs="EPSG:4326")
    _write_csv(catalog, [{"village": "A", "yield_kg": 1}])
    return root


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_temporal_duplicate_records_detected(duplicate_dataset, manager_factory):
    manager = manager_factory(duplicate_dataset)
    report = manager.validate()
    assert "V-TEMP-001" in _codes(report)
    dup = next(i for i in report.issues if i.code == "V-TEMP-001")
    assert dup.severity.value == "warning"
    assert len(dup.detail["files"]) == 2


def test_temporal_missing_years_detected(duplicate_dataset, manager_factory):
    manager = manager_factory(duplicate_dataset)
    report = manager.validate()
    assert "V-TEMP-002" in _codes(report)
    issue = next(i for i in report.issues if i.code == "V-TEMP-002")
    assert 2018 in issue.detail["missing_years"]


def test_spatial_out_of_range_latitude(spatial_bad_dataset, manager_factory):
    manager = manager_factory(
        spatial_bad_dataset,
        settings_overrides={"providers": {"tabular": {"root": str(
            spatial_bad_dataset / "raw" / "kaggle-crop-yield"
        )}}},
    )
    report = manager.validate()
    assert "V-SPAT-001" in _codes(report)
    issue = next(i for i in report.issues if i.code == "V-SPAT-001")
    assert issue.detail["latitude"] == 95.0
    assert issue.severity.value == "error"


def test_spatial_valid_locations_pass(spatial_bad_dataset, manager_factory):
    manager = manager_factory(
        spatial_bad_dataset,
        settings_overrides={"providers": {"tabular": {"root": str(
            spatial_bad_dataset / "raw" / "kaggle-crop-yield"
        )}}},
    )
    records = manager.spatial_index.records()
    good = [r for r in records if r.name == "Good"]
    assert good and good[0].latitude == 13.0


def test_crs_mismatch_detected(mixed_crs_dataset, manager_factory):
    manager = manager_factory(mixed_crs_dataset)
    report = manager.validate()
    assert "V-CRS-001" in _codes(report)
    issue = next(i for i in report.issues if i.code == "V-CRS-001")
    assert issue.severity.value == "warning"
    assert issue.detail["majority"] == "EPSG:32643"


def test_provider_unavailable_detected(duplicate_dataset, manager_factory):
    manager = manager_factory(duplicate_dataset)

    class _DeadProvider:
        name = "dead_image"
        kind = "image"

        def available(self):
            return False

    manager.provider_registry.register(
        "dead_image", "image", _DeadProvider(), enabled=True, priority=1
    )
    report = manager.validate()
    assert "V-PROV-001" in _codes(report)
    issue = next(i for i in report.issues if i.code == "V-PROV-001")
    assert issue.detail["name"] == "dead_image"


def test_extended_checks_skipped_without_dependencies(duplicate_dataset, manager_factory):
    manager = manager_factory(duplicate_dataset)
    validator = manager.validator
    validator.spatial_index = None
    validator.provider_registry = None
    report = manager.validate()
    codes = _codes(report)
    assert "V-SPAT-000" not in codes
    assert "V-PROV-000" not in codes
