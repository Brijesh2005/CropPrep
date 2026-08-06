"""Tests for the seven R2.2 report builders + :func:`generate_reports`.

Reports must render through the Dataset Manager surface (never direct file
reads), be JSON-serialisable, and be written under the report directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from training.dataset_manager.reports import REPORT_BUILDERS, generate_reports


def test_generate_reports_writes_seven_files(r22_manager_factory):
    manager = r22_manager_factory()
    paths = manager.reports()
    assert len(paths) == 7
    names = {p.name for p in paths}
    assert names == {
        "inventory_report.json",
        "csv_report.json",
        "image_report.json",
        "provider_report.json",
        "spatial_report.json",
        "temporal_report.json",
        "validation_report.json",
    }
    for path in paths:
        assert path.is_file()
        json.loads(path.read_text(encoding="utf-8"))


def test_generate_reports_respects_custom_dir(r22_manager_factory, tmp_path: Path):
    manager = r22_manager_factory()
    out = tmp_path / "my_reports"
    paths = manager.reports(report_dir=out)
    assert all(p.parent == out for p in paths)


def test_inventory_report(r22_manager_factory):
    manager = r22_manager_factory()
    report = REPORT_BUILDERS["inventory"](manager)
    assert report["kind"] == "inventory"
    assert report["counts"]["geotiff"] == 3
    assert any(e["category"] == "geotiff" for e in report["files"])


def test_csv_report(r22_manager_factory):
    manager = r22_manager_factory()
    report = REPORT_BUILDERS["csv"](manager)
    assert report["kind"] == "csv"
    assert report["count"] >= 1
    dataset = report["datasets"][0]
    assert dataset["rows"] == 3
    assert "village" in dataset["columns"]
    assert dataset["total_missing"] == 0


def test_image_report(r22_manager_factory):
    manager = r22_manager_factory()
    report = REPORT_BUILDERS["image"](manager)
    assert report["kind"] == "image"
    assert report["ndvi_count"] == 2
    assert report["evi_count"] == 1
    assert report["by_year_index"]["2019"] == {"ndvi": 1, "evi": 1}
    assert report["by_year_index"]["2020"] == {"ndvi": 1, "evi": 0}


def test_provider_report(r22_manager_factory):
    manager = r22_manager_factory()
    report = REPORT_BUILDERS["provider"](manager)
    assert report["kind"] == "provider"
    names = {reg["name"] for reg in report["registered"]}
    assert names == {"git_repository_tabular", "kaggle_hub_image"}
    assert report["availability"]["git_repository_tabular"] is True
    assert "kaggle_hub_image" in report["health"]
    assert "kaggle_hub_image" in report["capabilities"]


def test_spatial_report(r22_manager_factory):
    manager = r22_manager_factory()
    report = REPORT_BUILDERS["spatial"](manager)
    assert report["kind"] == "spatial"
    assert report["count"] == 3
    assert report["metadata"]["villages"] == 3
    names = {loc["name"] for loc in report["locations"]}
    assert names == {"Moodabidri", "Bantwal", "Sullia"}


def test_temporal_report(r22_manager_factory):
    manager = r22_manager_factory()
    report = REPORT_BUILDERS["temporal"](manager)
    assert report["kind"] == "temporal"
    keys = {(r["index_type"], r["year"]) for r in report["records"]}
    assert ("NDVI", 2019) in keys
    assert ("EVI", 2019) in keys
    assert ("NDVI", 2020) in keys


def test_validation_report(r22_manager_factory):
    manager = r22_manager_factory()
    report = REPORT_BUILDERS["validation"](manager)
    assert report["kind"] == "validation"
    assert report["files_scanned"] == 4  # 3 tiffs + 1 csv
    assert "issues" in report and "by_severity" in report


def test_single_broken_report_does_not_block_generate(r22_manager_factory, monkeypatch):
    manager = r22_manager_factory()

    def _boom(manager):
        raise RuntimeError("boom")

    monkeypatch.setitem(REPORT_BUILDERS, "inventory", _boom)
    paths = manager.reports()
    assert len(paths) == 7
    inventory = json.loads(
        (paths[0] if paths[0].name == "inventory_report.json" else paths[1]).read_text(
            encoding="utf-8"
        )
    )
    assert inventory["error"] == "boom"
