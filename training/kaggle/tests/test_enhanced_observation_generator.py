"""Tests for the R5.2.9 enhanced observation generator (audit/build/validate)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.kaggle.enhanced_observation_generator import (
    R529Config,
    run_audit,
    run_build,
    run_validate,
)

_CAT = ["Is_Cropland", "Land_Cover_Class", "Soil_Type_Class"]


def _make_dk_grid(tmp_path):
    cells = []
    for row in range(3):
        for col in range(3):
            cells.append(
                {
                    "system:index": f"{row}_{col}",
                    "Latitude": 12.70 + row * 0.01,
                    "Longitude": 75.00 + col * 0.01,
                    "Year": 2020,
                    "Season": "Annual + Kharif/Rabi composites",
                    "Annual_Rainfall_mm": 2000.0 + row * 100 + col * 50,
                    "NDVI": 0.2 + row * 0.1 + col * 0.05,
                    "EVI": 1.0 + row + col,
                    "Kharif_NDVI": 0.3 + row * 0.1 + col * 0.05,
                    "Kharif_EVI": 1.5 + row + col,
                    "Kharif_NDWI": -0.3 - row * 0.05,
                    "Rabi_NDVI": 0.1 + row * 0.05 + col * 0.02,
                    "Rabi_EVI": 0.8 + row * 0.5,
                    "Rabi_NDWI": -0.4 - col * 0.05,
                    "Soil_pH": 56.0 + row,
                    "Land_Cover_Class": f"lc{row}",
                    "Is_Cropland": "1",
                    "Soil_Type_Class": f"soil{col}",
                }
            )
    frame = pd.DataFrame(cells)
    frame.to_csv(tmp_path / "DK_Features_2020.csv", index=False)


def _make_v1_supervised(tmp_path):
    rows = [
        {
            "record_id": "gov_TEST_VILL_2020_Kharif_coconut_12.70_75.00",
            "source": "government_ogd",
            "source_record_id": "r1",
            "crop_label": "coconut",
            "crop_class_id": "4",
            "source_crop_name": "Coconut",
            "location_hobli": "TEST",
            "location_taluk": "Puttur",
            "location_village": "VILL",
            "location_district": "Dakshina Kannada",
            "lat": "12.70",
            "lon": "75.00",
            "year": "2020",
            "season": "Kharif",
            "survey_date": "2020-09-01",
            "spatial_match_distance_km": "0.0",
            "temporal_match_status": "EXACT_SEASON",
            "tabular_source": "district_grid",
            "image_source": "sentinel2",
            "ndvi_available": "True",
            "evi_available": "True",
            "satellite_status": "FULL",
        },
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "crop_supervised_v1.csv", index=False)


def _make_v1_ledger(tmp_path):
    rows = [
        {
            "source_crop": "Coconut",
            "crop_type": "coconut",
            "crop_status": "exact",
            "hobli": "TEST",
            "taluk": "Puttur",
            "village": "VILL",
            "lat": "12.70",
            "lon": "75.00",
            "year": "2020",
            "season": "Kharif",
            "survey_date": "2020-09-01",
            "distance_km": "0.0",
            "spatial_status": "MATCHED",
            "temporal_status": "EXACT_SEASON",
            "tabular_matched": "True",
            "tabular_level": "district_grid",
            "satellite_status": "FULL",
            "ndvi_available": "True",
            "evi_available": "True",
            "is_duplicate": "False",
            "valid_cropfusion_sample": "True",
            "rejection_reasons": "[]",
        },
        {
            "source_crop": "Pepper (Black)",
            "crop_type": "pepper",
            "crop_status": "alias",
            "hobli": "TEST",
            "taluk": "Belthangady",
            "village": "VILL2",
            "lat": "12.71",
            "lon": "75.01",
            "year": "2020",
            "season": "Kharif",
            "survey_date": "2020-08-15",
            "distance_km": "NaN",
            "spatial_status": "NO_MATCH",
            "temporal_status": "EXACT_SEASON",
            "tabular_matched": "True",
            "tabular_level": "district_grid",
            "satellite_status": "NOT_AVAILABLE",
            "ndvi_available": "False",
            "evi_available": "False",
            "is_duplicate": "False",
            "valid_cropfusion_sample": "False",
            "rejection_reasons": "['spatial_NO_MATCH', 'no_satellite']",
        },
        {
            "source_crop": "Coconut",
            "crop_type": "coconut",
            "crop_status": "exact",
            "hobli": "TEST",
            "taluk": "Sullia",
            "village": "VILL3",
            "lat": "12.72",
            "lon": "75.02",
            "year": "2020",
            "season": "Kharif",
            "survey_date": "2020-11-25",
            "distance_km": "NaN",
            "spatial_status": "NO_MATCH",
            "temporal_status": "OUTSIDE_TOLERANCE",
            "tabular_matched": "True",
            "tabular_level": "district_grid",
            "satellite_status": "FULL",
            "ndvi_available": "True",
            "evi_available": "True",
            "is_duplicate": "False",
            "valid_cropfusion_sample": "False",
            "rejection_reasons": "['temporal_OUTSIDE_TOLERANCE']",
        },
        {
            "source_crop": "Coconut",
            "crop_type": "coconut",
            "crop_status": "exact",
            "hobli": "TEST",
            "taluk": "Puttur",
            "village": "VILL",
            "lat": "12.70",
            "lon": "75.00",
            "year": "2020",
            "season": "Kharif",
            "survey_date": "2020-09-01",
            "distance_km": "0.0",
            "spatial_status": "MATCHED",
            "temporal_status": "EXACT_SEASON",
            "tabular_matched": "True",
            "tabular_level": "district_grid",
            "satellite_status": "FULL",
            "ndvi_available": "True",
            "evi_available": "True",
            "is_duplicate": "True",
            "valid_cropfusion_sample": "False",
            "rejection_reasons": "['duplicate']",
        },
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "government_crop_stam_match.csv", index=False)


@pytest.fixture
def env(tmp_path):
    _make_dk_grid(tmp_path)
    _make_v1_supervised(tmp_path)
    _make_v1_ledger(tmp_path)
    out_dir = tmp_path / "out"
    reports = tmp_path / "reports"
    out_dir.mkdir()
    reports.mkdir()
    cfg = R529Config(
        root=tmp_path,
        version="r5.2.9",
        max_search_radius_km=1.0,
        knn_k=2,
        idw_power=2.0,
        duplicate_tolerance_m=50.0,
        tolerance_days=15,
        temporal_relaxation_days=45,
        dk_years=[2020],
        exclude_columns=["Yield_Proxy_NPP"],
        categorical_features=_CAT,
        files={
            "v1_matched": str(tmp_path / "government_crop_stam_match.csv"),
            "v1_supervised": str(tmp_path / "crop_supervised_v1.csv"),
            "v1_manifest": str(tmp_path / "manifest.json"),
            "dk_dir": str(tmp_path),
            "out_dir": str(out_dir),
            "reports_dir": str(reports),
        },
    )
    return cfg, tmp_path


def test_audit_counts(env):
    cfg, _ = env
    audit = run_audit(cfg)
    assert audit["scan_set"]["records"] == 4
    assert audit["scan_set"]["accepted"] == 1
    assert audit["scan_set"]["rejected"] == 3
    assert audit["rejection_reasons_literal"]["duplicate"] == 1
    assert audit["recovery_potential"] == {
        "non_duplicate_rejected": 2,
        "with_full_satellite_imagery": 1,
        "blocked_missing_imagery": 1,
    }


def test_audit_csv_written(env):
    cfg, _ = env
    run_audit(cfg)
    path = Path(cfg.files["reports_dir"]) / "R5.2.9_rejection_audit.csv"
    df = pd.read_csv(path)
    assert len(df) == 4
    assert set(df["primary_reason"]) == {
        "accepted",
        "duplicate",
        "spatial_NO_MATCH",
        "temporal_OUTSIDE_TOLERANCE",
    }


def test_build_recovers_and_enriches(env):
    cfg, _ = env
    result = run_build(cfg)
    assert result.summary["matched_records"] == 3
    assert result.summary["supervised_records"] == 2
    assert result.summary["benchmark_eligible"] == 1
    assert result.summary["recovered_observations"] == 1
    assert result.summary["unmatched_rows"] == 0

    sup = pd.read_csv(result.supervised_path, dtype=str)
    rec = sup[sup["is_recovered_v2"] == "True"]
    assert len(rec) == 1
    row = rec.iloc[0]
    assert row["crop_label"] == "coconut"
    assert row["temporal_match_status"] == "WITHIN_RELAXED_TOLERANCE"
    assert row["benchmark_eligible"] == "False"
    assert row["env_match_confidence"] in {"HIGH", "MEDIUM"}
    assert "annual_rainfall_mm" in sup.columns
    assert "yield_proxy" not in " ".join(sup.columns).lower()

    matched = pd.read_csv(result.matched_path, dtype=str)
    assert len(matched) == 3
    bills = pd.read_csv(Path(result.matched_path).parent / "crop_supervised_v2.csv", dtype=str)
    assert len(bills) == 2


def test_build_writes_provenance(env):
    cfg, _ = env
    result = run_build(cfg)
    prov = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert prov["release"] == "r5.2.9"
    records = {r["record_id"] for r in prov["records"]}
    assert len(prov["records"]) == 3
    assert "env_dk_index" in prov["records"][0]


def test_missing_imagery_not_recovered(env):
    cfg, _ = env
    result = run_build(cfg)
    sup = pd.read_csv(result.supervised_path, dtype=str)
    # The pepper record had NO satellite imagery: must stay rejected.
    assert (sup["crop_label"] == "pepper").sum() == 0
    matched = pd.read_csv(result.matched_path, dtype=str)
    pepper = matched[matched["crop_type"] == "pepper"]
    assert pepper.iloc[0]["valid_cropfusion_sample"] == "False"


def test_validate_passes_on_build(env):
    cfg, _ = env
    result = run_build(cfg)
    ok, errors = run_validate(cfg, result)
    assert ok, errors


def test_validate_catches_removed_baseline(env):
    cfg, _ = env
    result = run_build(cfg)
    sup_path = result.supervised_path
    sup = pd.read_csv(sup_path, dtype=str)
    sup.drop(index=0, inplace=True)
    sup.to_csv(sup_path, index=False)
    ok, errors = run_validate(cfg, result)
    assert not ok
    assert any("missing from v2" in e for e in errors)


def test_validate_catches_mutated_label(env):
    cfg, _ = env
    result = run_build(cfg)
    sup_path = result.supervised_path
    sup = pd.read_csv(sup_path, dtype=str)
    sup.loc[0, "crop_label"] = "coffee"
    sup.to_csv(sup_path, index=False)
    ok, errors = run_validate(cfg, result)
    assert not ok
    assert any("mutated in v2" in e for e in errors)


def test_validate_catches_leakage_column(env):
    cfg, _ = env
    result = run_build(cfg)
    sup_path = result.supervised_path
    sup = pd.read_csv(sup_path, dtype=str)
    sup["Yield_Proxy_NPP"] = 5.0
    sup.to_csv(sup_path, index=False)
    ok, errors = run_validate(cfg, result)
    assert not ok
    assert any("leakage columns present" in e for e in errors)


def test_validate_catches_duplicate_recovered_point(env):
    cfg, _ = env
    result = run_build(cfg)
    sup_path = result.supervised_path
    sup = pd.read_csv(sup_path, dtype=str)
    # Move the recovered row on top of the baseline row (same GPS point).
    recovered_idx = sup.index[sup["is_recovered_v2"] == "True"][0]
    sup.loc[recovered_idx, "lat"] = "12.70"
    sup.loc[recovered_idx, "lon"] = "75.00"
    sup.to_csv(sup_path, index=False)
    ok, errors = run_validate(cfg, result)
    assert not ok
    assert any("within" in e for e in errors)