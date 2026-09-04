"""Tests for the correctness-first matcher audit (scripts/match_audit.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[3]

_spec = importlib.util.spec_from_file_location(
    "match_audit", _REPO / "scripts" / "match_audit.py"
)
_ma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ma)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["r1", "r2", "r3"],
            "crop_label": ["coconut", "pepper", "blackgram"],
            "satellite_status": ["FULL", "FULL", "FULL"],
            "temporal_match_status": ["EXACT_SEASON", "WITHIN_TOLERANCE", "EXACT_SEASON"],
            "spatial_match_distance_km": [0.1, 0.2, 0.3],
            "location_taluk": ["Bantwal", "Puttur", "Sullia"],
            "year": [2020, 2020, 2020],
            "season": ["Kharif", "Kharif", "Kharif"],
            "lat": [12.9, 12.8, 12.7],
            "lon": [75.1, 75.2, 75.3],
        }
    )


def test_classify_marks_supported_valid_as_preserved():
    classified = _ma.classify_rows(_frame())
    labels = classified.set_index("record_id")["_classification"]
    assert labels["r1"] == "PRESERVED"
    assert labels["r2"] == "PRESERVED"


def test_classify_removes_unsupported_crop():
    classified = _ma.classify_rows(_frame())
    labels = classified.set_index("record_id")["_classification"]
    assert labels["r3"] == "REMOVED"
    assert classified.set_index("record_id")["_reason"]["r3"] == "unsupported-crop"


def test_classify_removes_missing_imagery():
    df = _frame()
    df.loc[0, "satellite_status"] = "PARTIAL"
    assert _ma._single_reason(df.iloc[0]) == "missing-imagery"


def test_classify_removes_invalid_spatial_match():
    df = _frame()
    df.loc[0, "spatial_match_distance_km"] = 5.0
    assert _ma._single_reason(df.iloc[0]) == "invalid-spatial-match"


def test_leakage_audit_detects_taluk_disjointness():
    df = _frame()
    leak = _ma.leakage_audit(df)
    assert leak["taluks_in_multiple_splits"] == 0
    assert leak["exact_coords_in_multiple_splits"] == 0


def test_label_quality_counts_conflicts():
    df = _frame()
    issues = _ma.label_quality_issues(df)
    assert issues["class_counts"]["coconut"] == 1
    assert issues["unsupported_crops"] == {"blackgram": 1}


def test_exact_coord_label_conflict_detection():
    # Two different crops at the exact same coordinate + year + season.
    df = _frame()
    extra = df.iloc[[0]].copy()
    extra["record_id"] = "r4"
    extra["crop_label"] = "pepper"
    df = pd.concat([df, extra], ignore_index=True)
    issues = _ma.label_quality_issues(df)
    assert issues["exact_coord_label_conflicts"]["distinct_coords"] == 1
    assert issues["exact_coord_label_conflicts"]["rows_involved"] == 2
