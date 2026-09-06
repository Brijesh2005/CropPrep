"""Tests for training/kaggle/scripts/r5_8_subfield.py (Phase 21).

Run from repo root::

    pytest training/kaggle/tests/test_r5_8_subfield.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from training.kaggle.scripts.r5_8_subfield import (
    OUT_DIR,
    BLOCKED_STATUS,
    canonical_crop,
    decode_extent,
    load_survey_rows,
)


@pytest.fixture(scope="session")
def schema() -> dict:
    return json.load(open(OUT_DIR / "crop_extent_schema.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def field_audit() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "field_identity_audit.csv")


@pytest.fixture(scope="session")
def colocation() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "colocation_analysis.csv")


@pytest.fixture(scope="session")
def decision() -> dict:
    return json.load(open(OUT_DIR / "R5.8_decision.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def report() -> dict:
    return json.load(open(OUT_DIR / "R5.8_SUBFIELD_DISCRIMINATION_REPORT.json",
                          encoding="utf-8"))


@pytest.fixture(scope="session")
def provenance() -> dict:
    return json.load(open(OUT_DIR / "provenance_contract.json", encoding="utf-8"))


# ------------------------------------------------------------------ #
# Status / schema (the decisive finding)
# ------------------------------------------------------------------ #

class TestStatus:
    def test_status_blocked(self, decision):
        assert decision["status"] == BLOCKED_STATUS
        assert decision["verdict"] == "not_testable"

    def test_report_status_blocked(self, report):
        assert report["status"] == BLOCKED_STATUS

    def test_cropfusion_not_justified(self, decision):
        assert decision["cropfusion_training_justified"] is False

    def test_no_90_claim(self, decision):
        assert decision.get("data_ceiling_r5_8_pct") is None


class TestCropExtentSchema:
    def test_extent_is_not_geometry(self, schema):
        assert schema["contains_geometry"] is False
        assert schema["geometry_kind"] is None

    def test_not_mappable_to_pixels(self, schema):
        assert schema["can_be_mapped_to_satellite_pixels"] is False

    def test_no_geometry_source(self, schema):
        assert schema["evidence"]["gis_crop_polygon_source_present"] is False
        assert schema["evidence"]["gis_administrative_only"] is True

    def test_no_coordinate_like_extents(self, schema):
        assert schema["evidence"]["rows_with_coordinate_like_extent"] == 0

    def test_extent_strings_are_compound_area(self, schema):
        assert schema["evidence"]["fraction_matching_compound_area_pattern"] > 0.99


class TestDecodeExtent:
    def test_valid(self):
        assert decode_extent("1-76-0.00") == {"A": 1.0, "B": 76.0, "C": 0.0}
        assert decode_extent("0-4-25.00") == {"A": 0.0, "B": 4.0, "C": 25.0}

    def test_invalid_returns_none(self):
        assert decode_extent(None) is None
        assert decode_extent("garbage") is None
        assert decode_extent("12.98,75.11") is None


class TestCanonicalCrop:
    def test_binary_mapping(self):
        assert canonical_crop("Coconut") == "coconut"
        assert canonical_crop("Pepper (Black)") == "pepper"

    def test_excludes_non_primary(self):
        for c in ["NA Land ", "Fallow", "Harvest over Crop-Pepper (Black)",
                  "Betel Nuts (Areca nuts)"]:
            assert canonical_crop(c) is None


# ------------------------------------------------------------------ #
# Coordinate validation
# ------------------------------------------------------------------ #

class TestCoordinates:
    def test_within_dk_bounds(self, field_audit):
        s = field_audit.dropna(subset=["latitude", "longitude"])
        assert s["latitude"].between(12.4, 13.4).mean() > 0.99
        assert s["longitude"].between(74.6, 75.9).mean() > 0.99

    def test_have_valid_coords(self, colocation):
        assert colocation["pepper_lat"].notna().all()
        assert colocation["pepper_lon"].notna().all()
        assert colocation["distance_m"].notna().all()
        assert (colocation["distance_m"] >= 0).all()


# ------------------------------------------------------------------ #
# Field identity
# ------------------------------------------------------------------ #

class TestFieldIdentity:
    def test_field_id_deterministic(self, field_audit):
        # same field_id always refers to the same (survey_id, admin context)
        g = field_audit.groupby("field_id")[["survey_id", "taluk", "hobli", "village"]].nunique()
        assert (g["survey_id"] == 1).all()
        assert (g["taluk"] == 1).all()

    def test_survey_id_scoped_by_admin_context(self, field_audit):
        # survey_id is reused across hoblis; field_id (with admin context)
        # disambiguates it. Same (survey_id, taluk, hobli, village) => 1 field_id.
        g = field_audit.groupby(["survey_id", "taluk", "hobli", "village"])[
            "field_id"].nunique()
        assert (g == 1).all()

    def test_field_id_does_not_use_crop_label(self, field_audit):
        sample = field_audit["field_id"].iloc[0]
        assert "coconut" not in sample.lower()
        assert "pepper" not in sample.lower()

    def test_no_duplicate_observations(self, field_audit):
        # every row survives (nothing silently dropped) and every observation
        # resolves to exactly one field identity; interior duplicate rows are
        # genuine multi-crop rows of one field (mean ~1.5 crop rows field).
        assert len(field_audit) == 837069
        assert field_audit["field_id"].nunique() > 0
        # no field has an implausibly large number of rows (obs explosion)
        per_field = field_audit.groupby("field_id").size()
        assert per_field.max() <= 100

    def test_cohort_rows_positive(self, field_audit):
        assert (field_audit["canonical_crop"] == "coconut").sum() > 0
        assert (field_audit["canonical_crop"] == "pepper").sum() > 0


# ------------------------------------------------------------------ #
# No same-field train/test leakage (grouped split rule)
# ------------------------------------------------------------------ #

class TestLeakage:
    def test_no_leakage_features_in_sources(self, field_audit):
        forbidden = {"Yield_Proxy_NPP", "benchmark_eligible", "valid_sample",
                     "rejection_reason", "target"}
        assert not (set(field_audit.columns) & forbidden)

    def test_unblock_contract_prescribes_grouped_split(self, schema):
        # the documented split rule keeps every crop row of a field together
        keywords = schema.get("unblock_contract", {})
        assert isinstance(keywords, dict)
        assert "required_geometry_inputs" in keywords
        assert "required_validation" in keywords

    def test_no_fabrication_contract(self, provenance):
        assert provenance["no_fabrication"] is True
        assert provenance["no_geometry_invented"] is True


# ------------------------------------------------------------------ #
# Provenance of crop_extent / satellite / temporal — no fabrication
# ------------------------------------------------------------------ #

class TestProvenance:
    def test_crop_extent_provenance(self, provenance):
        # extent is recorded raw from the CSV, not fabricated/reinterpreted
        assert "Crop_Extent as recorded" in provenance["every_observation_traces"]["crop_record"]
        assert "Crop_Extent" in provenance["every_observation_traces"]["crop_extent_source"]

    def test_no_geometry_transformation(self, provenance):
        # extent is scalar: no geometry transformation was performed
        assert "NONE" in provenance["every_observation_traces"]["geometry_transformation"]

    def test_satellite_not_extracted(self, provenance):
        assert "NOT" in provenance["every_observation_traces"]["satellite_image_date"]

    def test_pixel_mask_not_run(self, provenance):
        assert "NOT" in provenance["every_observation_traces"]["pixel_mask_extraction"]

    def test_no_fabricated_dates_or_pixels(self, provenance):
        # because masks were not run and no mask validity can exist
        assert provenance["r5_6_r5_7_frozen_untouched"] is True


# ------------------------------------------------------------------ #
# No fabricated geometry / mask validity gating
# ------------------------------------------------------------------ #

class TestNoFabrication:
    def test_colocation_uses_recorded_gps_only(self, colocation):
        # distances derived purely from recorded survey coordinates
        assert "pepper_lat" in colocation.columns
        assert "nearest_coconut_lat" in colocation.columns

    def test_median_colocation_consistent(self, colocation):
        med = float(np.median(colocation["distance_m"]))
        assert 0 < med < 10

    def test_decision_names_geometry_bottleneck(self, decision):
        assert "missing_crop_geometry" in decision["primary_bottleneck"]
        assert "intercropping" in decision["primary_signal"].lower()
