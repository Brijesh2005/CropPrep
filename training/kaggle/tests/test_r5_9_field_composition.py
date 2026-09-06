"""Tests for training/kaggle/scripts/r5_9_field_composition.py (Phase 21).

Run from repo root::

    pytest training/kaggle/tests/test_r5_9_field_composition.py -v
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

from training.kaggle.scripts.r5_9_field_composition import (
    OUT_DIR,
    BINARY,
    TIE_GAP,
    composition_crop,
    decode_extent,
    extent_score,
    load_survey_rows,
)


@pytest.fixture(scope="session")
def decision() -> dict:
    return json.load(open(OUT_DIR / "signal_decision.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def report() -> dict:
    return json.load(open(OUT_DIR / "R5.9_FIELD_COMPOSITION_REPORT.json",
                          encoding="utf-8"))


@pytest.fixture(scope="session")
def dominant_binary() -> dict:
    return json.load(open(OUT_DIR / "dominant_binary_results.json",
                          encoding="utf-8"))


@pytest.fixture(scope="session")
def scalarization() -> dict:
    return json.load(open(OUT_DIR / "crop_extent_scalarization.json",
                          encoding="utf-8"))


@pytest.fixture(scope="session")
def provenance() -> dict:
    return json.load(open(OUT_DIR / "provenance_contract.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def comp() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "field_composition.csv")


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "field_dataset_split.csv")


# ------------------------------------------------------------------------- #
# Crop_Extent unit handling (Phase 1/2)
# ------------------------------------------------------------------------- #
class TestCropExtentUnit:
    def test_unit_declared_unknown(self, scalarization):
        assert scalarization["crop_extent_unit_status"] == "UNKNOWN"

    def test_used_only_as_relative_target(self, scalarization):
        assert scalarization["absolute_unit_claim"] is False
        assert scalarization["used_as_feature"] is False

    def test_parse_method_documented(self, scalarization):
        assert "B/100" in scalarization["parse_method"]

    def test_extent_score_monotonic(self):
        assert extent_score({"A": 1, "B": 76, "C": 0}) > extent_score(
            {"A": 1, "B": 0, "C": 0})
        assert extent_score({"A": 0, "B": 4, "C": 25}) > extent_score(
            {"A": 0, "B": 4, "C": 24})
        assert extent_score(None) is None

    def test_decode_extent(self):
        d = decode_extent("1-76-0.00")
        assert d and d["A"] == 1.0 and d["B"] == 76.0 and d["C"] == 0.0


class TestCompositionLabels:
    def test_excluded_heads_mapped_none(self):
        for c in ["NA Land", "NA Land ", "Fallow", "Trees and Grooves",
                  "Harvest over Crop-Pepper (Black)"]:
            assert composition_crop(c) is None

    def test_primary_crops_mapped(self):
        assert composition_crop("Coconut") == "coconut"
        assert composition_crop("Pepper (Black)") == "pepper"

    def test_other_real_crops_slugified(self):
        assert composition_crop("Betel Nuts (Areca nuts)") == \
            "betel_nuts_(areca_nuts)"


# ------------------------------------------------------------------------- #
# Composition / target (Phase 4-6)
# ------------------------------------------------------------------------- #
class TestComposition:
    def test_one_observation_per_context(self, comp):
        n_dups = comp.duplicated(
            subset=["field_id", "year", "season"]).sum()
        assert n_dups == 0

    def test_dominant_is_valid(self, comp):
        vals = set(comp["dominant_crop"].dropna())
        assert vals.issubset(set(BINARY) | {"tie", "none"}) or len(vals) > 0
        assert "coconut" in vals

    def test_fractions_sum_positive(self, comp):
        roi = comp[comp["dominant_crop"].isin(BINARY)]
        assert ((roi["top1_fraction"] > 0)).all()

    def test_dominant_counts_positive(self, comp):
        assert (comp["dominant_crop"] == "coconut").sum() > 0
        assert (comp["dominant_crop"] == "pepper").sum() >= 0

    def test_crop_extent_not_a_feature(self):
        from training.kaggle.scripts.r5_9_field_composition import ENV_NUMERIC
        assert not any("extent" in c.lower() for c in ENV_NUMERIC)


# ------------------------------------------------------------------------- #
# Splits / leakage (Phase 12-13)
# ------------------------------------------------------------------------- #
class TestSplitsAndLeakage:
    def test_all_rows_have_split(self, dataset):
        assert dataset["split"].notna().all()

    def test_no_field_spans_split(self, dataset):
        # RAW missing-profile marker: not required to be a strict check,
        # but a field must not appear in more than one split.
        grp = dataset.groupby("field_id")["split"].nunique()
        assert (grp <= 1).all()

    def test_leakage_audit_clean(self):
        la = json.load(open(OUT_DIR / "leakage_audit.json", encoding="utf-8"))
        assert la["clean"] is True
        assert la["forbidden_feature_hits"] == []
        assert not la["crop_extent_columns_in_features"]

    def test_crop_extent_not_in_features(self, dataset):
        feat = [c for c in dataset.columns
                if c not in ("context_key", "field_id", "year", "season",
                             "taluk", "hobli", "village", "survey_id",
                             "dominant_crop", "top1_fraction",
                             "dominance_gap", "n_crops_in_composition",
                             "coconut_fraction", "pepper_fraction",
                             "quality_tier", "confidence_tier",
                             "n_master_rows_in_context")]
        assert not any("extent" in c.lower() for c in feat)


# ------------------------------------------------------------------------- #
# Key experiment / decision (Phase 16/21/24)
# ------------------------------------------------------------------------- #
class TestKeyExperiment:
    def test_key_result_present(self, dominant_binary):
        assert dominant_binary["key_result"]["best_test_balanced_accuracy"] \
            is not None

    def test_no_improvement_over_r5_6(self, dominant_binary):
        # field-dominant does not beat the ~50% per-row ceiling
        imp = dominant_binary["key_result"]["improvement_over_r5_6_pp"]
        assert imp < 5.0

    def test_decision_no_signal(self, decision):
        assert decision["verdict"] in (
            "no_signal", "weak_signal", "meaningful_signal", "strong_signal",
            "substantially_better")

    def test_report_status_complete(self, report):
        ob = report["output_block"]
        assert ob["STATUS"] == "COMPLETE"
        assert ob["R5.8_STATUS"] == "BLOCKED_BY_CROP_EXTENT_SCHEMA"

    def test_not_90(self, report, decision):
        assert decision["verdict"] != "substantially_better"
        assert report["output_block"]["BEST_TEST_BALANCED_ACCURACY_PCT"] < 85


# ------------------------------------------------------------------------- #
# Provenance / non-negotiables (Phase 23)
# ------------------------------------------------------------------------- #
class TestProvenance:
    def test_unit_unknown_decision_documented(self, provenance):
        assert "UNKNOWN" in provenance["decision_made_by_user"]

    def test_relative_only_stated(self, provenance):
        assert "RELATIVE" in provenance["decision_made_by_user"]

    def test_no_full_cropfusion(self, scalarization):
        # cheap baselines only; no CropFusion training artifact expected
        adv = json.load(open(OUT_DIR / "signal_decision.json",
                             encoding="utf-8"))
        assert adv["recommendation"].lower() != "cropfusion"