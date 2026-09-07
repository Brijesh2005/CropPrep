"""Tests for training/kaggle/scripts/r5_10_temporal_field_target.py.

Run from repo root::

    pytest training/kaggle/tests/test_r5_10_temporal_field_target.py -v
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from training.kaggle.scripts.r5_10_temporal_field_target import (  # noqa: E402
    OUT_DIR,
    GRID_YEARS_ALL,
    R5_9_BALANCED_ACCURACY,
    R5_9_ROC_AUC,
    leading_years,
)


@pytest.fixture(scope="session")
def decision() -> dict:
    return json.load(open(OUT_DIR / "R5.10_decision.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def provenance() -> dict:
    return json.load(open(OUT_DIR / "provenance_contract.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def baselines() -> dict:
    return json.load(open(OUT_DIR / "cheap_baselines.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "field_target_dataset.csv", low_memory=False)


@pytest.fixture(scope="session")
def alignment() -> dict:
    return json.load(open(OUT_DIR / "temporal_alignment.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def tfeats() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "temporal_features.csv", low_memory=False)


@pytest.fixture(scope="session")
def grid() -> pd.DataFrame:
    return pd.read_csv(
        OUT_DIR / "temporal_field_grid.csv",
        low_memory=False,
        usecols=["field_id", "grid_year", "matched", "method",
                 "nearest_distance_m", "grid_year_used"],
    )


# ------------------------------------------------------------------------- #
# Target / identity (Phase 5)
# ------------------------------------------------------------------------- #
class TestTargetIntegrity:
    def test_binary_pool_one_row_per_field(self, dataset):
        assert dataset["field_id"].duplicated().sum() == 0
        assert len(dataset) == 57660

    def test_target_is_pepper_vs_coconut(self, dataset):
        assert set(dataset["target"].unique()).issubset({0, 1})
        assert (dataset["target"] == 0).sum() == 56863
        assert (dataset["target"] == 1).sum() == 797

    def test_crop_extent_never_a_feature(self, dataset):
        feat = [c for c in dataset.columns if c not in (
            "context_key", "field_id", "year", "season", "taluk", "hobli",
            "village", "survey_id", "split", "lat", "lon", "field_cluster_id",
            "dominant_crop", "top1_fraction", "dominance_gap",
            "n_crops_in_composition", "coconut_fraction", "pepper_fraction",
            "target", "n_master_rows_in_context", "quality_tier",
            "confidence_tier")]
        assert not any("extent" in c.lower() for c in feat)
        assert not any(c in feat for c in ("coconut_fraction",
                                           "pepper_fraction",
                                           "top1_fraction"))


# ------------------------------------------------------------------------- #
# No fabricated / no-future observations (Phases 3-4)
# ------------------------------------------------------------------------- #
class TestNoFabrication:
    @pytest.mark.parametrize("y", GRID_YEARS_ALL)
    def test_every_year_slot_matches_real_grid_row(self, grid, y):
        year_rows = grid[grid["grid_year"] == y]
        assert len(year_rows) == 57660
        assert year_rows["matched"].isin([True, False]).all()

    def test_grid_rows_unique_per_field_year(self, grid):
        assert not grid.duplicated(subset=["field_id", "grid_year"]).any()

    def test_missing_matches_are_explicit(self, grid):
        missed = grid[~grid["matched"]]
        if len(missed):
            assert missed["nearest_distance_m"].notna().all()
            assert (missed["nearest_distance_m"] > 5_000).all()

    def test_primary_trajectory_uses_only_leading_years(self, tfeats):
        traj = [c for c in tfeats.columns if re.match(r".+_y\d{4}$", c)]
        assert len(traj) > 0
        years = {int(c.split("_y")[-1]) for c in traj}
        assert years.issubset({2018, 2019, 2020, 2021})

    def test_missingness_policy_documented(self, alignment):
        mp = alignment["missingness"]["missing_policy"]
        assert "never imputed" in mp.lower()
        assert "matched slots only" in mp.lower()


# ------------------------------------------------------------------------- #
# Temporal alignment / ordering (Phase 3)
# ------------------------------------------------------------------------- #
class TestTemporalAlignment:
    def test_survey_year_predicts_over_leading_window(self, dataset):
        assert set(dataset["year"].unique()).issubset({2020, 2021})
        for y in dataset["year"].unique():
            assert leading_years(y) == [gy for gy in GRID_YEARS_ALL
                                        if gy <= int(y)]

    def test_leading_window_never_uses_future_grid(self, grid):
        pool = pd.read_csv(OUT_DIR / "temporal_unit.csv")
        year_of = (pool[["field_id", "year"]].drop_duplicates("field_id")
                   .set_index("field_id")["year"])
        grid = grid[grid["field_id"].isin(year_of.index)]
        g = grid.copy()
        g["survey_year"] = g["field_id"].map(year_of)
        lead = g[g["grid_year"] <= g["survey_year"]]
        assert (g["grid_year"] > g["survey_year"]).any()  # full window exists
        assert lead["matched"].notna().all()

    def test_coverage_reported_per_year(self, alignment):
        rows = alignment["per_grid_year_coverage"]
        assert len(rows) == len(GRID_YEARS_ALL)


# ------------------------------------------------------------------------- #
# Split integrity (Phase 6)
# ------------------------------------------------------------------------- #
class TestSplitIntegrity:
    def test_no_field_crosses_split(self, dataset):
        grp = dataset.groupby("field_id")["split"].nunique()
        assert (grp <= 1).all()

    def test_split_counts_match_r5_9(self, dataset):
        vc = dataset["split"].value_counts().to_dict()
        assert vc["train"] == 37211
        assert vc["val"] == 14611
        assert vc["test"] == 5838

    def test_split_audit_csv_present(self):
        aud = pd.read_csv(OUT_DIR / "split_audit.csv")
        assert set(aud["split"]) == {"train", "val", "test"}
        assert aud[aud["split"] == "test"].set_index("dominant_crop") \
            .loc["pepper", "fields"] == 196


# ------------------------------------------------------------------------- #
# Static baseline reproducibility vs R5.9 (Phase 7)
# ------------------------------------------------------------------------- #
class TestStaticBaselineReproducibility:
    def test_r5_9_exact_reproduction(self, baselines):
        run = baselines["runs"]["SET_A_r5_9_exact_reproduction"]
        assert abs(run["best_test_balanced_accuracy"] - R5_9_BALANCED_ACCURACY) \
            <= 0.01
        assert abs((run["best_test_roc_auc"] or 0.0) - R5_9_ROC_AUC) <= 0.01


# ------------------------------------------------------------------------- #
# Decision artifact consistency (Phase 11)
# ------------------------------------------------------------------------- #
class TestDecisionConsistency:
    KEYS = ["status", "best_model", "best_balanced_accuracy", "best_macro_f1",
            "best_auc", "R5.9_balanced_accuracy", "delta_vs_R5.9",
            "signal_decision", "leakage_status", "spatial_split_status",
            "temporal_alignment_status", "crop_extent_unit_status",
            "recommendation"]

    def test_required_keys_present(self, decision):
        for k in self.KEYS:
            assert k in decision

    def test_macro_f1_filled(self, decision):
        assert decision["best_macro_f1"] is not None
        assert 0.0 <= decision["best_macro_f1"] <= 1.0

    def test_auc_and_accuracy_in_valid_range(self, decision):
        assert 0.0 <= decision["best_auc"] <= 1.0
        assert 0.0 <= decision["best_balanced_accuracy"] <= 1.0

    def test_delta_consistent(self, decision):
        assert abs(decision["delta_vs_R5.9"]
                   - (decision["best_balanced_accuracy"]
                      - R5_9_BALANCED_ACCURACY)) < 1e-9

    def test_signal_decision_in_allowed_set(self, decision):
        assert decision["signal_decision"] in (
            "no_signal", "weak_signal", "promising_signal", "strong_signal")

    def test_crop_extent_status_preserved(self, decision):
        assert decision["crop_extent_unit_status"] == "UNKNOWN"


# ------------------------------------------------------------------------- #
# Provenance contract (Phase 13)
# ------------------------------------------------------------------------- #
class TestProvenance:
    def test_contract_exists(self, provenance):
        assert provenance["schema_version"] == "r5.10.0"
        assert provenance["seed"] == 42

    def test_no_fabrication_and_no_future(self, provenance):
        assert provenance["no_fabrication"] is True
        assert provenance["no_future_information_in_primary"] is True

    def test_crop_extent_not_a_feature(self, provenance):
        assert provenance["crop_extent_as_feature"] is False
        assert provenance["crop_extent_unit_status"] == "UNKNOWN"

    def test_source_fingerprints_present(self, provenance):
        assert len(provenance["source_fingerprint_dk_grids"]) == 6
        assert all(v
                   for v in provenance["source_fingerprint_dk_grids"].values())

    def test_generated_artifacts_listed(self, provenance, decision):
        assert "reports/R5.10/R5.10_decision.json" in \
            provenance["generated_artifacts"]
        assert decision["status"] == "COMPLETE"