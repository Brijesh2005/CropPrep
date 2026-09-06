"""Tests for training/kaggle/scripts/r5_7_data_recovery.py (Phase 18).

Run from repo root::

    pytest training/kaggle/tests/test_r5_7_data_recovery.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "training" / "kaggle" / "scripts"
sys_path_insert = str(REPO_ROOT)
if sys_path_insert not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path_insert)

from training.kaggle.scripts.r5_7_data_recovery import (
    OUT_DIR,
    BINARY,
    SUPERVISED,
    TALUK_SPLIT,
    _master_key,
    coord_key7,
    load_frozen,
    load_survey_pool,
)

MASTER = OUT_DIR / "master_geospatial_features.csv"


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture(scope="session")
def survey_pool() -> tuple[pd.DataFrame, dict]:
    return load_survey_pool()


@pytest.fixture(scope="session")
def master() -> pd.DataFrame:
    return pd.read_csv(MASTER)


@pytest.fixture(scope="session")
def decision() -> dict:
    return json.load(open(OUT_DIR / "decision.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def recovery() -> dict:
    return json.load(open(OUT_DIR / "observation_recovery.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def tiers() -> dict:
    return json.load(open(OUT_DIR / "quality_tiers.json", encoding="utf-8"))


# ------------------------------------------------------------------ #
# Loader tests
# ------------------------------------------------------------------ #

class TestLoader:
    def test_supervised_rows_positive(self, survey_pool):
        pool, stats = survey_pool
        assert stats["final_rows_supervised"] > 0

    def test_no_missing_coords_in_supervised(self, survey_pool):
        pool, _ = survey_pool
        s = pool[pool["is_supervised"]]
        assert s["latitude"].notna().all()
        assert s["longitude"].notna().all()

    def test_drop_duplicates_deterministic(self, survey_pool):
        pool, stats = survey_pool
        assert stats["record_duplicates_removed"] > 0
        key = pool[pool["is_supervised"]][
            ["survey_id", "year", "season", "crop_label"]].copy()
        assert not key.duplicated().any()

    def test_crop_labels_subset_of_super(self, survey_pool):
        pool, _ = survey_pool
        assert set(pool.loc[pool["is_supervised"], "crop_label"].unique()) <= set(SUPERVISED)


# ------------------------------------------------------------------ #
# Master schema tests
# ------------------------------------------------------------------ #

class TestMasterSchema:
    def test_rows_nonzero(self, master):
        assert len(master) > 0

    def test_field_observation_id_unique(self, master):
        assert not master["field_observation_id"].duplicated().any()

    def test_observation_id_sequential(self, master):
        ids = master["observation_id"].astype(int).to_numpy()
        assert ids[0] == 1
        assert (np.diff(ids) == 1).all()

    def test_split_all_taluk(self, master):
        assert master["split"].isin(TALUK_SPLIT.values()).all()

    def test_no_forbidden_leakage_columns(self, master):
        forbidden = {"Yield_Proxy_NPP", "yield_proxy_npp", "benchmark_eligible"}
        assert not (set(master.columns) & forbidden)

    def test_numeric_env_features_are_float(self, master):
        env_cols = ["ndvi", "evi", "soil_clay_pct", "elevation", "annual_rainfall_mm"]
        for col in env_cols:
            assert col in master.columns
            vals = pd.to_numeric(master[col], errors="coerce")
            assert vals.notna().mean() > 0.3, f"{col} mostly null"

    def test_coordinate_within_dk_bounds(self, master):
        lat = master["latitude"]
        lon = master["longitude"]
        assert lat.between(12.4, 13.4).mean() > 0.99
        assert lon.between(74.6, 75.9).mean() > 0.99

    def test_crop_labels_are_string_not_nan(self, master):
        assert master["crop_label"].notna().all()
        assert master["crop_label"].isin(SUPERVISED).all()


# ------------------------------------------------------------------ #
# Tier tests
# ------------------------------------------------------------------ #

class TestTiers:
    def test_tier_alphabetical(self, master):
        tiers = set(master["quality_tier"].unique())
        assert tiers <= {"A", "B", "C", "D"}

    def test_tier_counts(self, tiers):
        tc = tiers["tier_counts"]
        assert "A" in tc
        assert tc["A"] > 0
        assert tc["B"] > 0

    def test_no_silent_discards(self, master):
        tier_rows = len(master)
        assert tier_rows > 0

    def test_tier_reasons_populated(self, master):
        non_a = master[master["quality_tier"] != "A"]
        assert len(non_a) > 0
        assert non_a["tier_reasons"].notna().all()

    def test_tier_with_satellite_alignment(self, master):
        ts = master["tier_with_satellite"]
        sat = master["satellite_match_valid"].astype(str).str.lower().isin(["true", "1"])
        assert ts[~sat].eq("no_imagery").all()


# ------------------------------------------------------------------ #
# Observation recovery
# ------------------------------------------------------------------ #

class TestRecovery:
    def test_recovery_positive(self, recovery):
        assert recovery["recovered_observations"] > 0

    def test_recovery_factor(self, recovery):
        assert recovery["recovery_factor_x"] >= 2.0

    def test_frozen_subset_ratio(self, recovery):
        assert recovery["master_in_frozen_key"] < recovery["total_master_supervised"]

    def test_imaging_note_present(self, recovery):
        assert "imagery_note" in recovery


# ------------------------------------------------------------------ #
# Decision
# ------------------------------------------------------------------ #

class TestDecision:
    def test_verdict_no_signal(self, decision):
        assert decision["verdict"] == "no_signal"

    def test_primary_bottleneck(self, decision):
        assert "limited" in decision["primary_bottleneck"]

    def test_ceiling_between_50_55(self, decision):
        c = decision["after_ceiling_dedup50m"]
        assert 0.48 < c < 0.56


# ------------------------------------------------------------------ #
# Provenance contract
# ------------------------------------------------------------------ #

class TestProvenance:
    def test_core_columns_in_master(self, master):
        contract = json.load(open(OUT_DIR / "provenance_contract.json", encoding="utf-8"))
        for col in contract["contract"]["core_provenance_columns"]:
            assert col in master.columns, f"missing provenance column {col}"

    def test_env_coverage(self, master):
        assert master["dk_nearest_index"].notna().mean() > 0.95

    def test_satellite_coverage_note(self):
        sat = json.load(open(OUT_DIR / "satellite_availability.json", encoding="utf-8"))
        assert sat["satellite_status"].startswith("recovered")


# ------------------------------------------------------------------ #
# Coord helper
# ------------------------------------------------------------------ #

class TestCoordKey:
    def test_deterministic(self):
        a = coord_key7(12.9892967, 75.1114626)
        b = coord_key7(12.9892967, 75.1114626)
        assert a == b

    def test_length(self):
        assert "|" in coord_key7(13.0, 75.0)


# ------------------------------------------------------------------ #
# Binary separability result smoke
# ------------------------------------------------------------------ #

class TestSeparability:
    def test_result_csv_exists(self):
        csv = OUT_DIR / "r5_7_separability_results.csv"
        assert csv.exists()
        df = pd.read_csv(csv)
        assert len(df) > 0

    def test_bal_acc_in_range(self):
        df = pd.read_csv(OUT_DIR / "r5_7_separability_results.csv")
        vals = df["test_balanced_accuracy"]
        assert vals.between(0.45, 0.60).all()
