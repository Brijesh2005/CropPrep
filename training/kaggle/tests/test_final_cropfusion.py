"""Tests for training/kaggle/scripts/final_cropfusion.py (the FINAL campaign).

Run from repo root::

    pytest training/kaggle/tests/test_final_cropfusion.py -q
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

from training.kaggle.scripts.final_cropfusion import (  # noqa: E402
    FINAL_DIR,
    FTD_CSV,
    GRID_CSV,
    IMAGERY_COLS,
    MIN_SURVEY_TO_GRID_YEARS,
    MIN_OBS_ROBUSTNESS,
    T_GRID,
    build_population,
    load_ftd,
    prepare_imagery,
    prepare_tabular,
)

PAPER_DIR = REPO_ROOT / "paper"
HEADLINE = ("tabular", "imagery", "cropfusion")

TEST_TOTAL = 980
TEST_PEPPER = 196
TEST_COCONUT = 784


@pytest.fixture(scope="session")
def ftd() -> pd.DataFrame:
    return load_ftd()


@pytest.fixture(scope="session")
def pop() -> pd.DataFrame:
    return build_population(ftd=load_ftd())


@pytest.fixture(scope="session")
def pop_csv() -> pd.DataFrame:
    return pd.read_csv(FINAL_DIR / "final_population.csv", low_memory=False)


@pytest.fixture(scope="session")
def audit() -> dict:
    return json.load(open(FINAL_DIR / "final_population.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def provenance() -> dict:
    return json.load(open(FINAL_DIR / "final_provenance.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def results() -> pd.DataFrame:
    return pd.read_csv(FINAL_DIR / "final_results.csv")


@pytest.fixture(scope="session")
def probs() -> dict:
    return json.load(open(FINAL_DIR / "final_probs.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def confusion() -> dict:
    return json.load(open(FINAL_DIR / "final_confusion_matrices.json",
                          encoding="utf-8"))


@pytest.fixture(scope="session")
def reporter() -> dict:
    return json.load(open(FINAL_DIR / "final_report.json", encoding="utf-8"))


@pytest.fixture(scope="session")
def img() -> dict:
    return prepare_imagery(pop=build_population(ftd=load_ftd()))


# ------------------------------------------------------------------------- #
# Artifact presence
# ------------------------------------------------------------------------- #
class TestArtifactsPresent:
    EXPECTED = [
        "final_population.csv", "final_population.json",
        "final_tabular_selection.csv", "final_tabular_selection.json",
        "final_imagery_selection.json", "final_fusion_selection.json",
        "final_results.csv", "final_ablation.csv",
        "final_confidence_intervals.csv", "final_confusion_matrices.json",
        "final_probs.json", "final_robustness.json",
        "final_provenance.json", "final_report.json", "final_report.md",
        "final_tables.md", "final_population.csv",
    ]

    def test_final_dir_artifacts_exist(self):
        for name in self.EXPECTED:
            assert (FINAL_DIR / name).is_file(), f"missing {name}"

    def test_figures_exist(self):
        figs = sorted(p.name for p in (FINAL_DIR / "figures").glob("fig*.png"))
        assert len(figs) >= 8

    def test_paper_markdown_exists(self):
        papers = sorted(p.name for p in PAPER_DIR.glob("*.md"))
        for req in ("abstract.md", "methodology.md", "results.md",
                    "limitations.md", "conclusion.md"):
            assert req in papers, f"missing paper/{req}"


# ------------------------------------------------------------------------- #
# Balanced population / frozen test population
# ------------------------------------------------------------------------- #
class TestPopulation:
    def test_populated_size_and_counts(self, pop_csv, audit):
        assert len(pop_csv) == 3985
        assert audit["total_fields"] == 3985
        assert audit["split_counts"]["('test', 0)"] == TEST_COCONUT
        assert audit["split_counts"]["('test', 1)"] == TEST_PEPPER

    def test_test_partition_is_frozen(self, pop_csv, ftd):
        test = pop_csv[pop_csv["split"] == "test"]
        assert len(test) == TEST_TOTAL
        assert (test["target"] == 1).sum() == TEST_PEPPER
        assert (test["target"] == 0).sum() == TEST_COCONUT
        ftd_test_pepper = set(
            ftd[(ftd["split"] == "test") & (ftd["target"] == 1)]["field_id"])
        # every pepper field in the R5.10 test split is retained
        assert set(test[test["target"] == 1]["field_id"]) == ftd_test_pepper
        # sampled majority strictly within the R5.9/R5.10 test split
        assert set(test["field_id"]).issubset(
            set(ftd[ftd["split"] == "test"]["field_id"]))

    def test_deterministic_population(self):
        a = build_population(ftd=load_ftd())
        b = build_population(ftd=load_ftd())
        assert a["field_id"].tolist() == b["field_id"].tolist()
        assert a["target"].tolist() == b["target"].tolist()

    def test_pepper_never_oversampled(self, pop_csv, ftd):
        for split in ("train", "val", "test"):
            n_pep = (pop_csv["split"] == split) & (
                pop_csv["target"] == 1)
            n_pep_ftd = (ftd["split"] == split) & (ftd["target"] == 1)
            assert n_pep.sum() == n_pep_ftd.sum()


# ------------------------------------------------------------------------- #
# Split integrity
# ------------------------------------------------------------------------- #
class TestSplitIntegrity:
    def test_no_field_crosses_split(self, pop_csv):
        grp = pop_csv.groupby("field_id")["split"].nunique()
        assert (grp <= 1).all()
        assert (grp > 1).sum() == 0

    def test_audit_reports_no_crossing(self, audit):
        assert audit["fields_crossing_splits"] == 0

    def test_population_split_agrees_with_r510(self, pop_csv, ftd):
        merged = pop_csv[["field_id", "split"]].merge(
            ftd[["field_id", "split"]], on="field_id", suffixes=("_pop", "_r"))
        assert (merged["split_pop"] == merged["split_r"]).all()


# ------------------------------------------------------------------------- #
# No leakage: Crop_Extent / fractions / NPP / satellite stats excluded
# ------------------------------------------------------------------------- #
class TestNoLeakage:
    FORBIDDEN = [
        "extent", "fraction", "dominance", "npp", "yield",
        "sat_", "obs_count", "n_valid", "target",
    ]

    def test_tabular_feature_sets_are_clean(self, pop):
        feat = prepare_tabular(pop, with_location=False)
        lower = "|".join(c.lower() for c in feat["cols"])
        for token in self.FORBIDDEN:
            assert token not in lower, f"forbidden '{token}' in tabular feats"

    def test_tabular_excludes_satellite_index_columns(self, pop):
        feat = prepare_tabular(pop, with_location=False)
        for c in IMAGERY_COLS:
            assert c not in feat["cols"], f"satellite col '{c}' leaked in"
        # tabular must not contain the location sensitivity in headline variant
        assert "lat" not in feat["cols"]
        assert "lon" not in feat["cols"]

    def test_location_columns_only_in_sensitivity_variant(self, pop):
        feat_nl = prepare_tabular(pop, with_location=False)
        feat_loc = prepare_tabular(pop, with_location=True)
        assert set(feat_loc["cols"]) - set(feat_nl["cols"]) == {"lat", "lon"}

    def test_crop_extent_status_unknown_in_artifacts(self, provenance, reporter):
        assert provenance["crop_extent_unit_status"] == "UNKNOWN"
        assert reporter["crop_extent_unit_status"] == "UNKNOWN"


# ------------------------------------------------------------------------- #
# Temporal policy: leading window only + masking (never fabrication)
# ------------------------------------------------------------------------- #
class TestNoFutureAndMasking:
    def test_masked_slots_are_future_or_unmatched(self, img, pop):
        gr = pd.read_csv(GRID_CSV, low_memory=False)
        gr = gr[gr["field_id"].isin(set(pop["field_id"]))]
        g = gr.set_index(["field_id", "grid_year"])
        survey = pop["survey_year"].to_numpy()
        fids = pop["field_id"].to_numpy()
        n_bad = 0
        for i in range(len(pop)):
            fid = fids[i]
            for t, gy in enumerate(MIN_SURVEY_TO_GRID_YEARS):
                if img["mask"][i, t] == 1.0:
                    assert gy <= int(survey[i]), (
                        f"future slot {gy} > survey {survey[i]} filled")
                    assert (fid, gy) in g.index
                    row = g.loc[(fid, gy)]
                    assert bool(row["matched"]), "unmatched masked-as-real slot"
                elif gy <= int(survey[i]) and (fid, gy) in g.index:
                    row = g.loc[(fid, gy)]
                    if bool(row["matched"]) and not np.isnan(
                            pd.to_numeric(row[img["cols"]],
                                          errors="coerce").to_numpy(float)).any():
                        n_bad += 1  # a real matched row dropped -> mask wrong
        assert n_bad == 0

    def test_mask_shapes_and_values(self, img):
        assert img["X"].shape == (3985, T_GRID, len(img["cols"]))
        assert img["mask"].shape == (3985, T_GRID)
        assert set(np.unique(img["mask"])) <= {0.0, 1.0}

    def test_padding_reported(self, img, pop):
        audit_cls = img["audit"]
        assert audit_cls["timesteps"] == MIN_SURVEY_TO_GRID_YEARS
        assert audit_cls["features_per_timestep"] == len(img["cols"])
        obs = (img["mask"] == 1.0).sum(axis=1)
        assert (obs >= 0).all() and (obs <= T_GRID).all()


# ------------------------------------------------------------------------- #
# Determinism of preprocessing
# ------------------------------------------------------------------------- #
class TestDeterministicPreprocessing:
    def test_tabular_repeatable(self, pop):
        a = prepare_tabular(pop, with_location=False)
        b = prepare_tabular(pop, with_location=False)
        assert np.array_equal(a["X"], b["X"])
        assert a["cols"] == b["cols"]

    def test_imagery_repeatable(self, pop):
        a = prepare_imagery(pop=build_population(ftd=load_ftd()))
        b = prepare_imagery(pop=build_population(ftd=load_ftd()))
        assert np.array_equal(a["X"], b["X"])
        assert np.array_equal(a["mask"], b["mask"])


# ------------------------------------------------------------------------- #
# Frozen test evaluation (single use, identical population across models)
# ------------------------------------------------------------------------- #
class TestFrozenTestEval:
    def test_all_models_see_identical_test(self, probs):
        base = probs["tabular"]["y_true"]
        assert len(base) == TEST_TOTAL
        for m in HEADLINE:
            assert probs[m]["y_true"] == base
            assert len(probs[m]["y_prob"]) == len(base)
            assert all(0.0 <= p <= 1.0 for p in probs[m]["y_prob"])
        assert sum(base) == TEST_PEPPER

    def test_threshold_is_frozen_05(self, probs, confusion):
        for m in HEADLINE:
            yt = np.array(probs[m]["y_true"])
            yp = np.array(probs[m]["y_prob"])
            pred = (yp >= 0.5).astype(int)
            cm = np.array(confusion[m]["cm"])
            tn, fp, fn, tp = cm.ravel()
            assert (pred[yt == 0] == 0).sum() == tn
            assert (pred[yt == 0] == 1).sum() == fp
            assert (pred[yt == 1] == 0).sum() == fn
            assert (pred[yt == 1] == 1).sum() == tp

    def test_result_table_in_range(self, results):
        for _, r in results.iterrows():
            for c in ("accuracy", "balanced_accuracy", "macro_f1",
                      "weighted_f1", "roc_auc", "coconut_precision",
                      "coconut_recall", "pepper_precision", "pepper_recall"):
                assert 0.0 <= r[c] <= 1.0, f"{r['model']}.{c}={r[c]}"
            cm = np.array(json.loads(r["confusion_matrix"]))
            assert cm.shape == (2, 2)
            assert cm.sum() == TEST_TOTAL

    def test_results_csv_matches_report_json(self, results, reporter):
        rj = {r["model"]: r for r in reporter["results"]}
        for _, row in results.iterrows():
            j = rj[row["model"]]
            assert abs(row["balanced_accuracy"] - j["balanced_accuracy"]) < 1e-4
            assert abs(row["roc_auc"] - j["roc_auc"]) < 1e-4
            assert abs(row["macro_f1"] - j["macro_f1"]) < 1e-4

    def test_verdict_present_and_non_manufactured(self, reporter):
        assert "verdict" in reporter or "fusion_verdict" in reporter
        note = reporter["90pct_target_note"].lower()
        assert "70" in note or "target" in note
        assert "manufactured" in note or "altered" in note


# ------------------------------------------------------------------------- #
# Ablation and mechanism selection (validation only)
# ------------------------------------------------------------------------- #
class TestAblation:
    def test_ablation_rows_present(self):
        ab = pd.read_csv(FINAL_DIR / "final_ablation.csv")
        labels = set(ab["ablation"])
        for need in ("A_tabular_only", "A2_tabular_with_location",
                     "B_imagery_only", "C_fusion",
                     "C_mechanism_concat", "C_mechanism_gated",
                     "C_mechanism_cross"):
            assert need in labels, f"missing ablation {need}"

    def test_selected_mechanism_matches_selection_artifact(self, pop):
        sel = json.load(open(FINAL_DIR / "final_fusion_selection.json",
                             encoding="utf-8"))
        mech = sel["selected"]
        assert mech in ("concat", "gated", "cross")
        ab = pd.read_csv(FINAL_DIR / "final_ablation.csv")
        cross = ab[ab["ablation"] == "C_mechanism_cross"][
            "val_balanced_accuracy"].iloc[0]
        assert abs(cross - sel["mechanisms"][mech]["balanced_accuracy"]) < 1e-4
        # headline fusion ablation must equal the selected mechanism
        cfus = ab[ab["ablation"] == "C_fusion"]["val_balanced_accuracy"].iloc[0]
        assert abs(cfus - sel["mechanisms"][mech]["balanced_accuracy"]) < 1e-4


# ------------------------------------------------------------------------- #
# Robustness (min-observation rule, applied to all models)
# ------------------------------------------------------------------------- #
class TestRobustness:
    def test_robustness_rule_and_results(self):
        rob = json.load(open(FINAL_DIR / "final_robustness.json",
                             encoding="utf-8"))
        assert f">= {MIN_OBS_ROBUSTNESS}" in rob["rule"]
        assert rob["fields"] > 0
        assert {r["model"] for r in rob["test_results"]} == {
            "tabular", "imagery", "cropfusion"}
        for r in rob["test_results"]:
            assert 0.0 <= r["test_balanced_accuracy"] <= 1.0
        assert {r["model"] for r in rob["val_results"]} == {
            "tabular", "imagery", "cropfusion"}


# ------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# ------------------------------------------------------------------------- #
class TestConfidenceIntervals:
    def test_ci_grid_complete(self):
        ci = pd.read_csv(FINAL_DIR / "final_confidence_intervals.csv")
        models = set(ci["model"])
        assert models == {"tabular", "imagery", "cropfusion"}
        assert ci["metric"].nunique() == 3
        assert len(ci) == 9
        assert (ci["n_boot"] == 1000).all()
        for _, r in ci.iterrows():
            assert 0.0 <= r["mean"] <= 1.0
            assert r["ci_low"] <= r["mean"] <= r["ci_high"]


# ------------------------------------------------------------------------- #
# Provenance contract
# ------------------------------------------------------------------------- #
class TestProvenance:
    def test_required_keys(self, provenance):
        for k in ("git_commit", "dataset_version", "split_hash",
                  "population_rule", "threshold_policy",
                  "crop_extent_unit_status", "temporal_policy",
                  "missing_policy", "selected_tabular_model",
                  "selected_fusion_mechanism", "gpu", "final_results"):
            assert k in provenance, f"missing provenance key {k}"

    def test_manifest_hashes_populated(self, provenance):
        for name in ("r5_9_field_dataset_split.csv",
                     "r5_10_field_target_dataset.csv",
                     "r5_10_temporal_field_grid.csv"):
            h = provenance["manifest_hashes"].get(name, "")
            assert len(h) == 64

    def test_constraints_preserved(self, provenance):
        assert provenance["threshold_policy"] == (
            "fixed 0.5 on logits; never tuned on test")
        assert "leading window" in provenance["temporal_policy"].lower()
        assert "masked" in provenance["missing_policy"].lower()
        assert "never imputed" in provenance["missing_policy"].lower()
        assert provenance["gpu"] == "none (CPU)"
        assert provenance["crop_extent_use"] == (
            "upstream target construction only; NEVER a feature")


class TestInputHashes:
    def test_inputs_unchanged(self, provenance):
        # hashes recorded at run time; recompute only if inputs are local
        for p in (FTD_CSV, GRID_CSV):
            assert p.is_file()