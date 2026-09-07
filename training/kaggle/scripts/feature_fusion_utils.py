"""Shared utilities for the feature-level fusion campaign.

Owns the leakage audit, the frozen-data pipeline loading (reusing the exact
final-cropfusion helpers), threshold selection (validation-only), metric
collection, and the image-embedding / tabular cache persistence.

All data is loaded through the frozen R5.10/R5.9 inputs and the deterministic
balanced population. Nothing here touches the test split for scaling,
imputation, weights, early stopping or threshold choice.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.kaggle.scripts.final_cropfusion import (  # noqa: E402
    build_population, load_ftd, prepare_imagery, prepare_tabular, split_y,
)
from training.kaggle.scripts.r5_10_temporal_field_target import (  # noqa: E402
    _metrics_full,
)

REPORT_DIR = REPO_ROOT / "reports" / "feature_fusion"
MODELS_DIR = REPORT_DIR / "models"
FIG_DIR = REPORT_DIR / "figures"
ART_DIR = REPO_ROOT / "artifacts" / "feature_fusion"
CACHE_DIR = ART_DIR / "image_embeddings"
IMG_CACHE_FILE = CACHE_DIR / "image_embeddings.npz"
IMG_CACHE_META = CACHE_DIR / "image_embeddings_meta.json"
TAB_CACHE_DIR = ART_DIR / "tabular"
TAB_CACHE_FILE = TAB_CACHE_DIR / "tabular_features.npz"
TAB_CACHE_META = TAB_CACHE_DIR / "tabular_features_meta.json"

# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------
# Exact/token-level bans. A feature column is rejected if it is (a) in the
# exact ban set, (b) contains a target/quality token (target, label), or
# (c) contains a task-quality / outcome / yield token (fraction, dominance,
# extent, benchmark, npp, yield, provenance-of-outcome).
BANNED_EXACT = {
    "crop_label", "crop_class_id", "source_crop_name", "dominant_crop",
    "coconut_fraction", "pepper_fraction", "top1_fraction",
    "dominance_margin", "dominance_gap", "benchmark_eligible",
    "yield_proxy_npp", "yield_proxy", "system:index", "crop_extent",
    "benchmark", "matched", "reuse_in_reason", "rejection_reason",
    "target_year", "survey_id",
}
BANNED_TOKENS = {
    "target", "label", "fraction", "dominance", "extent", "benchmark",
    "npp", "yield", "proxy",
}
ALLOWED_TOKENS = {"cropland"}   # is_cropland is a valid land-use indicator


def _normalize(col: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(col).lower()).strip("_")


def audit_feature_leakage(feat_cols: list[str],
                          img_cols: list[str] | None = None) -> dict:
    """Return a dict of issues for any forbidden feature column."""
    img_cols = img_cols or []
    issues = []
    for col in list(feat_cols) + list(img_cols):
        norm = _normalize(col)
        if norm in BANNED_EXACT:
            issues.append(f"exact-ban column '{col}'")
            continue
        words = set(norm.split("_"))
        banned = words & (BANNED_TOKENS - ALLOWED_TOKENS)
        if banned:
            issues.append(f"forbidden-token {sorted(banned)} in column '{col}'")
    return {"feature_columns_checked": len(feat_cols) + len(img_cols),
            "issues": issues}


def audit_split_overlap(pop) -> dict:
    issues = []
    by_split = {}
    for s in ("train", "val", "test"):
        by_split[s] = set(pop.loc[pop["split"] == s, "field_id"])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = by_split[a] & by_split[b]
        if inter:
            issues.append(f"field overlap {a}/{b}: {len(inter)} fields")
    return {"split_sizes": {s: len(v) for s, v in by_split.items()},
            "issues": issues}


def audit_all(pop, feat, img=None) -> dict:
    feat_issues = audit_feature_leakage(feat["cols"])["issues"]
    img_issues = audit_feature_leakage(img["cols"] if img is not None else [])["issues"]
    split_issues = audit_split_overlap(pop)["issues"]
    return {
        "feature_issues": feat_issues,
        "imagery_issues": img_issues,
        "split_issues": split_issues,
        "pass": not (feat_issues or img_issues or split_issues),
    }


# ---------------------------------------------------------------------------
# Pipeline loading (frozen, deterministic)
# ---------------------------------------------------------------------------
def load_pipeline(with_location: bool = False) -> tuple:
    ftd = load_ftd()
    pop = build_population(ftd)
    feat = prepare_tabular(pop, with_location=with_location)
    img = prepare_imagery(pop)
    idx, y = split_y(pop)
    return pop, feat, img, idx, y


def split_hash(pop) -> str:
    rows = "\n".join(f"{f}|{s}" for f, s in
                     zip(pop["field_id"].to_numpy(), pop["split"].to_numpy()))
    return hashlib.sha256(rows.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache persistence (image embeddings / tabular), all reproducibly derived
# from frozen R5.10 inputs with train-only normalization.
# ---------------------------------------------------------------------------
def save_image_cache(pop, img, out: str | Path | None = IMG_CACHE_FILE,
                     meta_out: str | Path | None = IMG_CACHE_META) -> None:
    out, meta_out = Path(out), Path(meta_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=img["X"], mask=img["mask"],
                        field_id=pop["field_id"].to_numpy(),
                        split=pop["split"].to_numpy(),
                        survey_year=pop["survey_year"].to_numpy())
    meta = {
        "cols": img["cols"],
        "timesteps": img["audit"]["timesteps"],
        "normalization": "train-only mean/std (valid cells)",
        "field_count": int(len(pop)),
        "split_hash": split_hash(pop),
        "source": str(Path("reports") / "R5.10" / "temporal_field_grid.csv"),
    }
    meta_out.write_text(json.dumps(meta, indent=2))


def load_image_cache(out: str | Path | None = IMG_CACHE_FILE) -> dict:
    out = Path(out)
    if not out.exists():
        raise FileNotFoundError(
            f"image-embedding cache missing at {out}; run "
            "extract_image_embeddings.py first")
    z = np.load(out, allow_pickle=True)
    return {"X": z["X"], "mask": z["mask"], "field_id": z["field_id"],
            "split": z["split"], "survey_year": z["survey_year"]}


def verify_image_cache(pop, img, out: str | Path | None = IMG_CACHE_FILE) -> dict:
    """Compare freshly-computed imagery arrays against the cached ones.

    Returns a report; raises if the cache disagrees with the frozen pipeline
    (signals pipeline drift / silent re-derivation).
    """
    cached = load_image_cache(out)
    same_x = np.array_equal(img["X"], cached["X"])
    same_m = np.array_equal(img["mask"], cached["mask"])
    same_f = list(pop["field_id"].to_numpy()) == list(cached["field_id"])
    report = {"cache": str(Path(out)), "consistent": bool(same_x and same_m and same_f),
              "X_equal": bool(same_x), "mask_equal": bool(same_m),
              "field_id_equal": bool(same_f)}
    if not report["consistent"]:
        raise RuntimeError(
            f"image-embedding cache mismatch: {report}; re-run "
            "extract_image_embeddings.py")
    return report


# ---------------------------------------------------------------------------
# Threshold selection (validation-only) + metrics
# ---------------------------------------------------------------------------
def select_threshold(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
    """Best threshold on VALIDATION maximizing balanced accuracy (frozen after)."""
    from sklearn.metrics import balanced_accuracy_score
    best_t, best_ba = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 199):
        ba = balanced_accuracy_score(y_val, (p_val >= t).astype(int))
        if ba > best_ba:
            best_t, best_ba = t, ba
    return float(best_t), float(best_ba)


def metrics_row(tag: str, mode: str, fusion_type: str, t: float,
                m_val: dict, m_test: dict, m_test_fixed: dict) -> dict:
    return {
        "model": tag, "mode": mode, "fusion_type": fusion_type,
        "threshold_val": round(t, 4),
        "val_balanced_accuracy": m_val.get("balanced_accuracy"),
        "val_roc_auc": m_val.get("roc_auc"),
        "test_accuracy": m_test.get("accuracy"),
        "test_balanced_accuracy": m_test.get("balanced_accuracy"),
        "test_macro_f1": m_test.get("macro_f1"),
        "test_roc_auc": m_test.get("roc_auc"),
        "test_recall_coconut": m_test.get("recall_coconut"),
        "test_recall_pepper": m_test.get("recall_pepper"),
        "test_confusion_matrix": m_test.get("confusion_matrix"),
        "fixed_threshold_balanced_accuracy": m_test_fixed.get("balanced_accuracy"),
        "fixed_threshold_roc_auc": m_test_fixed.get("roc_auc"),
    }


def git_head_revision() -> str:
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def write_py_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))