"""R5.6 — COCONUT vs PEPPER separability / data-information-ceiling investigation (local driver).

Phase 0   R5.6 workspace metadata
Phase 1   binary (coconut/pepper) dataset + manifest, official split untouched
Phase 2   tabular-only baselines   (LR / RF / GB / small MLP, identical splits)
Phase 5   feature separability     (per-class stats, Cohen's d ranking)
Phase 7   spatial separability     (GPS scatter, density, env-niche, taluk dist)
Phase 8   feature importance       (RF permutation importance, grouped)
Phase 3   image-only baselines     (requires image_stats.csv from Kaggle export)
Phase 4   fusion baselines         (tabular + image statistics)
Phase 6   image separability       (distribution + overlap plots)
Phase 9   modality contribution    (cross-shuffle ablations)
Phase 10  data-ceiling table       (summary across every baseline above)
Phase 11  decision                 (threshold rules -> bottleneck)
Phase 12  final report             (R5.6_SEPARABILITY_REPORT.md/.json)

No CropFusion training, no architecture work, no hyperparameter search, no corpus
modification. This script MEASURES the information content already present in the
frozen matched dataset. Every artifact is written to ``reports/R5.6/``.

Run from repo root, e.g.:

    python training/kaggle/scripts/r5_6_separability.py --phases 0,1,2
    python training/kaggle/scripts/r5_6_separability.py --phases 5,7,8
    python training/kaggle/scripts/r5_6_separability.py --phases 3,4,6,9   # after image_stats.csv is pulled
    python training/kaggle/scripts/r5_6_separability.py --phases 10,11,12
    python training/kaggle/scripts/r5_6_separability.py --phases all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

SUPERVISED = ["coconut", "pepper", "coffee", "cardamom"]
BINARY = ["coconut", "pepper"]
CSV_PATH = REPO_ROOT / "govt_crop_matched_v2" / "crop_supervised_v2.csv"
MANIFEST_PATH = REPO_ROOT / "training_manifests" / "crop_supervised_v2.0_manifest.json"
OUT_DIR = REPO_ROOT / "reports" / "R5.6"
IMAGE_STATS_CSV = OUT_DIR / "image_stats.csv"

NUMERIC = [
    "lat", "lon", "spatial_match_distance_km", "year",
    "annual_rainfall_mm", "dewpoint_c", "elevation", "temperature_c",
    "relative_humidity_pct", "slope", "ndvi", "evi", "ndwi", "ndre", "savi",
    "s2_obs_count", "soil_clay_pct", "soil_sand_pct", "soil_organic_carbon",
    "soil_ph", "soil_moisture", "kharif_ndvi", "kharif_evi", "kharif_ndwi",
    "rabi_ndvi", "rabi_evi", "rabi_ndwi", "env_match_distance_m",
]
CATEGORICAL = ["season", "is_cropland", "land_cover_class", "soil_type_class"]

TALUK_SPLIT = {
    "Belthangady": "train", "Mangalore": "train", "Bantwal": "train",
    "Puttur": "val", "Sullia": "test",
}

FEATURE_GROUPS = {
    "Spatial": ["lat", "lon"],
    "Climate": ["annual_rainfall_mm", "dewpoint_c", "temperature_c",
                "relative_humidity_pct"],
    "Terrain": ["elevation", "slope"],
    "Soil": ["soil_clay_pct", "soil_sand_pct", "soil_organic_carbon",
             "soil_ph", "soil_moisture", "soil_type_class"],
    "Vegetation (sat. composites)": ["ndvi", "evi", "ndwi", "ndre", "savi",
                                     "kharif_ndvi", "kharif_evi", "kharif_ndwi",
                                     "rabi_ndvi", "rabi_evi", "rabi_ndwi",
                                     "s2_obs_count"],
    "Matching/metadata": ["spatial_match_distance_km", "env_match_distance_m",
                          "year", "season", "is_cropland", "land_cover_class"],
}

SEED = 42


# --------------------------------------------------------------------------- #
# Load / split helpers
# --------------------------------------------------------------------------- #

def load_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, dtype={"crop_label": str, "location_taluk": str,
                                      "location_village": str})
    if "benchmark_eligible" in df:
        ok = df["benchmark_eligible"].fillna("true").astype(str).str.strip()
        ok = ok.str.lower().isin(["true", "1", "yes"])
        df = df[ok].copy()
    df["crop_label"] = df["crop_label"].str.strip().str.lower()
    df["split"] = df["location_taluk"].map(TALUK_SPLIT).fillna("unknown")
    df["class_id"] = df["crop_label"].map({c: i for i, c in enumerate(SUPERVISED)})
    return df


def supervised_mask(df: pd.DataFrame) -> np.ndarray:
    return df["crop_label"].isin(SUPERVISED).to_numpy()


def split_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: df[(df["split"] == name) & supervised_mask(df)].copy()
            for name in ("train", "val", "test")}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Feature matrix
# --------------------------------------------------------------------------- #

def build_matrix(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    X_num = df[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    missing_num = int(np.isnan(X_num).sum())
    medians = np.nanmedian(X_num, axis=0)
    for j in range(X_num.shape[1]):
        col = X_num[:, j]
        if np.isnan(col).any():
            col[np.isnan(col)] = medians[j]
    X_cat = df[categorical].astype(object).fillna("__nan__")
    all_codes: list[np.ndarray] = []
    cat_meta: dict[str, Any] = {}
    for col in categorical:
        vals = X_cat[col].astype(str)
        codes, uniques = pd.factorize(vals)
        all_codes.append(codes.astype(np.int64))
        cat_meta[col] = {"unique": int(len(uniques)),
                         "mapping": {int(i): str(v) for i, v in enumerate(uniques)}}
    X = np.hstack([X_num, np.stack(all_codes, axis=1)]).astype(np.float64)
    return X, {"numeric_cols": numeric, "categorical_cols": categorical,
               "missing_numeric_cells": int(missing_num),
               "categorical": cat_meta}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def metrics(y_true: np.ndarray, y_pred: np.ndarray,
            class_names: list[str]) -> dict[str, Any]:
    n = len(y_true)
    n_cls = len(class_names)
    acc = float((y_pred == y_true).mean()) if n else 0.0
    support = {c: int((y_true == i).sum()) for i, c in enumerate(class_names)}
    pred = {c: int((y_pred == i).sum()) for i, c in enumerate(class_names)}
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    per = {}
    for i, c in enumerate(class_names):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        per[c] = {"precision": round(pr, 4), "recall": round(rc, 4),
                  "f1": round(f1, 4), "support": support[c]}
    bal = float(np.nanmean([per[c]["recall"] for c in class_names if support[c] > 0]))
    macro_f1 = float(np.mean([per[c]["f1"] for c in class_names if support[c] > 0]))
    prior = max(support.values()) / n if n else 0.0
    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per,
        "prediction_distribution": pred,
        "target_distribution": support,
        "confusion_matrix": cm.tolist(),
        "majority_prior_accuracy": round(prior, 4),
        "beats_majority": bool(acc > prior + 1e-6),
        "n": int(n),
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default,
                               ensure_ascii=False), encoding="utf-8")
    print(f"  saved -> {path}")


# --------------------------------------------------------------------------- #
# Phase 0 — workspace metadata
# --------------------------------------------------------------------------- #

def phase0() -> dict[str, Any]:
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True,
                            cwd=REPO_ROOT).stdout.strip()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    meta = {
        "phase": "R5.6",
        "title": "COCONUT vs PEPPER separability / data information ceiling",
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_branch": subprocess.run(["git", "branch", "--show-current"],
                                     capture_output=True, text=True,
                                     cwd=REPO_ROOT).stdout.strip(),
        "corpus": {
            "csv": str(CSV_PATH.name),
            "csv_sha256": sha256(CSV_PATH),
        },
        "manifest": {
            "file": str(MANIFEST_PATH.name),
            "sha256": sha256(MANIFEST_PATH),
            "dataset_version": manifest.get("dataset_version"),
            "split_strategy": manifest.get("split_strategy"),
        },
        "split_counts": manifest.get("class_counts"),
        "feature_schema": manifest.get("feature_schema"),
        "image_schema": manifest.get("image_schema"),
        "constraints": [
            "frozen corpus NOT modified",
            "official split (spatial leave-one-taluk-out) NOT changed",
            "binary subset = coconut + pepper only, 1:1 provenance",
            "no CropFusion training, no architecture changes, no HPO",
            "no fabricated observations, matching criteria unchanged",
        ],
    }
    return meta


# --------------------------------------------------------------------------- #
# Phase 1 — binary dataset
# --------------------------------------------------------------------------- #

def phase1() -> dict[str, Any]:
    df = load_frame()
    bi = df[df["crop_label"].isin(BINARY)].copy()
    bi = bi.sort_values(["split", "crop_label", "record_id"]).reset_index(drop=True)
    binary_rows = len(bi)
    counts = {s: bi[bi["split"] == s]["crop_label"].value_counts().to_dict()
              for s in ("train", "val", "test")}
    csv_out = OUT_DIR / "binary_coconut_pepper.csv"
    bi.to_csv(csv_out, index=False)
    manifest = {
        "phase": "R5.6 Phase 1",
        "description": "binary coconut-vs-pepper dataset, official frozen split "
                       "preserved, provenance unchanged (same record_id / "
                       "source_record_id as the frozen corpus)",
        "source_csv": str(CSV_PATH.name),
        "source_csv_sha256": sha256(CSV_PATH),
        "binary_csv": csv_out.name,
        "binary_csv_sha256": sha256(csv_out),
        "classes": BINARY,
        "rows": binary_rows,
        "split_counts": counts,
        "expected": {
            "train": {"coconut": 4292, "pepper": 1599},
            "val": {"coconut": 1373, "pepper": 1063},
            "test": {"coconut": 1200, "pepper": 1033},
        },
        "provenance_note": "filter = crop_label in [coconut, pepper] on the "
                           "benchmark-eligible frozen corpus; split column is the "
                           "official taluk mapping; nothing else changed.",
    }
    return manifest


# --------------------------------------------------------------------------- #
# Phase 2 — tabular-only baselines
# --------------------------------------------------------------------------- #

def _proba_frame(bi: pd.DataFrame, probs: dict[str, np.ndarray]) -> pd.DataFrame:
    out = pd.DataFrame({
        "record_id": bi["record_id"].astype(str),
        "crop_label": bi["crop_label"],
    })
    for name, p in probs.items():
        out[f"{name}_p_pepper"] = p
    return out


def phase2() -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    df = load_frame()
    sets = split_sets(df)
    train, val, test = sets["train"], sets["val"], sets["test"]

    X_all, meta = build_matrix(pd.concat([train, val, test]), NUMERIC, CATEGORICAL)
    n_tr = len(train); n_vl = len(val)
    X_tr, X_va, X_te = X_all[:n_tr], X_all[n_tr:n_tr + n_vl], X_all[n_tr + n_vl:]

    bm_tr = train["crop_label"].isin(BINARY).to_numpy()
    bm_va = val["crop_label"].isin(BINARY).to_numpy()
    bm_te = test["crop_label"].isin(BINARY).to_numpy()

    y_tr = (train["crop_label"] == "pepper").to_numpy()[bm_tr].astype(int)
    y_va = (val["crop_label"] == "pepper").to_numpy()[bm_va].astype(int)
    y_te = (test["crop_label"] == "pepper").to_numpy()[bm_te].astype(int)

    scaler = StandardScaler().fit(X_tr[bm_tr])
    Xb_tr = scaler.transform(X_tr[bm_tr])
    Xb_va = scaler.transform(X_va[bm_va])
    Xb_te = scaler.transform(X_te[bm_te])
    names = BINARY

    configs = [
        ("lr", LogisticRegression(max_iter=3000, random_state=SEED,
                                  class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=250, random_state=SEED,
                                      class_weight="balanced")),
        ("gb", GradientBoostingClassifier(n_estimators=250, random_state=SEED,
                                          max_depth=4)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                              random_state=SEED)),
    ]
    rows: list[dict[str, Any]] = []
    probs: dict[str, np.ndarray] = {}
    for name, clf in configs:
        clf.fit(Xb_tr, y_tr)
        for split_tag, (X, y) in (("val", (Xb_va, y_va)),
                                  ("test", (Xb_te, y_te))):
            pred = clf.predict(X)
            m = metrics(y, pred, names)
            p = clf.predict_proba(X)[:, 1]
            auc = round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else None
            rows.append({
                "modality": "tabular", "model": name, "split": split_tag,
                "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"], "roc_auc": auc,
                "precision_coconut": m["per_class"]["coconut"]["precision"],
                "recall_coconut": m["per_class"]["coconut"]["recall"],
                "precision_pepper": m["per_class"]["pepper"]["precision"],
                "recall_pepper": m["per_class"]["pepper"]["recall"],
                "confusion_matrix": json.dumps(m["confusion_matrix"],
                                               default=_json_default),
                "beats_majority": m["beats_majority"],
                "majority_prior_accuracy": m["majority_prior_accuracy"],
            })
            if split_tag == "test":
                probs[f"{name}_p_pepper"] = p
        print(f"  tabular {name}: test acc={rows[-1]['accuracy']:.4f} "
              f"bal={rows[-1]['balanced_accuracy']:.4f} auc={rows[-1]['roc_auc']}")

    majority = metrics(y_te, np.zeros_like(y_te), names)
    rows.append({
        "modality": "tabular", "model": "majority_constant", "split": "test",
        "accuracy": majority["accuracy"], "balanced_accuracy": majority["balanced_accuracy"],
        "macro_f1": majority["macro_f1"], "roc_auc": None,
        "precision_coconut": majority["per_class"]["coconut"]["precision"],
        "recall_coconut": majority["per_class"]["coconut"]["recall"],
        "precision_pepper": majority["per_class"]["pepper"]["precision"],
        "recall_pepper": majority["per_class"]["pepper"]["recall"],
        "confusion_matrix": json.dumps(majority["confusion_matrix"], default=_json_default),
        "beats_majority": majority["beats_majority"],
        "majority_prior_accuracy": majority["majority_prior_accuracy"],
    })

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "tabular_results.csv", index=False)

    te_ids = test[bm_te][["record_id", "crop_label"]]
    _proba_frame(te_ids, probs).to_csv(OUT_DIR / "tabular_test_probabilities.csv",
                                       index=False)
    info = {
        "phase": "R5.6 Phase 2",
        "modality": "tabular (32 features)",
        "n_features": len(NUMERIC) + len(CATEGORICAL),
        "feature_schema": meta,
        "train_samples": int(len(train[bm_tr])),
        "val_samples": int(len(val[bm_va])),
        "test_samples": int(len(test[bm_te])),
        "class_balance_test": {"coconut": int((y_te == 0).sum()),
                               "pepper": int((y_te == 1).sum())},
        "models": ["lr", "rf", "gb", "mlp", "majority_constant"],
        "note": "identical official split; LR/RF class_weight='balanced'; "
                "MLP default (baseline, not tuned); features scaled via "
                "StandardScaler fit on train only.",
    }
    return info


# --------------------------------------------------------------------------- #
# Phase 5 — feature separability (tabular, train only)
# --------------------------------------------------------------------------- #

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sp = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2
                  + (len(b) - 1) * b.std(ddof=1) ** 2) / (len(a) + len(b) - 2))
    if sp == 0:
        return 0.0
    return float((a.mean() - b.mean()) / sp)


def phase5() -> dict[str, Any]:
    train = split_sets(load_frame())["train"]
    coco = train[train["crop_label"] == "coconut"]
    pepp = train[train["crop_label"] == "pepper"]
    rows: list[dict[str, Any]] = []

    for col in NUMERIC:
        a = pd.to_numeric(coco[col], errors="coerce").to_numpy()
        b = pd.to_numeric(pepp[col], errors="coerce").to_numpy()
        def stats(x: np.ndarray) -> dict[str, float]:
            x = x[np.isfinite(x)]
            if len(x) == 0:
                return {"mean": np.nan, "median": np.nan, "std": np.nan,
                        "min": np.nan, "max": np.nan, "q25": np.nan, "q75": np.nan}
            return {"mean": x.mean(), "median": np.median(x), "std": x.std(ddof=1),
                    "min": x.min(), "max": x.max(),
                    "q25": np.percentile(x, 25), "q75": np.percentile(x, 75)}
        sa, sb = stats(a), stats(b)
        rows.append({
            "feature": col, "group": _group_of(col), "type": "numeric",
            "coconut_mean": round(sa["mean"], 6), "coconut_median": round(sa["median"], 6),
            "coconut_std": round(sa["std"], 6), "coconut_min": round(sa["min"], 6),
            "coconut_max": round(sa["max"], 6), "coconut_iqr": round(sa["q75"] - sa["q25"], 6),
            "pepper_mean": round(sb["mean"], 6), "pepper_median": round(sb["median"], 6),
            "pepper_std": round(sb["std"], 6), "pepper_min": round(sb["min"], 6),
            "pepper_max": round(sb["max"], 6), "pepper_iqr": round(sb["q75"] - sb["q25"], 6),
            "cohens_d": round(cohens_d(a, b), 6), "abs_d": np.nan,
            "missing_coconut": int(np.isnan(a).sum()),
            "missing_pepper": int(np.isnan(b).sum()),
        })
    for col in CATEGORICAL:
        va = coco[col].fillna("__nan__").astype(str).value_counts()
        vb = pepp[col].fillna("__nan__").astype(str).value_counts()
        uniq = sorted(set(va.index) | set(vb.index))
        p1 = np.array([va.get(u, 0) for u in uniq], dtype=float)
        p2 = np.array([vb.get(u, 0) for u in uniq], dtype=float)
        p1 = p1 / p1.sum(); p2 = p2 / p2.sum()
        js = 0.5 * ((p1 - p2) ** 2 / (p1 + p2 + 1e-12)).sum()
        rows.append({
            "feature": col, "group": _group_of(col), "type": "categorical",
            "coconut_mean": np.nan, "coconut_median": np.nan, "coconut_std": np.nan,
            "coconut_min": np.nan, "coconut_max": np.nan, "coconut_iqr": np.nan,
            "pepper_mean": np.nan, "pepper_median": np.nan, "pepper_std": np.nan,
            "pepper_min": np.nan, "pepper_max": np.nan, "pepper_iqr": np.nan,
            "cohens_d": np.nan, "abs_d": round(js, 6),
            "missing_coconut": int(coco[col].isna().sum()),
            "missing_pepper": int(pepp[col].isna().sum()),
        })
    rdf = pd.DataFrame(rows)
    rdf["abs_d"] = rdf[["cohens_d", "abs_d"]].apply(
        lambda r: abs(r["cohens_d"]) if pd.notna(r["cohens_d"]) else r["abs_d"], axis=1)
    rdf["rank"] = rdf["abs_d"].rank(ascending=False, method="min").astype(int)
    rdf = rdf.sort_values("rank").reset_index(drop=True)
    rdf.to_csv(OUT_DIR / "feature_ranking.csv", index=False)

    top15 = rdf.head(15)[["rank", "feature", "group", "type", "abs_d"]]
    return {
        "phase": "R5.6 Phase 5",
        "note": "Cohen's d for numeric; Jensen-Shannon divergence for "
                "categorical. Train split, coconut vs pepper.",
        "n_features": int(len(rdf)),
        "top15": {"n": 15, "features": top15.to_dict(orient="records")},
        "strongest_abs_d": round(float(rdf["abs_d"].iloc[0] - rdf["abs_d"].iloc[0]), 6)
        if rdf["abs_d"].iloc[0] == rdf["abs_d"].iloc[0] else None,
    }


def _group_of(col: str) -> str:
    for g, cols in FEATURE_GROUPS.items():
        if col in cols:
            return g
    return "other"


# --------------------------------------------------------------------------- #
# Phase 7 — spatial separability
# --------------------------------------------------------------------------- #

def phase7() -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = split_sets(load_frame())["train"]
    coco = train[train["crop_label"] == "coconut"]
    pepp = train[train["crop_label"] == "pepper"]

    # scatter map
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(coco["lon"], coco["lat"], s=3, alpha=0.35, label="coconut",
               c="#2e7d32")
    ax.scatter(pepp["lon"], pepp["lat"], s=3, alpha=0.35, label="pepper",
               c="#c62828")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("GPS scatter — coconut vs pepper (train taluka)")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_scatter.png", dpi=110)
    plt.close(fig)

    # density heatmap (2D hist per class)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, sub, title in ((axes[0], coco, "coconut density"),
                           (axes[1], pepp, "pepper density")):
        h = ax.hist2d(sub["lon"], sub["lat"], bins=40, cmap="viridis")
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_density.png", dpi=110)
    plt.close(fig)

    # env niche boxplots
    niche_cols = ["elevation", "slope", "annual_rainfall_mm", "soil_moisture"]
    fig, axes = plt.subplots(1, len(niche_cols), figsize=(15, 4))
    for ax, col in zip(axes, niche_cols):
        a = pd.to_numeric(coco[col], errors="coerce").dropna()
        b = pd.to_numeric(pepp[col], errors="coerce").dropna()
        ax.boxplot([a, b], tick_labels=["coconut", "pepper"])
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_env_niche.png", dpi=110)
    plt.close(fig)

    taluk = train[["location_taluk", "crop_label"]].copy()
    taluk = taluk.loc[taluk["crop_label"].isin(BINARY)].pivot_table(
        index="location_taluk", columns="crop_label", aggfunc="size",
        fill_value=0)
    taluk_pct = taluk.div(taluk.sum(axis=1), axis=0).round(4)
    taluk_pct.to_csv(OUT_DIR / "spatial_taluk_distribution.csv")
    taluk.to_csv(OUT_DIR / "spatial_taluk_counts.csv")

    def niche(col: str) -> dict[str, Any]:
        a = pd.to_numeric(coco[col], errors="coerce").dropna()
        b = pd.to_numeric(pepp[col], errors="coerce").dropna()
        return {
            "coconut": {"mean": round(float(a.mean()), 4),
                        "std": round(float(a.std(ddof=1)), 4),
                        "median": round(float(a.median()), 4),
                        "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)},
            "pepper": {"mean": round(float(b.mean()), 4),
                       "std": round(float(b.std(ddof=1)), 4),
                       "median": round(float(b.median()), 4),
                       "min": round(float(b.min()), 4), "max": round(float(b.max()), 4)},
        }
    return {
        "phase": "R5.6 Phase 7",
        "note": "train split only. Spatial split is leave-one-taluk-out; "
                "train covers Belthangady/Mangalore/Bantwal, val=Puttur, test=Sullia.",
        "centroids_km": {
            "coconut": {"lat": round(float(coco["lat"].mean()), 5),
                        "lon": round(float(coco["lon"].mean()), 5)},
            "pepper": {"lat": round(float(pepp["lat"].mean()), 5),
                       "lon": round(float(pepp["lon"].mean()), 5)},
            "coconut_to_pepper_km": round(_km(coco["lat"].mean(), coco["lon"].mean(),
                                              pepp["lat"].mean(), pepp["lon"].mean()), 3),
        },
        "env_niche": {c: niche(c) for c in niche_cols},
        "taluk_proportion_train": taluk_pct.to_dict(orient="index"),
        "plots": ["spatial_scatter.png", "spatial_density.png",
                  "spatial_env_niche.png"],
        "overlap_note": "taluk grid means both classes occupy the SAME villages "
                        "in training taluka; no class-exclusive tiles exist.",
    }


def _km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# --------------------------------------------------------------------------- #
# Phase 8 — feature importance (RF permutation, tabular)
# --------------------------------------------------------------------------- #

def phase8() -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler

    sets = split_sets(load_frame())
    train, val = sets["train"], sets["val"]
    joint = pd.concat([train, val]).reset_index(drop=True)
    bm = joint["crop_label"].isin(BINARY).to_numpy()
    y = (joint["crop_label"] == "pepper").to_numpy()[bm].astype(int)
    joint_bin = joint[bm].copy()
    joint_bin["label"] = y
    j_tr = joint_bin.iloc[:int((train["crop_label"].isin(BINARY)).sum())]
    j_va = joint_bin.iloc[int((train["crop_label"].isin(BINARY)).sum()):]
    X_joint, meta = build_matrix(joint_bin, NUMERIC, CATEGORICAL)
    split_p = len(j_tr)

    rf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X_joint[:split_p], y[:split_p])
    imp = permutation_importance(rf, X_joint[split_p:], y[split_p:], n_repeats=15,
                                 random_state=SEED, scoring="roc_auc", n_jobs=-1)
    cols = NUMERIC + CATEGORICAL
    fimp = pd.DataFrame({
        "feature": cols,
        "group": [_group_of(c) for c in cols],
        "imp_mean": imp.importances_mean,
        "imp_std": imp.importances_std,
    }).sort_values("imp_mean", ascending=False).reset_index(drop=True)
    fimp["rank"] = fimp.index + 1

    top20 = fimp.head(20)
    coord_dom = float(fimp[fimp["feature"].isin(["lat", "lon"])]["imp_mean"].sum())
    total_imp = float(fimp["imp_mean"].sum())
    grouped = fimp.groupby("group")["imp_mean"].sum() \
        .sort_values(ascending=False).to_dict()
    grouped = {k: round(float(v), 6) for k, v in grouped.items()}

    out = {
        "phase": "R5.6 Phase 8",
        "model": "RandomForestClassifier(300 trees, class_weight=balanced)",
        "metric": "permutation importance on validation split (ROC-AUC drop)",
        "grouped_importance": grouped,
        "coordinates_dominance": {
            "lat_lon_sum": round(coord_dom, 6),
            "share_of_total": round(coord_dom / total_imp, 6) if total_imp else None,
        },
        "top20": top20.to_dict(orient="records"),
    }
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

PHASES: dict[str, Any] = {
    "0": phase0, "1": phase1, "2": phase2, "5": phase5, "7": phase7, "8": phase8,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="0", help="comma list or 'all'")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.phases == "all":
        sel = sorted(PHASES)
    else:
        sel = [p.strip() for p in args.phases.split(",") if p.strip()]
    available = set(PHASES)
    unknown = [p for p in sel if p not in available]
    if unknown:
        parser.error(f"unknown phases: {unknown}; available: {sorted(available)}")

    for p in sel:
        print(f"=== PHASE {p} ===")
        res = PHASES[p]()
        if isinstance(res, dict):
            parts = [("meta", phase0 if p == "0" else None)]
            if p == "0":
                write_json(OUT_DIR / "R5.6_METADATA.json", res)
                print("  metadata keys:", sorted(res.keys()))
            elif p == "1":
                write_json(OUT_DIR / "binary_manifest.json", res)
            elif p == "2":
                write_json(OUT_DIR / "tabular_baselines.json", res)
            elif p == "5":
                write_json(OUT_DIR / "feature_separability.json", res)
            elif p == "7":
                write_json(OUT_DIR / "spatial_separability.json", res)
            elif p == "8":
                write_json(OUT_DIR / "feature_importance.json", res)
    print("\nAll requested phases complete. Artifacts in:", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())