"""R5.5 classifier-collapse — local (CPU, tabular-only) diagnostic suite.

Complements the Kaggle GPU phases in ``diagnose_collapse_kaggle.py``. This
script reads ONLY the frozen corpus CSV + manifest and runs:

  Phase 2b   per-class loss/class-weight arithmetic audit (torch CPU)
  Phase 5    tabular-only discrimination  (LR / RF / GB / MLP) on the SAME
             official frozen taluk split  + per-feature stats
  Phase 10   duplicate / spatial-cluster audit (train only)
  Phase 11   crop-specific spatial distribution (train only)
  Phase 13   simple-baseline comparison table (majority / tabular / geo-only)

No imagery, no model training, no hyperparameter search. Same eligibility and
split rules as the frozen corpus loader (benchmark_eligible + taluk mapping).

Run from repo root:

    python training/kaggle/scripts/r5_5_local_baselines.py
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
CSV_PATH = REPO_ROOT / "govt_crop_matched_v2" / "crop_supervised_v2.csv"
MANIFEST_PATH = REPO_ROOT / "training_manifests" / "crop_supervised_v2.0_manifest.json"
REPORTS = REPO_ROOT / "reports"

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


# --------------------------------------------------------------------------- #
# Feature matrix
# --------------------------------------------------------------------------- #

def build_matrix(df: pd.DataFrame, numeric: list[str], categorical: list[str],
                 impute_done: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    X_num = df[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    missing_num = int(np.isnan(X_num).sum())
    medians = np.nanmedian(X_num, axis=0)
    for j in range(X_num.shape[1]):
        col = X_num[:, j]
        if np.isnan(col).any():
            col[np.isnan(col)] = medians[j]
    X_cat = df[categorical].astype(object)
    X_cat = X_cat.fillna("__nan__")
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


def onehot(X: np.ndarray, categorical_cols: list[str],
           cat_meta: dict[str, Any]) -> np.ndarray:
    n_num = len(NUMERIC)
    X_num = X[:, :n_num]
    parts = [X_num]
    for j, col in enumerate(categorical_cols):
        codes = X[:, n_num + j].astype(int)
        n = cat_meta[col]["unique"]
        oh = np.zeros((X.shape[0], n), dtype=np.float64)
        oh[np.arange(X.shape[0]), codes] = 1.0
        parts.append(oh)
    return np.hstack(parts)


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


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y_true, score)), 4)


# --------------------------------------------------------------------------- #
# Phase 2b — loss arithmetic audit
# --------------------------------------------------------------------------- #

def phase2_loss_audit() -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    counts = torch.tensor([4292.0, 1599.0, 29.0, 4.0])
    w_sqrt = 1.0 / torch.sqrt(counts)
    w_sqrt = w_sqrt / w_sqrt.mean()
    w_bal = counts.sum() / (counts.numel() * counts)
    w_bal = w_bal / w_bal.mean()

    rng = np.random.RandomState(0)
    logits = torch.tensor(rng.normal(0, 1, (8, 4)).astype(np.float32), requires_grad=True)
    targets = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3])

    def ce_none(weights=None, smoothing=0.0):
        if smoothing > 0:
            lp = F.log_softmax(logits, dim=-1)
            td = torch.full_like(lp, smoothing / 3)
            td.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
            loss = -(td * lp).sum(-1)
        else:
            loss = F.cross_entropy(logits, targets, reduction="none")
        if weights is not None:
            loss = loss * weights.to(loss.device).gather(0, targets)
        return loss

    def focal_none(gamma=2.0, alpha=None):
        lp = F.log_softmax(logits, dim=-1)
        p = lp.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        lpn = lp.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -(1 - p) ** gamma * lpn
        if alpha is not None:
            loss = loss * alpha.to(loss.device).gather(0, targets)
        return loss

    frames = {
        "unweighted_ce": {"per_class": {}, "mean": None},
        "sqrt_inv_ce": {"per_class": {}, "mean": None},
        "balanced_ce": {"per_class": {}, "mean": None},
        "smoothing015_sqrt_inv_ce": {"per_class": {}, "mean": None},
        "focal_g2": {"per_class": {}, "mean": None},
        "focal_g2_sqrt_inv": {"per_class": {}, "mean": None},
    }
    vals = {
        "unweighted_ce": ce_none(),
        "sqrt_inv_ce": ce_none(weights=w_sqrt),
        "balanced_ce": ce_none(weights=w_bal),
        "smoothing015_sqrt_inv_ce": ce_none(weights=w_sqrt, smoothing=0.15),
        "focal_g2": focal_none(gamma=2.0),
        "focal_g2_sqrt_inv": focal_none(gamma=2.0, alpha=w_sqrt),
    }
    for k, v in vals.items():
        for c in range(4):
            mask = targets == c
            if mask.any():
                frames[k]["per_class"][SUPERVISED[c]] = round(
                    v[mask].mean().item(), 5)
        frames[k]["mean"] = round(v.mean().item(), 5)

    # Gradient contribution per class measured against the LOGITS token palette
    # (how much each class's logit moves under each loss formulation).
    logits.requires_grad_(True)
    grads = {}
    keys = list(vals)
    for i, (k, v) in enumerate(vals.items()):
        v.mean().backward(retain_graph=(i != len(keys) - 1))
        g = logits.grad.abs()
        per = {SUPERVISED[c]: round(float(g[:, c].sum()), 6) for c in range(4)}
        grads[k] = per
        logits.grad.zero_()

    return {
        "train_counts": {c: int(n) for c, n in zip(SUPERVISED, counts.tolist())},
        "class_weights": {
            "sqrt_inv": {c: round(float(w_sqrt[i]), 6) for i, c in enumerate(SUPERVISED)},
            "balanced": {c: round(float(w_bal[i]), 6) for i, c in enumerate(SUPERVISED)},
        },
        "weight_ratios": {
            "sqrt_inv_coffee_over_coconut": round(float(w_sqrt[2] / w_sqrt[0]), 3),
            "sqrt_inv_cardamom_over_coconut": round(float(w_sqrt[3] / w_sqrt[0]), 3),
            "balanced_cardamom_over_coffee": round(float(w_bal[3] / w_bal[2]), 3),
        },
        "per_class_loss": frames,
        "per_class_grad_norm": grads,
        "verification_notes": [
            "weights gathered along the TARGET class dim (correct dimension)",
            "focal = (1-p)^gamma * CE; gamma=2.0 multiplicatively SUPPRESSES the "
            "target-token gradient by (1-p)^2 — for uncertain confident draws the "
            "rare-class boost is mostly cancelled",
            "focal + sqrt_inv alpha is applied AFTER the (1-p)^gamma down-weight",
            "label_smoothing 0.15 splits target probability 0.85/0.05 to all "
            "off-target classes, further diluting the per-sample gradient",
        ],
    }


# --------------------------------------------------------------------------- #
# Phase 5 — feature stats
# --------------------------------------------------------------------------- #

def phase5_feature_stats(df: pd.DataFrame) -> dict[str, Any]:
    train = df[df["split"] == "train"]
    out: dict[str, Any] = {}
    for col in NUMERIC + CATEGORICAL:
        if col in NUMERIC:
            v = pd.to_numeric(train[col], errors="coerce")
            entry: dict[str, Any] = {
                "type": "numeric",
                "unique": int(v.nunique()),
                "missing": int(v.isna().sum()),
                "mean": round(float(v.mean()), 4) if v.notna().any() else None,
                "std": round(float(v.std()), 4) if v.notna().any() else None,
                "min": round(float(v.min()), 4) if v.notna().any() else None,
                "max": round(float(v.max()), 4) if v.notna().any() else None,
                "p95": round(float(v.quantile(0.95)), 4) if v.notna().any() else None,
            }
        else:
            v = train[col].astype(str)
            entry = {
                "type": "categorical",
                "unique": int(v.nunique()),
                "missing": int(train[col].isna().sum()),
                "top": {str(k): int(c) for k, c in v.value_counts().head(6).items()},
            }
        out[col] = entry

    # constant / near-constant numeric flags
    constants = [c for c in NUMERIC if out[c]["unique"] <= 2]
    near_const = [c for c in NUMERIC
                  if out[c]["std"] is not None and out[c]["std"] < 1e-4]
    return {"per_feature": out,
            "train_n": int(len(train)),
            "near_constant_numeric": constants,
            "near_zero_std_numeric": near_const,
            "note": "unique counts are on the TRAIN split only; many features are "
                    "near-constant because a single env composite per grid cell "
                    "is shared across hundreds of village samples",
            }


# --------------------------------------------------------------------------- #
# Phase 11 — crop spatial distribution (train only)
# --------------------------------------------------------------------------- #

def phase11_spatial(train: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for crop in SUPERVISED:
        sub = train[train["crop_label"] == crop]
        d: dict[str, Any] = {"n": int(len(sub))}
        for coord in ("lat", "lon"):
            v = pd.to_numeric(sub[coord], errors="coerce")
            d[coord] = {
                "mean": round(float(v.mean()), 5), "std": round(float(v.std()), 5),
                "min": round(float(v.min()), 5), "max": round(float(v.max()), 5),
                "unique": int(v.nunique()),
            }
        d["taluk"] = sub["location_taluk"].value_counts().to_dict()
        d["year"] = sub["year"].value_counts().to_dict()
        d["season"] = sub["season"].value_counts().to_dict()
        d["hobli"] = sub["location_hobli"].value_counts().to_dict() \
            if "location_hobli" in sub else {}
        out[crop] = d
    # pairwise pairwise centroid distances (train, ec) and overlap in lat/lon box
    cents = {}
    for crop in SUPERVISED:
        sub = train[train["crop_label"] == crop]
        cents[crop] = (float(sub["lat"].mean()), float(sub["lon"].mean()))
    d_box: dict[str, Any] = {}
    for i in range(len(SUPERVISED)):
        for j in range(i + 1, len(SUPERVISED)):
            a, b = SUPERVISED[i], SUPERVISED[j]
            d_box[f"{a}_vs_{b}"] = {
                "lat_centroid_km": round(
                    _km(cents[a][0], cents[a][1], cents[b][0], cents[b][1]), 3),
            }
    out["centroid_distances_km"] = d_box
    return out


def _km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# --------------------------------------------------------------------------- #
# Phase 10 — duplicate / spatial-cluster audit (train only)
# --------------------------------------------------------------------------- #

def phase10_duplicates(train: pd.DataFrame) -> dict[str, Any]:
    from scipy.spatial import cKDTree
    sub = train.reset_index(drop=True)
    lat = pd.to_numeric(sub["lat"], errors="coerce").to_numpy()
    lon = pd.to_numeric(sub["lon"], errors="coerce").to_numpy()
    valid = np.isfinite(lat) & np.isfinite(lon)
    lat_v, lon_v = lat[valid], lon[valid]
    ids = sub["record_id"].astype(str).to_numpy()[valid]
    crops = sub["crop_label"].to_numpy()[valid]
    xy = np.stack([
        (lon_v + 180.0) * np.pi / 180.0 * 6371.0,
        np.radians(lat_v) * 6371.0,
    ], axis=1)
    tree = cKDTree(xy)
    dists, idx = tree.query(xy, k=2)
    nn_dist_km = dists[:, 1]
    m = nn_dist_km * 1000.0

    thresholds_m = [10, 50, 100, 250, 500, 1000]
    band = {}
    for t in thresholds_m:
        band[str(t)] = int((m <= t).sum())
    exact_same_gps = int((nn_dist_km < 1e-9).sum())

    same_geometry = 0
    geometry_keys = []
    for c in ["lat", "lon"]:
        geometry_keys.append((c, sub["record_id"]))
    # exact GPS duplicates (same rounded to 6dp)
    key = (sub[valid].assign(gps=lat_v.round(6).astype(str) + "_"
                                  + lon_v.round(6).astype(str))["gps"])
    dup_gps = int(key.duplicated().sum())

    by_crop: dict[str, Any] = {}
    for crop in SUPERVISED:
        cm = m[crops == crop]
        by_crop[crop] = {
            "nn_mean_m": round(float(cm.mean()), 1),
            "nn_median_m": round(float(np.median(cm)), 1),
            "nn_p95_m": round(float(np.percentile(cm, 95)), 1),
            "nn_max_m": round(float(cm.max()), 1),
            "share_within_100m": round(float((cm <= 100).mean()), 4),
            "share_within_500m": round(float((cm <= 500).mean()), 4),
        }

    # minimal spanning of distinct env cells
    env_unique = int(sub["env_match_distance_m"].nunique())
    return {
        "train_n": int(len(sub)),
        "valid_coords": int(valid.sum()),
        "same_gps_exact": int(exact_same_gps),
        "duplicate_gps_ids_6dp": dup_gps,
        "nearest_neighbour_distance_km": {
            "mean": round(float(nn_dist_km.mean()), 4),
            "median": round(float(np.median(nn_dist_km)), 4),
            "p95": round(float(np.percentile(nn_dist_km, 95)), 4),
            "max": round(float(nn_dist_km.max()), 4),
        },
        "share_within": {str(t): round(float((m <= t).mean()), 4) for t in thresholds_m},
        "count_within": band,
        "per_crop_nn": by_crop,
        "distinct_env_diag": {"env_match_distance_m_unique": env_unique},
        "note": "nearest-neighbour is the nearest OTHER sample at gps-haversine "
                "distance. 100m in DK taluk grid scale corresponds to samples in "
                "the SAME village/env cell.",
    }


# --------------------------------------------------------------------------- #
# Baseline model experiments
# --------------------------------------------------------------------------- #

def _binary_variants(X_train, y_train, X_val, y_val, seed: int) -> list[dict[str, Any]]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    binm = {"train": (X_train[y_train < 2], y_train[y_train < 2]),
            "val": (X_val[y_val < 2], y_val[y_val < 2])}
    xt, yt = binm["train"]
    xv, yv = binm["val"]
    scaler = StandardScaler().fit(xt)
    xt_s, xv_s = scaler.transform(xt), scaler.transform(xv)
    names = [SUPERVISED[0], SUPERVISED[1]]
    prior = float((yv == 0).mean())
    rows: list[dict[str, Any]] = []
    configs = [
        ("binary_lr", LogisticRegression(max_iter=2000, random_state=seed,
                                         class_weight="balanced")),
        ("binary_lr_unweighted", LogisticRegression(max_iter=2000, random_state=seed)),
        ("binary_rf", RandomForestClassifier(n_estimators=200, random_state=seed,
                                             class_weight="balanced")),
        ("binary_gb", GradientBoostingClassifier(n_estimators=200, random_state=seed,
                                                 max_depth=4)),
    ]
    for name, clf in configs:
        clf.fit(xt, yt)
        pred = clf.predict(xv)
        m = metrics(yv, pred, names)
        auc = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(xv)
            if proba.shape[1] == 2:
                auc = binary_auc(yv, proba[:, 1])
        rows.append({"variant": name, "metrics": m, "roc_auc": auc,
                     "train_acc": round(float((clf.predict(xt) == yt).mean()), 4)})
    # coords-only binary LR (leak check for geographic shortcut)
    coord_idx = [NUMERIC.index("lat"), NUMERIC.index("lon")]
    clf_c = LogisticRegression(max_iter=2000, random_state=seed)
    clf_c.fit(xt[:, coord_idx], yt)
    pred_c = clf_c.predict(xv[:, coord_idx])
    rows.append({"variant": "binary_lr_coords_only",
                 "metrics": metrics(yv, pred_c, names),
                 "roc_auc": binary_auc(yv, clf_c.predict_proba(xv[:, coord_idx])[:, 1])
                 if hasattr(clf_c, "predict_proba") else None,
                 "train_acc": round(float((clf_c.predict(xt[:, coord_idx]) == yt).mean()), 4),
                 })
    return {"val_majority_prior_acc": round(prior, 4), "variants": rows}


def phase5_tabular(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    df = load_frame()
    sets = split_sets(df)
    n = {"train": len(sets["train"]), "val": len(sets["val"]),
         "test": len(sets["test"])}

    X_all, meta = build_matrix(pd.concat([sets["train"], sets["val"], sets["test"]]),
                               NUMERIC, CATEGORICAL)
    n_num = len(NUMERIC)
    tr = len(sets["train"]); vl = len(sets["val"])
    X_tr, X_va, X_te = X_all[:tr], X_all[tr:tr + vl], X_all[tr + vl:]
    y_tr = sets["train"]["class_id"].to_numpy()
    y_va = sets["val"]["class_id"].to_numpy()
    y_te = sets["test"]["class_id"].to_numpy()

    # numeric-only scaled matrix used by the geometric baselines
    scaler = StandardScaler().fit(X_tr[:, :n_num])
    Xn_tr = np.hstack([scaler.transform(X_tr[:, :n_num]), X_tr[:, n_num:]])
    Xn_va = np.hstack([scaler.transform(X_va[:, :n_num]), X_va[:, n_num:]])
    Xn_te = np.hstack([scaler.transform(X_te[:, :n_num]), X_te[:, n_num:]])

    from sklearn.preprocessing import OneHotEncoder
    oh = OneHotEncoder(handle_unknown="ignore").fit(X_tr[:, n_num:])
    Xho_tr = np.hstack([scaler.transform(X_tr[:, :n_num]),
                        oh.transform(X_tr[:, n_num:]).toarray()])
    Xho_va = np.hstack([scaler.transform(X_va[:, :n_num]),
                        oh.transform(X_va[:, n_num:]).toarray()])
    Xho_te = np.hstack([scaler.transform(X_te[:, :n_num]),
                        oh.transform(X_te[:, n_num:]).toarray()])

    results: list[dict[str, Any]] = []

    def run(name, clf, Xa, Xb, ya, yb, key):
        clf.fit(Xa, ya)
        pred_b = clf.predict(Xb)
        m = metrics(yb, pred_b, SUPERVISED)
        train_acc = round(float((clf.predict(Xa) == ya).mean()), 4)
        results.append({"model": name, "split": key, **m,
                        "implicit_train_acc": train_acc})
        return m

    # majority reference
    maj_pred = np.full(len(y_va), 0)
    results.append({"model": "majority_constant", "split": "val",
                    **metrics(y_va, maj_pred, SUPERVISED)})
    maj_pred_t = np.full(len(y_te), 0)
    results.append({"model": "majority_constant", "split": "test",
                    **metrics(y_te, maj_pred_t, SUPERVISED)})

    cd = {"coords": [NUMERIC.index("lat"), NUMERIC.index("lon")],
          "distance": [NUMERIC.index("spatial_match_distance_km"),
                       NUMERIC.index("env_match_distance_m")]}
    geom = list(set(cd["coords"] + cd["distance"]))

    configs = [
        ("lr_numeric", LogisticRegression(max_iter=3000, random_state=seed,
                                          class_weight="balanced")),
        ("lr_plain_numeric", LogisticRegression(max_iter=3000, random_state=seed)),
        ("rf_numeric", RandomForestClassifier(n_estimators=250, random_state=seed,
                                              class_weight="balanced")),
        ("gb_numeric", GradientBoostingClassifier(n_estimators=250, random_state=seed,
                                                  max_depth=4)),
        ("mlp_numeric", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                                      random_state=seed)),
    ]
    for name, clf in configs:
        run(name + "_raw", clf, Xn_tr[:, :n_num], Xn_va[:, :n_num],
            y_tr, y_va, "val")
    # coords-only LR to measure geographic shortcut
    run("lr_coords_only", LogisticRegression(max_iter=2000, random_state=seed,
                                             class_weight="balanced"),
        Xn_tr[:, cd["coords"]], Xn_va[:, cd["coords"]], y_tr, y_va, "val")
    run("lr_no_coords", LogisticRegression(max_iter=2000, random_state=seed,
                                           class_weight="balanced"),
        np.delete(Xn_tr, cd["coords"], axis=1), np.delete(Xn_va, cd["coords"], axis=1),
        y_tr, y_va, "val")
    run("rf_onehot", RandomForestClassifier(n_estimators=250, random_state=seed,
                                            class_weight="balanced"),
        Xho_tr, Xho_va, y_tr, y_va, "val")
    # evaluate the best test-set proxy: RF + LR on TEST split too
    rf_probe = RandomForestClassifier(n_estimators=250, random_state=seed,
                                      class_weight="balanced")
    rf_probe.fit(Xn_tr[:, :n_num], y_tr)
    results.append({"model": "rf_numeric", "split": "test",
                    **metrics(y_te, rf_probe.predict(Xn_te[:, :n_num]), SUPERVISED)})
    lr_probe = LogisticRegression(max_iter=3000, random_state=seed,
                                  class_weight="balanced")
    lr_probe.fit(Xn_tr[:, :n_num], y_tr)
    results.append({"model": "lr_numeric", "split": "test",
                    **metrics(y_te, lr_probe.predict(Xn_te[:, :n_num]), SUPERVISED)})

    binr = _binary_variants(Xn_tr[:, :n_num], y_tr, Xn_va[:, :n_num], y_va, seed)

    return {"split_sizes": n, "feature_schema": meta,
            "results": results, "binary": binr}


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    df = load_frame()
    sets = split_sets(df)

    report: dict[str, Any] = {
        "phase": "R5.5 local (CPU) diagnostics",
        "corpus": {"csv": str(CSV_PATH.name)},
        "split_sizes": {k: len(v) for k, v in sets.items()},
        "class_mapping": {c: i for i, c in enumerate(SUPERVISED)},
        "train_counts": sets["train"]["crop_label"].value_counts().to_dict(),
        "val_counts": sets["val"]["crop_label"].value_counts().to_dict(),
        "test_counts": sets["test"]["crop_label"].value_counts().to_dict(),
        "seed": args.seed,
    }

    print("=== PHASE 2b: LOSS / CLASS-WEIGHT ARITHMETIC AUDIT ===")
    report["loss_audit"] = phase2_loss_audit()
    print(json.dumps(report["loss_audit"], indent=2, default=_json_default)[:2400])

    print("\n=== PHASE 5: FEATURE STATS (train) ===")
    report["feature_stats"] = phase5_feature_stats(df)
    print("  near-constant numeric:", report["feature_stats"]["near_constant_numeric"])
    print("  near-zero-std numeric:",
          report["feature_stats"]["near_zero_std_numeric"])

    print("\n=== PHASE 11: CROP SPATIAL DISTRIBUTION (train) ===")
    report["spatial"] = phase11_spatial(sets["train"])
    print(json.dumps(report["spatial"], indent=2, default=_json_default)[:1800])

    print("\n=== PHASE 10: DUPLICATE / CLUSTER AUDIT (train) ===")
    report["duplicates"] = phase10_duplicates(sets["train"])
    print(json.dumps(report["duplicates"], indent=2, default=_json_default)[:1600])

    print("\n=== PHASE 5/13: TABULAR BASELINES (frozen split) ===")
    report["tabular"] = phase5_tabular(args.seed)
    for r in report["tabular"]["results"]:
        tag = f"{r['model']:<20s} [{r['split']}]"
        print(f"  {tag} acc={r['accuracy']:.4f} bal={r['balanced_accuracy']:.4f} "
              f"macro_f1={r['macro_f1']:.4f} beats={r['beats_majority']} "
              f"pred={r['prediction_distribution']}")
    binr = report["tabular"]["binary"]
    print(f"  binary val majority prior: {binr['val_majority_prior_acc']}")
    for v in binr["variants"]:
        print(f"    {v['variant']:<26s} acc={v['metrics']['accuracy']:.4f} "
              f"bal={v['metrics']['balanced_accuracy']:.4f} "
              f"auc={v['roc_auc']} beats={v['metrics']['beats_majority']}")

    out = REPORTS / "R5.5_local_baselines.json"
    out.write_text(json.dumps(report, indent=2, default=_json_default, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n  saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())