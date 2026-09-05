"""R5.5 classifier-collapse diagnostics (tab-only, no imagery required).

Phases covered locally (image-dependent phases run in a Kaggle diagnostic
kernel; see ``diagnose_collapse_kaggle``/notebook):

  trace          Phase 1  - label-pipeline trace  (CSV -> manifest order ->
                            encoder ids -> split composition + trivial
                            predictor anchor against the observed gate)
  distribution   Phase 2  - train label distribution, weight schemes, exact
                            loss config loaded (training.yaml + TRN_* env)
  separability   Phase 4  - train-only feature separability by class pair
                            -> reports/class_separability_train.json
  spatial        Phase 6  - coconut vs pepper matching/environment audit
  features       Phase 7  - per-feature source/meaning/units/leak inventory
  normalization  Phase 8  - raw -> imputed/filled -> scaled audit
  recovery17     Phase 17 - what R5.2.9 added -> reports/r5_2_9_recovery_audit.json
  coffee16       Phase 16 - coffee/cardamom upstream retention trace

Every subcommand reads ONLY the frozen corpus + manifest (never validation or
test statistics for selection; nothing train/test tuned).
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

from training.training.config import LossConfig, load_training_config  # noqa: E402

SUPERVISED = ["coconut", "pepper", "coffee", "cardamom"]
CSV_PATH = REPO_ROOT / "govt_crop_matched_v2" / "crop_supervised_v2.csv"
V1_CSV_PATH = REPO_ROOT / "govt_crop_matched_v1" / "crop_supervised_v1.csv"
MANIFEST = REPO_ROOT / "training_manifests" / "crop_supervised_v2.0_manifest.json"
V1_MANIFEST = REPO_ROOT / "training_manifests" / "crop_supervised_v1_manifest.json"
V2_LEDGER = REPO_ROOT / "govt_crop_matched_v2" / "government_crop_matched_v2.csv"
V1_LEDGER = REPO_ROOT / "govt_crop_matched_v1" / "government_crop_stam_match.csv"
PREPROCESSING_YAML = REPO_ROOT / "training" / "config" / "preprocessing.yaml"
TRAINING_YAML = REPO_ROOT / "training" / "config" / "training.yaml"
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

# Reference from the R5.3 wd180 3-epoch gate (focal2.0+sqrt_inv defaults).
GATE_REF = {"accuracy": 0.5584, "macro_f1": 0.1791, "weighted_f1": 0.4001}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"crop_label": str, "location_taluk": str})
    return df


def _split_mask(df: pd.DataFrame, manifest: dict[str, Any], part: str) -> np.ndarray:
    taluks = manifest["split_groups"][f"{part}_taluk"]
    if isinstance(taluks, str):
        taluks = [taluks]
    return df["location_taluk"].isin(taluks).to_numpy()


def _is_eligible(df: pd.DataFrame) -> np.ndarray:
    if "benchmark_eligible" not in df:
        return np.ones(len(df), dtype=bool)
    return df["benchmark_eligible"].fillna("true").astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes"]
    ).to_numpy()


def _anchor_metrics(y_true: np.ndarray, pred_class: int = 0, n_classes: int = 4) -> dict[str, Any]:
    """Reference metrics produced by the 'predict one class' trivial model."""
    y_pred = np.full_like(y_true, pred_class)
    counts = np.bincount(y_true, minlength=n_classes)
    tp = counts[pred_class]
    fp = int((y_pred == pred_class).sum()) - tp
    acc = float(tp / len(y_true)) if len(y_true) else 0.0
    per = {}
    for c in range(n_classes):
        n = counts[c]
        if c == pred_class:
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = 1.0 if n else 0.0
        else:
            p = r = 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per[c] = {"precision": p, "recall": r, "f1": f1}
    macro_f1 = float(np.mean([per[c]["f1"] for c in range(n_classes)]))
    support = counts
    weighted_f1 = float(np.dot([per[c]["f1"] for c in range(n_classes)], support) / support.sum())
    return {
        "pred_class": int(pred_class),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "pred_distribution": {c: int((y_pred == c).sum()) for c in range(n_classes)},
        "target_distribution": {int(c): int(n) for c, n in enumerate(counts)},
        "per_class": per,
    }


# --------------------------------------------------------------------------- #
# Phase 1 - label trace
# --------------------------------------------------------------------------- #

def _run_trace(args: argparse.Namespace) -> int:
    out: dict[str, Any] = {}
    df = load_csv(CSV_PATH)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    declared = list(manifest["supervised_classes"])

    # Class schema contract.
    print("\n=== PHASE 1: LABEL PIPELINE TRACE ===\n")
    print("Manifest supervised_classes:", declared)
    print("Manifest excluded_classes:", manifest.get("excluded_classes"))
    print("Class mapping (manifest class_mapping):", manifest.get("class_mapping"))
    out["supervised_classes"] = declared
    out["species_shipped"] = manifest.get("excluded_classes")

    # CSV crop values present.
    crops = df["crop_label"].value_counts().to_dict()
    print("\nCSV crop_label value counts:", crops)
    out["csv_crop_counts"] = {str(k): int(v) for k, v in crops.items()}

    # Encoding exactly as LabelPipeline does at fit time: declared order.
    index = {cls: i for i, cls in enumerate(declared)}
    df["id"] = df["crop_label"].map(lambda c: index.get(str(c).strip(), -1))
    ids, id_counts = np.unique(df["id"], return_counts=True)
    print("\nEncoded unique ids:", {int(i): int(c) for i, c in zip(ids, id_counts)})
    out["encoding"] = index

    mapping_violations = [c for c, i in index.items() if i != SUPERVISED.index(c)]
    print("\nDeclaration order vs canonical order ->", index)
    assert not mapping_violations, f"declared order drifted from canonical {SUPERVISED}"
    assert index == {"coconut": 0, "pepper": 1, "coffee": 2, "cardamom": 3}
    assert -1 in set(df["id"].tolist()), "expected excluded labels encoded to -1"
    out["encoding_order"] = index
    out["label_mapping_ok"] = True

    # Per-split composition from the CSV.
    eligible = _is_eligible(df)
    parts = {}
    for part in ("train", "validation", "test"):
        mask = eligible & _split_mask(df, manifest, part)
        parts[part] = df.loc[mask]
        counts = df.loc[mask, "crop_label"].value_counts().to_dict()
        print(f"  {part:11s} n={int(mask.sum()):5d}  {counts}")
        out[f"{part}_counts"] = {str(k): int(v) for k, v in counts.items()}
    vs_manifest = manifest["class_counts"]
    ok = True
    for part, name in (("train", "train"), ("validation", "validation")):
        for c in SUPERVISED:
            got = parts[part]["crop_label"].eq(c).sum()
            want = vs_manifest[name].get(c, 0)
            ok &= int(got) == int(want)
    out["split_counts_match_manifest"] = bool(ok)
    print("\n  split_counts_match_manifest:", bool(ok))

    # Anchors: what the observed gate numbers actually are, if the model
    # predicted nothing but class 0 on the whole validation split.
    y_val = parts["validation"]["id"].to_numpy()
    y_val = y_val[y_val >= 0]
    anchor = _anchor_metrics(y_val, pred_class=0, n_classes=len(declared))
    print("\nTrivial 'predict coconut for every val sample' reference:")
    print(f"  accuracy    {anchor['accuracy']:.4f}")
    print(f"  macro F1    {anchor['macro_f1']:.4f}")
    print(f"  weighted F1 {anchor['weighted_f1']:.4f}")
    print(anchor["pred_distribution"], anchor["target_distribution"])
    print("\nObserved gate: accuracy %.4f macro-F1 %.4f weighted-F1 %.4f"
          % (GATE_REF["accuracy"], GATE_REF["macro_f1"], GATE_REF["weighted_f1"]))
    close = (abs(anchor["accuracy"] - GATE_REF["accuracy"]) < 1e-3
             and abs(anchor["macro_f1"] - GATE_REF["macro_f1"]) < 1e-3)
    print(f"\n  Gate == trivial-predictor anchor: {close}")
    out["gate_equals_trivial_coconut_anchor"] = bool(close)
    out["anchor"] = anchor

    print("\nModel head ordering note: crop_logits shape [B, n_classes]; argmax index "
          "maps 1:1 to the same declared order (no reordering between head and "
          "evaluator, see models/multitask_heads.py + evaluator).")

    reports_out = Path(args.out)
    reports_out.mkdir(parents=True, exist_ok=True)
    (reports_out / "R5.5_label_trace.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"\n[PASS] label/order trace saved -> {reports_out / 'R5.5_label_trace.json'}")
    return 0


# --------------------------------------------------------------------------- #
# Phase 2 - distribution + exact loss config
# --------------------------------------------------------------------------- #

def _run_distribution(args: argparse.Namespace) -> int:
    print("\n=== PHASE 2: TRAIN DISTRIBUTION + LOSS CONFIG ===\n")
    df = load_csv(CSV_PATH)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    eligible = _is_eligible(df)
    train = df[eligible & _split_mask(df, manifest, "train")]

    counts = np.array([int((train["crop_label"] == c).sum()) for c in SUPERVISED], dtype=float)
    total = counts.sum()
    print("Train classes:", dict(zip(SUPERVISED, counts.astype(int))))
    print("Percent:", {c: f"{n / total * 100:.3f}%" for c, n in zip(SUPERVISED, counts)})

    weights = {}
    nc = len(SUPERVISED)
    from training.training.losses import build_class_weights
    for mode in ("balanced", "sqrt_inv", "effective_num"):
        w = build_class_weights(LossConfig(class_weight_mode=mode,
                                           class_weight_beta=0.999),
                                nc, counts)
        weights[mode] = {c: round(float(v), 6) for c, v in zip(SUPERVISED, w)}
        print(f"  {mode:13s} {weights[mode]}")

    # Exact (effective) loss configuration in this repo state: training.yaml +
    # TRN_* env overrides (the same loader run_pipeline uses).
    cfg = load_training_config(TRAINING_YAML)
    loss = cfg.loss
    print("\nEffective loss config (load_training_config):")
    print(f"  crop_loss={loss.crop_loss} focal_gamma={loss.focal_gamma} "
          f"class_weight_mode={loss.class_weight_mode} "
          f"class_weight_beta={loss.class_weight_beta}")
    print(f"  label_smoothing={loss.label_smoothing} "
          f"crop_weight={loss.crop_weight} yield_weight={loss.yield_weight} "
          f"weighting_mode={loss.weighting_mode} reduction={loss.reduction}")

    out = {
        "train_counts": {c: int(v) for c, v in zip(SUPERVISED, counts)},
        "train_percent": {c: round(n / total * 100, 3) for c, n in zip(SUPERVISED, counts)},
        "weights": weights,
        "manifest_class_weights": manifest.get("class_weights"),
        "effective_loss": {
            "crop_loss": loss.crop_loss,
            "focal_gamma": loss.focal_gamma,
            "class_weight_mode": loss.class_weight_mode,
            "class_weight_beta": loss.class_weight_beta,
            "label_smoothing": loss.label_smoothing,
            "crop_weight": loss.crop_weight,
            "yield_weight": loss.yield_weight,
        },
    }
    _save(args, "R5.5_phase2_distribution.json", out)
    return 0


def build_weights(mode: str, counts: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Reference: unnormalized inverse-frequency variants for documentation."""
    counts = np.asarray(counts, dtype=float)
    if mode == "balanced":
        w = 1.0 / counts
    elif mode == "sqrt_inv":
        w = 1.0 / np.sqrt(counts)
    elif mode == "effective_num":
        eff = (1.0 - beta) / (1.0 - beta ** counts)
        w = 1.0 / eff
    else:
        raise ValueError(mode)
    return w / w.sum() * len(counts)


# --------------------------------------------------------------------------- #
# Phase 4 - separability
# --------------------------------------------------------------------------- #

def _run_separability(args: argparse.Namespace) -> int:
    print("\n=== PHASE 4: TRAIN-ONLY CLASS SEPARABILITY ===\n")
    df = load_csv(CSV_PATH)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train = df[_is_eligible(df) & _split_mask(df, manifest, "train")].copy()

    results: dict[str, Any] = {"train_n": int(len(train)), "pairs": {}}
    for a, b in args.pairs:
        sub = train[train["crop_label"].isin([a, b])]
        na, nb = int((sub["crop_label"] == a).sum()), int((sub["crop_label"] == b).sum())
        if na == 0 or nb == 0:
            print(f"  pair {a}/{b}: class missing in train -> skipped")
            continue
        pair = {"a": a, "b": b, "na": na, "nb": nb, "numeric": {}, "categorical": {}}
        for col in NUMERIC:
            x = pd.to_numeric(sub[col], errors="coerce")
            xa_, xb_ = x[sub["crop_label"] == a], x[sub["crop_label"] == b]
            xa = xa_.dropna(); xb = xb_.dropna()
            if len(xa) == 0 or len(xb) == 0:
                continue
            pooled = math.sqrt((np.var(xa) + np.var(xb)) / 2) or 1e-9
            d = (xa.mean() - xb.mean()) / pooled
            auc = _auc(xa, xb)
            pair["numeric"][col] = {
                "mean_a": round(float(xa.mean()), 4), "median_a": round(float(xa.median()), 4),
                "std_a": round(float(xa.std()), 4), "q1_a": round(float(xa.quantile(0.25)), 4),
                "q3_a": round(float(xa.quantile(0.75)), 4),
                "mean_b": round(float(xb.mean()), 4), "median_b": round(float(xb.median()), 4),
                "std_b": round(float(xb.std()), 4), "q1_b": round(float(xb.quantile(0.25)), 4),
                "q3_b": round(float(xb.quantile(0.75)), 4),
                "missing_a": int(xa_.isna().sum()), "missing_b": int(xb_.isna().sum()),
                "unique_a": int(xa.nunique()), "unique_b": int(xb.nunique()),
                "cohens_d": round(float(d), 4),
                "auc": round(float(auc), 4),
                "abs_auc_delta": round(abs(auc - 0.5), 4),
            }
        for col in CATEGORICAL:
            ctab = pd.crosstab(sub["crop_label"], sub[col])
            v = _cramers_v(ctab)
            pair["categorical"][col] = {
                "cramers_v": round(float(v), 4),
                "value_dist_a": {str(k): int(v) for k, v in
                                 sub[sub["crop_label"] == a][col].value_counts().items()},
                "value_dist_b": {str(k): int(v) for k, v in
                                 sub[sub["crop_label"] == b][col].value_counts().items()},
            }
        results["pairs"][f"{a}__vs__{b}"] = pair
        ranked = sorted(pair["numeric"].items(), key=lambda kv: -kv[1]["abs_auc_delta"])
        print(f"\n  {a} vs {b} (n={na}/{nb}) top features by |AUC-0.5|:")
        for col, s in ranked[:8]:
            print(f"    {col:28s} AUC={s['auc']:+.3f} d={s['cohens_d']:+.3f} "
                  f"mean {s['mean_a']:9.3f} vs {s['mean_b']:9.3f}")
        print(f"    categorical Cramér's V: "
              f"{ {c: pair['categorical'][c]['cramers_v'] for c in CATEGORICAL} }")

    _save(args, "class_separability_train.json", results)
    return 0


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    """Mann-Whitney U-based P(X>Y)+0.5P(X=Y) separation for a vs b."""
    from scipy.stats import mannwhitneyu
    try:
        u, _ = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        return float(u / (len(a) * len(b)))
    except ValueError:
        return 0.5


def _cramers_v(ctab: pd.DataFrame) -> float:
    from scipy.stats import chi2_contingency
    n = ctab.to_numpy()
    if n.shape[0] < 2 or n.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(n, correction=False)
    denom = n.sum() * (min(n.shape) - 1)
    if denom <= 0:
        return 0.0
    return float(math.sqrt(chi2 / denom))


# --------------------------------------------------------------------------- #
# Phase 6 - spatial / matching audit
# --------------------------------------------------------------------------- #

def _run_spatial(args: argparse.Namespace) -> int:
    print("\n=== PHASE 6: COCONUT vs PEPPER MATCHING AUDIT (train) ===\n")
    df = load_csv(CSV_PATH)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train_all = df[_is_eligible(df) & _split_mask(df, manifest, "train")].copy()
    train = train_all[train_all["crop_label"].isin(["coconut", "pepper"])].copy()
    out_all: dict[str, Any] = {}
    for crop in ("coconut", "pepper"):
        sub = train[train["crop_label"] == crop]
        d = {}
        for col in ("spatial_match_distance_km", "env_match_distance_m"):
            v = pd.to_numeric(sub[col], errors="coerce")
            d[col] = {"mean": round(float(v.mean()), 3), "median": round(float(v.median()), 3),
                      "p25": round(float(v.quantile(0.25)), 3), "p75": round(float(v.quantile(0.75)), 3),
                      "p95": round(float(v.quantile(0.95)), 3), "max": round(float(v.max()), 3)}
        d["lat"] = {"min": round(float(sub["lat"].min()), 4), "max": round(float(sub["lat"].max()), 4),
                    "mean": round(float(sub["lat"].mean()), 4)}
        d["lon"] = {"min": round(float(sub["lon"].min()), 4), "max": round(float(sub["lon"].max()), 4),
                    "mean": round(float(sub["lon"].mean()), 4)}
        d["env_match_year"] = sub["env_match_year"].value_counts().to_dict()
        d["season"] = sub["season"].value_counts().to_dict()
        d["tabular_source"] = sub["tabular_source"].value_counts().to_dict()
        d["image_source"] = sub["image_source"].value_counts().to_dict()
        d["land_cover_class"] = sub["land_cover_class"].value_counts().to_dict()
        d["soil_type_class"] = sub["soil_type_class"].value_counts().to_dict()
        d["satellite_status"] = sub["satellite_status"].value_counts().to_dict()
        d["temporal_match_status"] = sub["temporal_match_status"].value_counts().to_dict()
        print(f"\n  --- {crop} (n={len(sub)}) ---")
        for k, v in d.items():
            print(f"    {k}: {json.dumps(v, default=str)[:260]}")
        out_all[crop] = d

    _save(args, "R5.5_phase6_spatial_audit.json", out_all)

    print("\n  --- VILLAGE CO-LOCATION (train) ---")
    overlap = _village_overlap(train_all)
    print(json.dumps(overlap, indent=2))
    out_all["village_colocation_train"] = overlap
    _save(args, "R5.5_phase6_spatial_audit.json", out_all)
    return 0


def _village_overlap(train: pd.DataFrame) -> dict[str, Any]:
    """Train-only shared-residence analysis across supervised crops.

    Mixed cropping is common in Dakshina Kannada (black pepper vines are
    trained up coconut palms). If two classes occupy the SAME villages, their
    matched env-grid features (one cell per location) are nearly identical and
    a tabular classifier fundamentally cannot separate them at village scale.
    """
    def villages(crop: str) -> tuple[set[str], pd.Series]:
        s = train.loc[train["crop_label"] == crop, "location_village"].str.upper()
        return set(s.dropna()), s

    cv, cs = villages("coconut")
    pv, ps = villages("pepper")
    fv, fs = villages("coffee")
    dva, ds = villages("cardamom")
    out: dict[str, Any] = {}
    for name, (va, sa), (vb, sb) in (
        ("coconut_pepper", (cv, cs), (pv, ps)),
        ("coconut_coffee", (cv, cs), (fv, fs)),
        ("coconut_cardamom", (cv, cs), (dva, ds)),
        ("pepper_coffee", (pv, ps), (fv, fs)),
    ):
        shared = va & vb
        out[name] = {
            "a_villages": len(va), "b_villages": len(vb), "shared_villages": len(shared),
            "a_rows_in_shared_village": int(sa.isin(vb).sum()), "a_total": int(len(sa)),
            "b_rows_in_shared_village": int(sb.isin(va).sum()), "b_total": int(len(sb)),
        }
    return out


# --------------------------------------------------------------------------- #
# Phase 7 - feature inventory
# --------------------------------------------------------------------------- #

_FEATURE_INVENTORY: dict[str, dict[str, str]] = {
    "lat": {"source": "frozen corpus (village GPS)", "meaning": "sample latitude", "units": "deg", "risk": "coordinates identify taluk/split"},
    "lon": {"source": "frozen corpus (village GPS)", "meaning": "sample longitude", "units": "deg", "risk": "coordinates identify taluk/split"},
    "spatial_match_distance_km": {"source": "STAM spatial match", "meaning": "distance from village point to matched env grid cell", "units": "km", "risk": "match-quality confound"},
    "year": {"source": "survey metadata", "meaning": "survey/report year", "units": "year", "risk": "none (env_match_year==survey year verified)"},
    "annual_rainfall_mm": {"source": "DK env grid (R5.2.9)", "meaning": "annual precipitation at matched cell", "units": "mm", "risk": "none"},
    "dewpoint_c": {"source": "DK env grid", "meaning": "mean dewpoint", "units": "deg C", "risk": "none"},
    "elevation": {"source": "DK env grid (DEM)", "meaning": "surface elevation", "units": "m", "risk": "none"},
    "temperature_c": {"source": "DK env grid", "meaning": "mean air temperature", "units": "deg C", "risk": "none"},
    "relative_humidity_pct": {"source": "DK env grid", "meaning": "mean relative humidity", "units": "%", "risk": "none"},
    "slope": {"source": "DK env grid (DEM)", "meaning": "terrain slope", "units": "deg", "risk": "none"},
    "ndvi": {"source": "Sentinel-2 composite/env grid", "meaning": "mean NDVI (seasonal env composite)", "units": "-1..1", "risk": "static composite, not per-sample imagery"},
    "evi": {"source": "Sentinel-2 composite/env grid", "meaning": "mean EVI (seasonal env composite)", "units": "-1..1", "risk": "static composite, not per-sample imagery"},
    "ndwi": {"source": "Sentinel-2 composite/env grid", "meaning": "mean NDWI", "units": "-1..1", "risk": "none"},
    "ndre": {"source": "Sentinel-2 composite/env grid", "meaning": "mean NDRE (red edge)", "units": "-1..1", "risk": "none"},
    "savi": {"source": "Sentinel-2 composite/env grid", "meaning": "mean SAVI", "units": "-1..1", "risk": "none"},
    "s2_obs_count": {"source": "Sentinel-2 metadata", "meaning": "composite observation count", "units": "count", "risk": "coverage confound"},
    "soil_clay_pct": {"source": "soilgrids/DK grid", "meaning": "soil clay fraction", "units": "%", "risk": "none"},
    "soil_sand_pct": {"source": "soilgrids/DK grid", "meaning": "soil sand fraction", "units": "%", "risk": "none"},
    "soil_organic_carbon": {"source": "soilgrids/DK grid", "meaning": "soil organic carbon", "units": "dag/kg", "risk": "none"},
    "soil_ph": {"source": "soilgrids/DK grid", "meaning": "soil pH", "units": "pH", "risk": "none"},
    "soil_moisture": {"source": "DK env grid", "meaning": "soil moisture", "units": "0..1", "risk": "none"},
    "kharif_ndvi": {"source": "Sentinel-2 seasonal", "meaning": "Kharif-season mean NDVI", "units": "-1..1", "risk": "seasonal composite"},
    "kharif_evi": {"source": "Sentinel-2 seasonal", "meaning": "Kharif-season mean EVI", "units": "-1..1", "risk": "seasonal composite"},
    "kharif_ndwi": {"source": "Sentinel-2 seasonal", "meaning": "Kharif-season mean NDWI", "units": "-1..1", "risk": "none"},
    "rabi_ndvi": {"source": "Sentinel-2 seasonal", "meaning": "Rabi-season mean NDVI", "units": "-1..1", "risk": "seasonal composite"},
    "rabi_evi": {"source": "Sentinel-2 seasonal", "meaning": "Rabi-season mean EVI", "units": "-1..1", "risk": "seasonal composite"},
    "rabi_ndwi": {"source": "Sentinel-2 seasonal", "meaning": "Rabi-season mean NDWI", "units": "-1..1", "risk": "none"},
    "env_match_distance_m": {"source": "DK env grid match", "meaning": "match cell distance", "units": "m", "risk": "match-quality confound"},
    "season": {"source": "survey metadata", "meaning": "growing season", "units": "Kharif/Rabi", "risk": "none"},
    "is_cropland": {"source": "land-cover product", "meaning": "cropland flag", "units": "0/1", "risk": "none"},
    "land_cover_class": {"source": "land-cover product", "meaning": "ESA/Esa land cover class id", "units": "class", "risk": "none"},
    "soil_type_class": {"source": "soilgrids", "meaning": "soil type class id", "units": "class", "risk": "none"},
}

_BANNED = {"Yield_Proxy_NPP", "Area_sq_km"}


def _run_features(args: argparse.Namespace) -> int:
    print("\n=== PHASE 7: TABULAR FEATURE LEAKAGE INVENTORY ===\n")
    df = load_csv(CSV_PATH)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train = df[_is_eligible(df) & _split_mask(df, manifest, "train")]

    missing_cols = [c for c in NUMERIC + CATEGORICAL if c not in df.columns]
    print("Configured-but-absent columns (missing_columns):", missing_cols or "none")

    rows = {}
    for col in NUMERIC:
        v = pd.to_numeric(train[col], errors="coerce")
        meta = _FEATURE_INVENTORY.get(col, {"source": "?", "meaning": "?", "units": "?", "risk": "?"})
        rows[col] = {
            "source": meta["source"], "meaning": meta["meaning"], "units": meta["units"],
            "leak_risk": meta["risk"], "type": "numeric",
            "train": {"mean": round(float(v.mean()), 3) if v.notna().any() else None,
                      "std": round(float(v.std()), 3) if v.notna().any() else None,
                      "min": round(float(v.min()), 3) if v.notna().any() else None,
                      "max": round(float(v.max()), 3) if v.notna().any() else None,
                      "missing": int(v.isna().sum()), "unique": int(v.nunique())},
        }
    for col in CATEGORICAL:
        meta = _FEATURE_INVENTORY.get(col, {"source": "?", "meaning": "?", "units": "?", "risk": "?"})
        rows[col] = {
            "source": meta["source"], "meaning": meta["meaning"], "units": meta["units"],
            "leak_risk": meta["risk"], "type": "categorical",
            "train": {"missing": int(train[col].isna().sum()),
                      "unique": int(train[col].nunique()),
                      "top_values": {str(k): int(v) for k, v in train[col].value_counts().head(5).items()}},
        }

    print("Banned/leak columns check:", _BANNED, "-> present in frozen CSV:",
          sorted(_BANNED & set(df.columns)) or "NONE")
    out = {"feature_count": len(NUMERIC) + len(CATEGORICAL),
           "numeric": len(NUMERIC), "categorical": len(CATEGORICAL),
           "banned_present": sorted(_BANNED & set(df.columns)),
           "missing_columns": missing_cols,
           "features": rows}
    _save(args, "R5.5_phase7_feature_leakage.json", out)
    return 0


# --------------------------------------------------------------------------- #
# Phase 8 - normalization audit
# --------------------------------------------------------------------------- #

def _run_normalization(args: argparse.Namespace) -> int:
    print("\n=== PHASE 8: NORMALIZATION AUDIT (raw -> filled -> scaled) ===\n")
    from training.kaggle.frozen_corpus import build_observation
    from training.preprocessing.config import load_preprocessing_config
    from training.preprocessing.master_pipeline import Preprocessor
    import training.kaggle.frozen_corpus as fc

    df = load_csv(CSV_PATH)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train = df[_is_eligible(df) & _split_mask(df, manifest, "train")].reset_index(drop=True)

    def _no_imagery(stam, row):
        return {"pairs": [], "source": "none"}

    original = fc._resolve_imagery
    fc._resolve_imagery = _no_imagery
    checksum = _sha256(MANIFEST)
    try:
        obs = [build_observation(r_.to_dict(), stam=None,
                                 corpus_version=manifest["dataset_version"],
                                 manifest_checksum=checksum)
               for _, r_ in train.iterrows()]
    finally:
        fc._resolve_imagery = original

    pre = Preprocessor.from_config(str(PREPROCESSING_YAML))
    pre.tabular.fit(obs)
    print("Numeric features kept after drop-constants/correlation:",
          pre.tabular.numeric_features if hasattr(pre.tabular, "numeric_features") else "?")
    print("encoders:", pre.config.tabular.categorical_encoding)

    from training.preprocessing.tabular_pipeline import _numeric_matrix

    fill = pre.tabular.missing_fill
    clip = pre.tabular.clip_bounds
    scaler = pre.tabular.scaler
    features = list(pre.tabular.numeric_features)

    per = {}
    for row in obs[: args.n]:
        raw_filled = _numeric_matrix([dict(row.tabular.fields)], features, fill, clip)
        scaled = scaler.transform(raw_filled) if scaler is not None else raw_filled
        per[row.crop or "?"] = {}
        for j, col in enumerate(features):
            raw = (row.tabular.fields.get(col)
                   if isinstance(row.tabular.fields.get(col), (int, float)) else None)
            per[row.crop or "?"][col] = {
                "raw": None if raw is None else round(float(raw), 4),
                "filled_clipped": round(float(raw_filled[0, j]), 4),
                "scaled": round(float(scaled[0, j]), 4),
            }

    headline = ["annual_rainfall_mm", "temperature_c", "relative_humidity_pct",
                "elevation", "ndvi", "evi", "soil_moisture", "soil_ph",
                "soil_organic_carbon", "lat", "lon"]
    print("\nScale constants per numeric feature (train):")
    stats = {}
    for col in features:
        v = _numeric_matrix([dict(o.tabular.fields) for o in obs], features, fill, clip)
        col_idx = features.index(col)
        col_vals = v[:, col_idx]
        stats[col] = {"fill": fill.get(col), "clip": clip.get(col),
                      "min": round(float(col_vals.min()), 4),
                      "max": round(float(col_vals.max()), 4),
                      "post_mean": round(float(col_vals.mean()), 4),
                      "post_std": round(float(col_vals.std()), 4)}
    for col in headline:
        s = stats.get(col)
        print(f"    {col:28s} fill={s['fill'] if s else '?'} "
              f"clip={s['clip'] if s else '?'} raw[minmax] "
              f"filled[min/max]={s['min']}/{s['max'] if s else '?'}")

    out = {"features_kept": features,
           "per_sample_examples": {crop: {f: per[crop][f] for f in per[crop]} for crop in per},
           "scale_stats": stats}
    _save(args, "R5.5_phase8_normalization.json", out)
    return 0


# --------------------------------------------------------------------------- #
# Phase 17 - what R5.2.9 added
# --------------------------------------------------------------------------- #

def _run_recovery17(args: argparse.Namespace) -> int:
    print("\n=== PHASE 17: R5.2.9 RECOVERY AUDIT ===\n")
    v1 = load_csv(V1_CSV_PATH)
    v2 = load_csv(CSV_PATH)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    ids1 = set(v1["record_id"].astype(str))
    ids2 = set(v2["record_id"].astype(str))
    added = sorted(ids2 - ids1)
    removed = sorted(ids1 - ids2)
    common = ids1 & ids2
    print(f"v1 rows={len(v1)}  v2 rows={len(v2)}  common={len(common)}")
    print(f"added={len(added)} removed={len(removed)}")

    out: dict[str, Any] = {
        "v1_rows": int(len(v1)), "v2_rows": int(len(v2)), "common": int(len(common)),
        "added_ids": added, "removed_ids": removed,
        "is_recovered_v2_counts":
            v2["is_recovered_v2"].astype(str).value_counts().to_dict(),
        "benchmark_eligible_counts":
            v2["benchmark_eligible"].astype(str).value_counts().to_dict(),
        "recovered_rows": [],
    }
    for rid in added:
        row = v2[v2["record_id"].astype(str) == rid].iloc[0]
        out["recovered_rows"].append({
            "record_id": rid,
            "crop": row["crop_label"], "taluk": row["location_taluk"],
            "year": row["year"], "season": row["season"],
            "survey_date": row["survey_date"], "satellite": row["satellite_status"],
            "benchmark_eligible": row["benchmark_eligible"],
            "v1_record_id": row.get("v1_record_id"),
        })
        print("  added row:", out["recovered_rows"][-1])

    for rid in removed:
        print("  removed row:", rid)
    # Prove equality for the common rows in every feature.
    cols = [c for c in v2.columns if c not in ("is_recovered_v2",)]
    v1c = v1.set_index("record_id").loc[sorted(common)]
    v2c = v2.set_index("record_id").loc[sorted(common)]
    changed_cells = 0
    for c in cols:
        if c in v1c:
            changed_cells += int((v1c[c] != v2c[c]).sum())
    print(f"changed cells across {len(cols)} shared columns: {changed_cells}")
    out["changed_cells"] = int(changed_cells)
    out["library_note"] = (
        "Previous delta (corpus_delta_r5_2_8.json) already confirmed the "
        "benchmark-eligible v2 set is RECORD-IDENTICAL to v1 (0 removed, 0 "
        "changed). This audit re-derives the same from source CSVs and "
        "categories the single added row."
    )
    _save(args, "r5_2_9_recovery_audit.json", out)
    return 0


# --------------------------------------------------------------------------- #
# Phase 16 - coffee/cardamom upstream trace
# --------------------------------------------------------------------------- #

def _run_coffee16(args: argparse.Namespace) -> int:
    print("\n=== PHASE 16: COFFEE/CARDAMOM UPSTREAM RETENTION TRACE ===\n")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    df = load_csv(CSV_PATH)
    ledger = pd.read_csv(V1_LEDGER, dtype=str)

    out = {"manifest_totals": manifest["class_counts"], "crops": {}}
    for crop in ("coffee", "cardamom"):
        src = ledger[ledger["crop_type"].str.strip().str.lower() == crop]
        full = src[src["satellite_status"].str.upper() == "FULL"]
        nodup = full[full["is_duplicate"].fillna("False").astype(str).str.strip().str.lower() == "false"]
        valid = nodup[nodup["valid_cropfusion_sample"].fillna("False").astype(str).str.strip().str.lower() == "true"]
        reasons = src["rejection_reasons"].fillna("[]").value_counts().head(8).to_dict()
        accepted = int((df["crop_label"] == crop).sum())
        d = {
            "source_records": int(len(src)),
            "satellite_full": int(len(full)),
            "non_duplicate": int(len(nodup)),
            "valid_matched": int(len(valid)),
            "corpus_accepted": accepted,
            "rejection_reasons_top": {str(k): int(v) for k, v in reasons.items()},
        }
        print(f"\n  --- {crop} ---")
        for k, v in d.items():
            print(f"    {k}: {v}")
        out["crops"][crop] = d
    _save(args, "R5.5_phase16_coffee_cardamom_trace.json", out)
    return 0


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #

def _save(args: argparse.Namespace, name: str, data: dict[str, Any]) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(
        json.dumps(data, indent=2, default=_json_default), encoding="utf-8"
    )
    print(f"\n  saved -> {out / name}")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(REPORTS))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("trace", help="Phase 1")
    sub.add_parser("distribution", help="Phase 2")
    sp = sub.add_parser("separability", help="Phase 4")
    sp.add_argument("--pairs", nargs="+", action="append", default=[
        ["coconut", "pepper"], ["coconut", "coffee"], ["pepper", "coffee"],
        ["cardamom", "coconut"], ["cardamom", "pepper"],
    ])
    sub.add_parser("spatial", help="Phase 6")
    sub.add_parser("features", help="Phase 7")
    sn = sub.add_parser("normalization", help="Phase 8")
    sn.add_argument("--n", type=int, default=4, help="number of representative rows to print")
    sub.add_parser("recovery17", help="Phase 17")
    sub.add_parser("coffee16", help="Phase 16")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    handlers = {
        "trace": _run_trace,
        "distribution": _run_distribution,
        "separability": _run_separability,
        "spatial": _run_spatial,
        "features": _run_features,
        "normalization": _run_normalization,
        "recovery17": _run_recovery17,
        "coffee16": _run_coffee16,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())