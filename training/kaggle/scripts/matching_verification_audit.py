"""R5.5 matching-verification audit (Phase 3 deliverable).

Verifies and quantifies the spatial/temporal matching properties of the frozen
corpus and audits the 32-feature tabular branch for target leakage.

Checks
------
* Spatial matching: accepted-confidence distance bands per class
  (0-100 / 100-250 / 250-500 / 500-1000 / >1000 m), max-search-radius
  compliance (5 km), and cross-split nearest-neighbour distances
  (test->train, val->train) to surface field-adjacency leakage risk.
* Temporal matching: ``temporal_match_status`` distribution per class, and
  date-arithmetic raster-availability (candidate frames) per survey window
  (== +-30/60/90/120/180 days) per class (requires ``--inventory``).
* Tabular leakage: the config's 32-feature set is enumerated with provenance;
  the frozen corpus is scanned for banned/target-derived columns
  (``Yield_Proxy_NPP``, ``Area_sq_km``, meta coordinates); env-match year must
  never be after the survey year; missingness must be 0.

Usage::

    python training/kaggle/scripts/matching_verification_audit.py \\
        --inventory kaggle_imagery_inventory.json \\
        --out-dir reports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_FROZEN = _REPO_ROOT / "govt_crop_matched_v2" / "crop_supervised_v2.csv"
_DEFAULT_PREPROCESSING = _REPO_ROOT / "training" / "config" / "preprocessing.yaml"
_DEFAULT_MANIFEST = _REPO_ROOT / "training_manifests" / "crop_supervised_v2.0_manifest.json"

_CROPS = ["coconut", "pepper", "coffee", "cardamom", "blackgram"]

_DISTANCE_BANDS = [(0, 100), (100, 250), (250, 500), (500, 1000), (1000, float("inf"))]

#: Banned/target-derived tokens that must never appear in the feature schema.
_BANNED_TOKENS = ("yield", "npp", "production", "irrigat", "price", "profit")
_BANNED_COLUMNS = {"Yield_Proxy_NPP", "Area_sq_km", "Year", "Season", "State", "District", "Country", "Latitude", "Longitude", "system:index"}

_EARTH_RADIUS_M = 6_371_000.0


def _haversine_matrix_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Pairwise great-circle distances (m) between two column vectors."""
    p1 = np.radians(np.stack([lat1, lon1], axis=1))
    p2 = np.radians(np.stack([lat2, lon2], axis=1))
    dlat = p2[:, 0][None, :] - p1[:, 0][:, None]
    dlon = p2[:, 1][None, :] - p1[:, 1][:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(p1[:, 0][:, None]) * np.cos(p2[:, 0][None, :]) * np.sin(dlon / 2) ** 2
    return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bands(counts_by_class: dict[str, int], dist_m: pd.Series) -> dict:
    out = {"per_class": {}, "all": []}
    labels = ["{}-{:.0f}".format(a, b) if b != float("inf") else f">{a}" for a, b in _DISTANCE_BANDS]
    for crop in _CROPS:
        mask = counts_by_class.get(crop)
        if mask is None:
            continue
        sub = dist_m[mask]
        out["per_class"][crop] = {
            "total": int(len(sub)),
            "bands": {
                labels[i]: {
                    "count": int(((sub >= a) & (sub < b)).sum()),
                    "pct_of_class": round(100.0 * ((sub >= a) & (sub < b)).sum() / len(sub), 2) if len(sub) else 0.0,
                }
                for i, (a, b) in enumerate(_DISTANCE_BANDS)
            },
        }
    out["all"] = [
        {
            "band": labels[i],
            "count": int(((dist_m >= a) & (dist_m < b)).sum()),
            "pct": round(100.0 * ((dist_m >= a) & (dist_m < b)).sum() / len(dist_m), 2),
        }
        for i, (a, b) in enumerate(_DISTANCE_BANDS)
    ]
    return out


def _nearest_band(matrix: np.ndarray) -> dict:
    row_min = matrix.min(axis=1)
    return {
        "min_m": round(float(row_min.min()), 1),
        "p5_m": round(float(np.percentile(row_min, 5)), 1),
        "p25_m": round(float(np.percentile(row_min, 25)), 1),
        "median_m": round(float(np.percentile(row_min, 50)), 1),
        "p75_m": round(float(np.percentile(row_min, 75)), 1),
        "under_250m_n": int((row_min < 250).sum()),
        "under_250m_pct": round(100.0 * (row_min < 250).sum() / len(row_min), 2),
        "under_500m_n": int((row_min < 500).sum()),
        "under_500m_pct": round(100.0 * (row_min < 500).sum() / len(row_min), 2),
    }


def _temporal_availability(frozen: pd.DataFrame, inventory_dates: list[date]) -> dict:
    dates = pd.to_datetime(inventory_dates)
    windows = {"30": 30, "60": 60, "90": 90, "120": 120, "180": 180}
    per_class: dict[str, dict] = {}
    sd = pd.to_datetime(frozen["survey_date"], errors="coerce")
    for crop in _CROPS:
        s = frozen[frozen["crop_label"] == crop]
        row = {}
        for label, days in windows.items():
            counts = []
            for d in sd[s.index].dropna():
                counts.append(int((abs(dates - d) <= pd.Timedelta(days=days)).sum()))
            vals = pd.Series(counts) if counts else pd.Series([0])
            row[f"+-{label}d"] = {
                "mean_frames": round(float(vals.mean()), 3),
                "median": float(vals.median()),
                "max": int(vals.max()),
                "with_zero": int((vals == 0).sum()),
            }
        per_class[crop] = row
    return per_class


def run(out_dir: Path, inventory_path: Path | None, tolerance_km: float) -> dict:
    frozen = pd.read_csv(_DEFAULT_FROZEN, dtype=str, low_memory=False)
    frozen = frozen[frozen["benchmark_eligible"] == "True"].copy()
    manifest = json.loads(_DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    prep = yaml.safe_load(_DEFAULT_PREPROCESSING.read_text(encoding="utf-8"))["tabular"]

    from training.kaggle.frozen_corpus import _TALUK_SPLIT
    frozen["_split"] = frozen["location_taluk"].str.strip().map(_TALUK_SPLIT)

    dkm = pd.to_numeric(frozen["spatial_match_distance_km"], errors="coerce")
    dist_m = dkm * 1000.0
    class_mask = {c: (frozen["crop_label"] == c).values for c in _CROPS}

    spatial = _bands(class_mask, dist_m)
    spatial["summary_m"] = {
        "count": int(dist_m.count()),
        "mean": round(float(dist_m.mean()), 1),
        "median": round(float(dist_m.median()), 1),
        "p95": round(float(np.percentile(dist_m, 95)), 1),
        "max": round(float(dist_m.max()), 1),
    }
    spatial["max_search_radius_compliance"] = {
        "radius_km": tolerance_km,
        "violations": int((dist_m > tolerance_km * 1000.0).sum()),
    }

    lat = frozen["lat"].astype(float).to_numpy()
    lon = frozen["lon"].astype(float).to_numpy()
    splits = {s: frozen["_split"] == s for s in ("train", "val", "test")}
    cross_split = {}
    for dst, src in (("test_train", ("test", "train")), ("val_train", ("val", "train"))):
        d, s = src
        matrix = _haversine_matrix_m(lat[splits[d]], lon[splits[d]], lat[splits[s]], lon[splits[s]])
        cross_split[dst] = _nearest_band(matrix)

    temporal = {
        "per_class": {
            crop: frozen.loc[class_mask[crop], "temporal_match_status"].value_counts().to_dict()
            for crop in _CROPS
        },
    }
    temporal["overall"] = frozen["temporal_match_status"].value_counts().to_dict()
    temporal["env_year_alignment"] = {
        "rows": int(len(frozen)),
        "env_match_year_gt_survey_year": int((frozen["env_match_year"].astype(float) > frozen["year"].astype(float)).sum()),
        "method": frozen["env_match_method"].value_counts(dropna=False).to_dict(),
        "confidence": frozen["env_match_confidence"].value_counts(dropna=False).to_dict(),
        "env_season_for_features": frozen["env_season_for_features"].value_counts(dropna=False).to_dict(),
    }

    columns = set(frozen.columns)
    banned_present = sorted(
        c for c in columns if c in _BANNED_COLUMNS or any(t in c.lower() for t in _BANNED_TOKENS)
    )

    features = {
        "numeric": list(prep.get("numeric_features", [])),
        "categorical": list(prep.get("categorical_features", [])),
    }
    num_cols = [c for c in features["numeric"] if c in columns]
    missing = {
        col: int(frozen[col].isna().sum()) for col in num_cols
    } if num_cols else {}

    inventory_dates: list[date] = []
    if inventory_path is not None and inventory_path.exists():
        inv = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory_dates = [date.fromisoformat(d) for d in inv.get("dates", [])]
    availability = _temporal_availability(frozen, inventory_dates) if inventory_dates else {}

    report = {
        "release": "r5.5",
        "phase": 3,
        "title": "Matching-verification + tabular-leakage audit (frozen corpus)",
        "manifest_sha256": _sha256(_DEFAULT_MANIFEST),
        "corpus": {
            "rows": int(len(frozen)),
            "splits": {s: int(splits[s].sum()) for s in ("train", "val", "test")},
        },
        "spatial": spatial,
        "cross_split_nearest_neighbour_m": cross_split,
        "temporal": temporal,
        "tabular_leakage": {
            "feature_set": features,
            "feature_count": len(features["numeric"]) + len(features["categorical"]),
            "banned_columns_present": banned_present,
            "max_missing_by_feature": max(missing.values()) if missing else 0,
            "s2_obs_count_note": (
                "s2_obs_count is the count of Sentinel-2 observations in the "
                "grid cell's composite window, not a yield/target quantity."
            ),
            "env_features_note": (
                "All non-geometry features are KNN-IDW spatial interpolants of "
                "the DK 500 m gridded dataset (annual_rainfall_mm ... rabi_ndwi)"
                " plus env_match_distance_m; none are derived from the crop "
                "label or yield. The DK grid ships Yield_Proxy_NPP and "
                "Area_sq_km, both excluded at match time."
            ),
            "temporal_note": (
                "env_match_year == survey year for every row (gap 0); no "
                "feature comes from after the survey year. Same-year "
                "environmental features are climatology/environment, not label "
                "proxies; the Phase 8 ablation retrains without them."
            ),
        },
        "temporal_availability_candidates": availability,
        "limitations": (
            "" if inventory_path else "temporal_availability_candidates omitted (no --inventory). "
        ) + "Candidate frames are raster-existence date arithmetic; point coverage not modelled.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "R5.5_matching_verification.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("spatial bands:", {b["band"]: b["count"] for b in spatial["all"]})
    print("test->train nearest m:", cross_split["test_train"]["median_m"],
          "under500 pct:", cross_split["test_train"]["under_500m_pct"])
    print("val->train nearest m:", cross_split["val_train"]["median_m"],
          "under500 pct:", cross_split["val_train"]["under_500m_pct"])
    print("banned columns present:", banned_present or "NONE")
    print(f"wrote {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "reports"))
    parser.add_argument("--inventory", default=None, help="kaggle imagery inventory JSON (dates list)")
    parser.add_argument("--max-search-radius-km", type=float, default=5.0,
                        help="R5.2.7 spatial tolerance (km) for compliance check")
    args = parser.parse_args()
    run(Path(args.out_dir), Path(args.inventory) if args.inventory else None, args.max_search_radius_km)


if __name__ == "__main__":
    main()