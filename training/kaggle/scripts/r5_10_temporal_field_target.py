"""R5.10 — Temporal field-target recoverability experiment.

Questions whether the coconut-vs-pepper field-composition classification
problem becomes recoverable when the satellite/environmental signal is
represented TEMPORALLY (multiple grid years) rather than as the single
survey-season observation used in R5.9.

Context
  * R5.9 (commit 6a83f94a) built a field-composition dominant-crop target on
    339,557 physical fields (coconut-dominant 56,863; pepper-dominant 797)
    and found NO recoverable signal with a single-season/static satellite +
    environmental representation:
        best test balanced accuracy = 50.42%  (gb)
        best test ROC AUC          = 0.4781
        verdict                    = no_signal
  * R5.10 re-runs the SAME target and SAME grouped taluk split, but replaces /
    extends the feature representation with per-field TEMPORAL satellite
    composites derived from the authoritative DK_Features_YYYY grids
    (2018-2023), which are the legitimate remotely-sensed per-year vegetation
    and environmental composites already used by R5.7/R5.9.

Scientific constraints implemented here
  1. Crop_Extent is NEVER a feature. It is used ONLY to construct the R5.9
     field-composition target, which we consume EXACTLY as R5.9 emitted it
     (reports/R5.9/field_dataset_split.csv).
  2. CROP_EXTENT_UNIT_STATUS = UNKNOWN (preserved from R5.9).
  3. No fabricated geometry. Field locations are the real survey GPS.
  4. No fabricated satellite observations. The per-year DK grid composites
     are real; a grid-year is only a temporal slot if a grid exists for it and
     the field matches a cell within 5 km.
  5. No synthetic temporal observations.
  6. No target-derived features.
  7. Forecasting direction is defined (prediction date = the composition
     survey year). Primary representation uses ONLY grid years <= survey year
     ("leading window"). Years after the survey year are never used in the
     primary experiments (an explicit full-window 2018-2023 sensitivity is
     reported separately in the robustness phase and never called primary).
  8. Physical field never spans two splits (taluk-grouped, inherited from R5.9).
  9. No random row-level splitting.
  10. No test-set optimization, no threshold tuning after seeing test results.
  11. No full CropFusion training (cheap baselines only).
  12. Complete provenance contract emitted.

Run from repo root::

    python training/kaggle/scripts/r5_10_temporal_field_target.py
    python training/kaggle/scripts/r5_10_temporal_field_target.py --phases 4  # extract only
    python training/kaggle/scripts/r5_10_temporal_field_target.py --limit-fields 2000  # dev guard

Output block (reports/R5.10)
  STATUS / BEST_TEST_BALANCED_ACCURACY / R5.9_BALANCED_ACCURACY /
  DELTA_VS_R5_9 / BEST_TEST_AUC / R5.9_TEST_AUC / BEST_MODEL /
  SIGNAL_DECISION / SPATIAL_SPLIT_STATUS / TEMPORAL_ALIGNMENT_STATUS /
  CROP_EXTENT_UNIT_STATUS / CROPFUSION_JUSTIFIED / RECOMMENDATION
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "reports" / "R5.10"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.matching.spatial_tabular_matcher import (  # noqa: E402
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_CONTINUOUS_FEATURES,
    SpatialTabularMatcher,
)
from training.kaggle.scripts.r5_7_data_recovery import (  # noqa: E402
    ENV_CATEGORICAL,
    ENV_CONTINUOUS,
    R56_CATEGORICAL,
    R56_NUMERIC,
    SEED,
    TALUK_SPLIT,
    to_py,
)
from training.kaggle.scripts.r5_8_subfield import (  # noqa: E402
    SURVEY_DIR,
    load_survey_rows,
)
from training.kaggle.scripts.r5_9_field_composition import (  # noqa: E402
    BINARY,
    ENV_NUMERIC,
    OUT_DIR as R59_OUT_DIR,
)

MASTER_CSV = REPO_ROOT / "reports" / "R5.7" / "master_geospatial_features.csv"
R59_DATASET = R59_OUT_DIR / "field_dataset_split.csv"
R59_REPORT = R59_OUT_DIR / "R5.9_FIELD_COMPOSITION_REPORT.json"
DK_DIR = REPO_ROOT / "Tabular_Datasets"

# Authoritative govt crop survey files (identical list to R5.9's SURVEY_FILES).
SURVEY_FILES = [
    "ogd_bantvala_kharif_2020_21.csv",
    "ogd_beltangadi_kharif_2020_21.csv",
    "ogd_kokkada_kharif_2020_21.csv",
    "ogd_mangaluru_a_kharif_2020_2021.csv",
    "ogd_mangaluru_b_kharif_2020_21.csv",
    "ogd_mulki_kharif_2021_22.csv",
    "ogd_panemangaluru_rabi_2021_2022.csv",
    "ogd_panja_kharif_2020_21.csv",
    "ogd_putturu_kharif_2020_21.csv",
    "ogd_sulya_kharif_2020_21.csv",
    "ogd_suratkal_kharif_2020_21.csv",
    "ogd_uppinangadi_kharif_2021_22.csv",
    "ogd_venuru_kharif_2020_21.csv",
    "ogd_venuru_rabi_2021_22.csv",
    "ogd_vitla_kharif_2020_21.csv",
]

# ---------------------------------------------------------------------------
# Frozen experiment constants (documented BEFORE any result is interpreted).
# ---------------------------------------------------------------------------
SEED = 42

# Baseline target / values from R5.9 (frozen reference for the delta).
R5_9_BALANCED_ACCURACY = 0.5042
R5_9_ROC_AUC = 0.4781
R5_9_BEST_MODEL = "gb"

# The legitimate per-year satellite/environmental composites.
GRID_YEARS_ALL = [2018, 2019, 2020, 2021, 2022, 2023]
SURVEY_YEARS = [2020, 2021]

# Primary temporal policy: no future information relative to the survey year.
def leading_years(survey_year: int) -> list[int]:
    return [y for y in GRID_YEARS_ALL if y <= int(survey_year)]

# Temporally-varying emitted grid variables (lowercase emission names).
TEMPORAL_VARS = [
    "annual_rainfall_mm", "dewpoint_c", "temperature_c",
    "relative_humidity_pct",
    "evi", "ndvi", "ndwi", "ndre", "savi", "s2_obs_count",
    "kharif_ndvi", "kharif_evi", "kharif_ndwi",
    "rabi_ndvi", "rabi_evi", "rabi_ndwi",
]
# Static grid variables (constant across years, belong to the static set).
STATIC_NUMERIC = [
    "elevation", "slope", "soil_clay_pct", "soil_sand_pct",
    "soil_organic_carbon", "soil_ph", "soil_moisture",
]
STATIC_CATEGORICAL = ["is_cropland", "land_cover_class", "soil_type_class"]

# Deduplicate coordinate keys for temporal extraction.
def coord_key(lat: float, lon: float) -> str:
    return f"{float(lat):.7f}|{float(lon):.7f}"

# Signal decision gates (defined BEFORE running models).
GATE_NO_SIGNAL_HI = 0.55
GATE_WEAK_HI = 0.60
GATE_PROMISING_HI = 0.70

PHASES = {
    "0": "source_schema",
    "1": "temporal_unit",
    "2": "extract_grid",
    "3": "temporal_alignment",
    "4": "temporal_features",
    "5": "target",
    "6": "split",
    "7": "cheap_baselines",
    "8": "ablation",
    "9": "shortcuts",
    "10": "robustness",
    "11": "decision",
    "12": "report",
    "13": "provenance",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_report_json(obj: Any, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w", encoding="utf-8") as fh:
        json.dump(to_py(obj), fh, indent=1, default=str)


# ---------------------------------------------------------------------------
# Loaders (single authoritative implementations; no rival field-id/split
# logic is introduced).
# ---------------------------------------------------------------------------
def load_binary_pool() -> pd.DataFrame:
    """R5.9 binary dominant-crop pool with the EXACT R5.9 target + split.

    The R5.9 field_dataset_split.csv already carries: field_id (R5.8 rule),
    year, season, the R5.9 dominant_crop target, the grouped taluk split and
    the R5.9 static environmental features (SET A source).
    """
    ds = pd.read_csv(R59_DATASET)
    ds = ds[ds["dominant_crop"].isin(BINARY)].copy()
    ds["target"] = (ds["dominant_crop"] == "pepper").astype(int)
    return ds.reset_index(drop=True)


def load_field_coords() -> pd.DataFrame:
    """field_id -> (lat, lon) from the R5.7 master, using the R5.9 rule."""
    m = pd.read_csv(
        MASTER_CSV,
        usecols=["survey_id", "year", "season", "taluk", "hobli", "village",
                 "latitude", "longitude", "crop_label", "field_cluster_id"],
        low_memory=False,
    )
    m["field_id"] = (m["taluk"].fillna("").astype(str).str.upper() + "|"
                     + m["hobli"].fillna("").astype(str).str.upper() + "|"
                     + m["village"].fillna("").astype(str).str.upper()
                     + "|SURVEYID=" + m["survey_id"].astype(str).str.upper())
    coords = (m.groupby("field_id", as_index=False)
              .agg(lat=("latitude", "mean"), lon=("longitude", "mean"),
                   field_cluster_id=("field_cluster_id", "first")))
    return coords


# ---------------------------------------------------------------------------
# Phase 0 — source & schema audit (mirrors R5.9's source_schema contract)
# ---------------------------------------------------------------------------
def phase0_source_schema() -> dict:
    def _cols(p: Path, n: int = 3) -> list[str]:
        return list(pd.read_csv(p, dtype=str, nrows=n).columns)

    survey_files = []
    nrows_total = 0
    for name in SURVEY_FILES:
        p = SURVEY_DIR / name
        df = pd.read_csv(p, dtype=str, nrows=3)
        survey_files.append({
            "name": name,
            "columns": list(df.columns),
            "sha256": sha(p),
        })
        nrows_total += int(sum(1 for _ in open(p, encoding="utf-8-sig",
                                               errors="replace")) - 1)
    for e in survey_files:
        e["rows_raw_approx"] = None
    survey_files[-1]["rows_raw_approx"] = None

    dk = [{
        "file": f"DK_Features_{y}.csv",
        "exists": (DK_DIR / f"DK_Features_{y}.csv").exists(),
        "sha256": sha(DK_DIR / f"DK_Features_{y}.csv")
        if (DK_DIR / f"DK_Features_{y}.csv").exists() else None,
        "columns": _cols(DK_DIR / f"DK_Features_{y}.csv")
        if (DK_DIR / f"DK_Features_{y}.csv").exists() else None,
    } for y in GRID_YEARS_ALL]

    out: dict[str, Any] = {
        "phase": "0",
        "title": "R5.10 source & schema audit",
        "crop_extent_unit_status": (
            "UNKNOWN - no authoritative parser or unit documentation exists in "
            "the repository; Crop_Extent is used ONLY to construct the R5.9 "
            "field-composition target consumed unchanged, never as a feature"),
        "field_identity_rule": (
            "field_id = (taluk||hobli||village).upper() + '|SURVEYID=' + "
            "survey_id.upper(); crop label NOT part of field_id (R5.8 rule)"),
        "split_rule": "grouped field splits by taluk (TALUK_SPLIT), inherited verbatim from reports/R5.9/field_dataset_split.csv",
        "survey_files": survey_files,
        "survey_rows_total_raw": nrows_total,
        "master_dataset": {"path": str(MASTER_CSV), "exists": MASTER_CSV.exists()},
        "r5_9_dataset": {"path": str(R59_DATASET), "exists": R59_DATASET.exists()},
        "dk_grids": dk,
        "temporal_signal_source": (
            "per-year DK_Features_YYYY grids (annual + Kharif/Rabi seasonal "
            "composites), matched to each field's real survey GPS with "
            "SpatialTabularMatcher (K-NN IDW, radius 5 km, k=5); Yield_Proxy_NPP "
            "excluded by the matcher contract"),
        "grid_year_note": ("DK_Features_2024 (1).csv exists on disk but is NOT "
                           "discoverable by SpatialTabularMatcher (stem parsing); "
                           "all temporal slots use 2018-2023 only"),
    }

    md = [
        "# R5.10 Source & Schema Audit",
        "",
        f"- **Crop_Extent** unit status: `UNKNOWN` (no authoritative parser; "
        "used ONLY as the within-field RELATIVE extent score that built the "
        "R5.9 dominant-crop target).",
        f"- **Field identity**: (taluk||hobli||village).upper() + "
        "`|SURVEYID=` + survey_id.upper() (R5.8 rule).",
        "- **Split**: grouped field splits by taluk, inherited unchanged from "
        "R5.9.",
        "",
        "## Survey files (govt crop survey)",
    ]
    for e in survey_files:
        md.append(f"- `{e['name']}` ({len(e['columns'])} cols, "
                  f"sha256 {e['sha256'][:12]})")
    md.append(f"\nTotal raw survey rows (approx): {nrows_total:,}")
    md.append("\n## R5.9 field dataset (target + split + static env)")
    md.append(f"- `{R59_DATASET.name}` sha256 {sha(R59_DATASET)[:16]}")
    md.append("\n## DK grid temporal composites (2018-2023)")
    for d in dk:
        md.append(f"- `{d['file']}` ({len(d['columns'])} cols "
                  f"if {d['exists']})")
    md.append("\n## Temporal signal source")
    md.append("Each field's real survey GPS is matched to the per-year DK grid "
              "with `SpatialTabularMatcher` (K-NN IDW, radius 5 km, k=5); "
              "`Yield_Proxy_NPP` is excluded by the matcher contract. The 2024 "
              "grid file is not matcher-discoverable, so temporal slots use "
              "2018-2023.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "source_schema.md").write_text("\n".join(md) + "\n",
                                              encoding="utf-8")
    write_report_json(out, "source_schema.json")
    return out


# ---------------------------------------------------------------------------
# Phase 1 — temporal unit
# ---------------------------------------------------------------------------
def phase1_temporal_unit() -> dict:
    pool = load_binary_pool()
    coords = load_field_coords()
    merged = pool.merge(coords, on="field_id", how="left")
    n_missing = int(merged["lat"].isna().sum())
    merged = merged[merged["lat"].notna()].copy()

    merged["leading"] = merged["year"].map(leading_years)
    merged["n_leading_years"] = merged["leading"].map(len)

    # survey-date granularity: fields with >=1 composition observation date
    obs_per_field = merged.groupby("field_id")["year"].nunique()
    # temporal grid-slot granularity
    slots = merged["n_leading_years"]

    audit = merged[["field_id", "year", "season", "taluk", "split",
                    "lat", "lon", "n_leading_years"]].copy()
    audit["leading_years_str"] = merged["leading"].map(
        lambda ys: "|".join(map(str, ys)))
    audit.to_csv(OUT_DIR / "temporal_unit.csv", index=False)

    n_gt = {k: int((slots >= k).sum()) for k in [2, 3, 4, 5]}
    out = {
        "phase": "1",
        "temporal_unit": "physical_field_id + (grid composite year) sequence; "
                         "survey years are the composition prediction dates",
        "unique_fields": int(merged["field_id"].nunique()),
        "unique_survey_dates": sorted(merged["year"].unique().tolist()),
        "unique_grid_years_available": GRID_YEARS_ALL,
        "composition_observations": int(len(merged)),
        "fields_missing_coordinates": n_missing,
        "composition_dates_per_field": {
            "mean": round(float(obs_per_field.mean()), 3),
            "median": int(obs_per_field.median()),
            "min": int(obs_per_field.min()),
            "max": int(obs_per_field.max()),
        },
        "temporal_grid_slots_per_field": {
            "mean": round(float(slots.mean()), 3),
            "median": int(slots.median()),
            "min": int(slots.min()),
            "max": int(slots.max()),
        },
        "fields_with_n_grid_slots": {
            "ge1": int(merged["field_id"].nunique()),
            "ge2": n_gt[2], "ge3": n_gt[3], "ge4": n_gt[4], "ge5": n_gt[5],
        },
        "fields_with_n_grid_slots_definition": (
            "a grid slot = one DK_Features_YYYY composite year in the leading "
            "window (grid years <= survey year); every binary field has >=3 "
            "slots because grids exist for 2018-2020 before every survey"),
        "prediction_date": "composition survey year per field (2020 or 2021)",
        "no_future_information_policy": (
            "primary experiments use grid years <= survey year only"),
    }
    write_report_json(out, "temporal_unit.json")
    return out


# ---------------------------------------------------------------------------
# Phase 2 — grid extraction (heavy; cached)
# ---------------------------------------------------------------------------
def _extract_chunk(records: list[dict]) -> list[dict]:
    matcher = SpatialTabularMatcher(
        DK_DIR, max_search_radius_km=5.0, knn_k=5, idw_power=2.0,
        years=GRID_YEARS_ALL,
        continuous_columns=list(DEFAULT_CONTINUOUS_FEATURES),
        categorical_columns=list(DEFAULT_CATEGORICAL_FEATURES),
    )
    out: list[dict] = []
    for rec in records:
        for gy in GRID_YEARS_ALL:
            r = matcher.match(float(rec["lon"]), float(rec["lat"]),
                              int(gy), str(rec["season"]))
            row: dict[str, Any] = {
                "field_id": rec["field_id"],
                "grid_year": gy,
                "matched": bool(r.matched),
                "method": r.method,
                "nearest_distance_m": r.nearest_distance_m,
                "grid_year_used": r.grid_year,
                "support": r.support,
                "confidence": r.confidence,
            }
            row.update(r.features if isinstance(r.features, dict) else {})
            out.append(row)
    return out


def phase2_extract_grid(limit_fields: int | None = None) -> pd.DataFrame:
    p = OUT_DIR / "temporal_field_grid.csv"
    if p.exists():
        df = pd.read_csv(p, low_memory=False)
        if limit_fields is None:
            print(f"[r5.10] cached temporal_field_grid.csv loaded "
                  f"({len(df)} rows)")
            return df
        pool = _temporal_unit_table(limit_fields)
        ids = set(pool["field_id"])
        df = df[df["field_id"].isin(ids)].copy()
        print(f"[r5.10] cached grid restricted to {len(ids)} fields")
        return df

    pool = _temporal_unit_table(limit_fields)
    records = pool[["field_id", "lat", "lon", "season"]].to_dict("records")
    all_rows: list[dict] = []
    chunk = 20_000
    t0 = time.perf_counter()
    for i in range(0, len(records), chunk):
        all_rows.extend(_extract_chunk(records[i:i + chunk]))
        frac = min(i + chunk, len(records)) / len(records)
        print(f"[r5.10] grid extraction {min(i + chunk, len(records))}/"
              f"{len(records)} in {time.perf_counter() - t0:.0f}s "
              f"(eta { (time.perf_counter() - t0) / max(frac, 1e-9) * (1 - frac) / 60:.1f} min)",
              flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(p, index=False)
    print(f"[r5.10] temporal_field_grid.csv written ({len(df)} rows)")
    return df


def _temporal_unit_table(limit_fields: int | None) -> pd.DataFrame:
    pool = load_binary_pool()
    coords = load_field_coords()
    merged = pool.merge(coords, on="field_id", how="left")
    merged = merged[merged["lat"].notna()].copy()
    if limit_fields is not None:
        rng = np.random.default_rng(SEED)
        ids = rng.choice(sorted(merged["field_id"].unique()),
                         size=min(limit_fields, merged["field_id"].nunique()),
                         replace=False)
        merged = merged[merged["field_id"].isin(ids)].copy()
    return merged


# ---------------------------------------------------------------------------
# Phase 3 — temporal alignment / coverage
# ---------------------------------------------------------------------------
def phase3_temporal_alignment(grid: pd.DataFrame | None = None) -> dict:
    if grid is None:
        grid = pd.read_csv(OUT_DIR / "temporal_field_grid.csv",
                           low_memory=False)
    pool = load_binary_pool()
    coords = load_field_coords()
    unit = pool.merge(coords, on="field_id", how="left").dropna(
        subset=["lat"]).copy()
    unit["leading"] = unit["year"].map(leading_years)

    grid = grid[grid["field_id"].isin(set(unit["field_id"]))].copy()
    grid["survey_year"] = grid["field_id"].map(
        unit.set_index("field_id")["year"])
    grid["in_leading"] = grid["grid_year"] <= grid["survey_year"]

    cov = grid.groupby(["grid_year"])["matched"].agg(["mean", "count"])
    leading = grid[grid["in_leading"]]
    per_field = leading.groupby("field_id").agg(
        n_slots=("grid_year", "count"),
        n_matched=("matched", "sum"),
    )
    per_field["coverage"] = per_field["n_matched"] / per_field["n_slots"]

    # Missingness: slots where matched=False (no grid cell within 5 km).
    missing = grid[~grid["matched"]]
    audit_rows: list[dict] = []
    for gy in sorted(grid["grid_year"].unique()):
        sub = grid[grid["grid_year"] == gy]
        audit_rows.append({
            "grid_year": int(gy),
            "fields_requested": int(len(sub)),
            "fields_matched": int(sub["matched"].sum()),
            "missing": int((~sub["matched"]).sum()),
            "median_nearest_distance_m": round(
                float(sub.loc[sub["matched"], "nearest_distance_m"].median()), 1)
            if sub["matched"].any() else None,
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_DIR / "temporal_coverage_audit.csv", index=False)

    # Representation B: season-aligned sequence (Kharif vs Rabi composites).
    s2 = leading[["field_id", "grid_year", "kharif_ndvi", "rabi_ndvi"]].copy()
    seq_b = {
        "kharif_composite_years": int(s2["grid_year"].nunique()),
        "rabi_composite_years": int(s2["grid_year"].nunique()),
        "note": ("each DK grid year carries a full per-season composite set "
                 "(\"Annual + Kharif/Rabi composites\"); the season-aligned "
                 "sequence uses the Kharif_* columns across years for Kharif "
                 "fields and Rabi_* for Rabi fields, and BOTH are emitted "
                 "every year for every field"),
    }

    out = {
        "phase": "3",
        "representations": {
            "A_available_date_sequence": (
                "ordered sequence of grid years 2018..survey_year per field"),
            "B_season_aligned_sequence": (
                "per-season composite columns (Kharif_* / Rabi_*) aligned "
                "across the same grid years"),
            "C_fixed_temporal_bins": (
                "the 2018..2021 grid years ARE fixed annual bins; every bin "
                "is an observed grid composite, never imputed"),
        },
        "grid_rows": int(len(grid)),
        "leading_rows": int(len(leading)),
        "usable_fields_with_any_leading_slot": int(per_field["n_slots"].gt(0).sum()),
        "fields_with_full_leading_coverage": int(
            (per_field["coverage"] == 1.0).sum()),
        "matched_fraction_in_leading_window": round(
            float(leading["matched"].mean()), 4),
        "per_grid_year_coverage": audit_rows,
        "observation_slots_per_field": {
            "mean": round(float(per_field["n_slots"].mean()), 3),
            "median": int(per_field["n_slots"].median()),
            "min": int(per_field["n_slots"].min()),
            "max": int(per_field["n_slots"].max()),
        },
        "missingness": {
            "missing_matches_in_leading": int(missing["in_leading"].sum()),
            "missing_policy": ("unmatched slots are kept as explicit missing "
                               "(never imputed as real observations); temporal "
                               "aggregates are computed over matched slots only"),
        },
        "season_aligned": seq_b,
        "template": ("grid years <= survey year define the leading window; "
                     "2020 fields -> [2018, 2019, 2020]; 2021 fields -> "
                     "[2018, 2019, 2020, 2021]"),
    }
    write_report_json(out, "temporal_alignment.json")
    return out


# ---------------------------------------------------------------------------
# Phase 4 — temporal features (SET B aggregates + SET C trajectory)
# ---------------------------------------------------------------------------
def phase4_temporal_features(grid: pd.DataFrame | None = None) -> pd.DataFrame:
    if grid is None:
        grid = pd.read_csv(OUT_DIR / "temporal_field_grid.csv",
                           low_memory=False)
    pool = load_binary_pool()
    coords = load_field_coords()
    unit = pool.merge(coords, on="field_id", how="left").dropna(
        subset=["lat"]).copy()
    unit["leading_yrs"] = unit["year"].map(leading_years)
    year_of = unit.set_index("field_id")["year"]

    grid = grid[grid["field_id"].isin(set(unit["field_id"]))].copy()
    grid["survey_year"] = grid["field_id"].map(year_of)
    lead = grid[grid["grid_year"] <= grid["survey_year"]].copy()
    lead = lead[lead["matched"]].copy()

    agg_cols: list[str] = []
    heat = pd.DataFrame({"field_id": unit["field_id"]})
    for v in TEMPORAL_VARS:
        for stat in ["mean", "std", "min", "max"]:
            col = f"{v}_tmean" if stat == "mean" else f"{v}_t{stat}"
            series = lead.groupby("field_id")[v].agg(stat)
            heat[col] = heat["field_id"].map(series)
            agg_cols.append(col)

    # range, first-to-last delta, linear slope (require >=2 / >=3 obs).
    def _per_field_slope(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) < 3:
            return float("nan")
        y = s.to_numpy(dtype=float)
        x = np.arange(len(y), dtype=float)
        x = x - x.mean()
        return float(np.polyfit(x, y, 1)[0])

    for v in TEMPORAL_VARS:
        tmp = lead[["field_id", "grid_year", v]].copy()
        tmp = tmp.sort_values("grid_year")
        tmp[v] = pd.to_numeric(tmp[v], errors="coerce")
        g = tmp.groupby("field_id")[v]
        heat[f"{v}_trange"] = heat["field_id"].map(
            (g.max() - g.min()).dropna())
        first_last = tmp.groupby("field_id")[v].agg(
            ["first", "last"])
        heat[f"{v}_tdelta"] = heat["field_id"].map(
            (first_last["last"] - first_last["first"]).dropna())
        heat[f"{v}_tslope"] = heat["field_id"].map(
            tmp.groupby("field_id")[v].apply(_per_field_slope).dropna())
        agg_cols.extend([f"{v}_trange", f"{v}_tdelta", f"{v}_tslope"])

    n_valid = lead.groupby("field_id")["grid_year"].count()
    heat["n_valid_temporal_obs"] = heat["field_id"].map(n_valid).fillna(0)
    heat["n_valid_s2_obs_total"] = heat["field_id"].map(
        lead.groupby("field_id")["s2_obs_count"].sum()).fillna(0)

    # SET C trajectory: ordered per-grid-year values for each temporal var.
    traj_cols: list[str] = []
    for v in TEMPORAL_VARS:
        piv = lead.pivot_table(index="field_id", columns="grid_year",
                               values=v, aggfunc="mean")
        for gy in sorted(piv.columns):
            col = f"{v}_y{gy}"
            heat[col] = heat["field_id"].map(piv[gy])
            traj_cols.append(col)

    heat = heat.set_index("field_id")
    # static numeric vars from the survey-year matched grid (constant values)
    # carry over from the R5.9 static set, so SET C keeps only the trajectory.
    feats = heat[[c for c in heat.columns if c in agg_cols + traj_cols
                  or c in ("n_valid_temporal_obs", "n_valid_s2_obs_total")]]
    feats = feats.reset_index()

    audit = pd.DataFrame({
        "feature": feats.columns,
        "non_null_count": feats.notna().sum().values,
    })
    audit["missing_fraction"] = round(
        1 - audit["non_null_count"] / max(len(feats), 1), 4)
    audit.to_csv(OUT_DIR / "temporal_feature_audit.csv", index=False)

    feats.to_csv(OUT_DIR / "temporal_features.csv", index=False)

    out = {
        "phase": "4",
        "set_b_temporal_aggregates": len(agg_cols),
        "set_c_trajectory_slots": len(traj_cols),
        "temporal_vars": TEMPORAL_VARS,
        "static_numeric_vars": STATIC_NUMERIC,
        "static_categorical_vars": STATIC_CATEGORICAL,
        "aggregate_stats": ["mean", "std", "min", "max", "range",
                            "first-to-last delta", "linear slope"],
        "trajectory_policy": ("ordered per-grid-year values emitted as "
                              "slot columns v_y<year>; missing slots stay "
                              "NaN; fields surveyed in 2020 have 3 slots, "
                              "fields surveyed in 2021 have 4 slots"),
        "min_obs_for_stats": ("range/delta require >=2 matched grid years; "
                              "slope requires >=3"),
        "rows": int(len(feats)),
        "fields_with_temporal_features": int(feats["field_id"].nunique()),
    }
    write_report_json(out, "temporal_features.json")
    return feats


# ---------------------------------------------------------------------------
# Phase 5 — target + dataset assembly
# ---------------------------------------------------------------------------
RESERVED_IDENTITY = [
    "context_key", "field_id", "year", "season", "taluk", "hobli", "village",
    "survey_id", "leading", "leading_yrs", "leading_years_str", "n_leading_years",
    "lat", "lon", "field_cluster_id", "split",
]

def phase5_target(ds_out: str = "field_target_dataset.csv") -> pd.DataFrame:
    pool = load_binary_pool()
    coords = load_field_coords()
    ds = pool.merge(coords, on="field_id", how="left").dropna(
        subset=["lat"]).copy()
    temporal_feats = pd.read_csv(OUT_DIR / "temporal_features.csv")
    ds = ds.merge(temporal_feats, on="field_id", how="left")

    target_audit = pd.DataFrame({
        "split": ds["split"],
        "dominant_crop": ds["dominant_crop"],
    })
    target_audit["target"] = ds["target"]
    target_audit.to_csv(OUT_DIR / "target_audit.csv", index=False)

    out = {
        "phase": "5",
        "target": "R5.9 dominant-crop composition target (coconut=0/pepper=1); "
                  "exactly the field_dataset_split.csv dominant_crop column",
        "excluded_from_primary": "mixed/tie/other dominant fields (as R5.9)",
        "raw_class_counts": {"coconut_dominant": int((ds["target"] == 0).sum()),
                             "pepper_dominant": int((ds["target"] == 1).sum())},
        "class_ratio_pepper_to_coconut": round(
            float((ds["target"] == 1).sum() / max((ds["target"] == 0).sum(), 1)),
            4),
        "train_val_test_counts": to_py(ds["split"].value_counts().to_dict()),
        "by_split_class": to_py(
            ds.groupby(["split", "dominant_crop"]).size().to_dict()),
        "feature_columns": [c for c in ds.columns
                            if c not in RESERVED_IDENTITY
                            and c not in ("dominant_crop", "top1_fraction",
                                          "dominance_gap",
                                          "n_crops_in_composition",
                                          "coconut_fraction",
                                          "pepper_fraction", "target",
                                          "n_master_rows_in_context",
                                          "quality_tier", "confidence_tier")],
        "crop_extent_active": {
            "as_feature": False,
            "as_target": True,
            "unit": "UNKNOWN (preserved from R5.9; never claimed as acres/m2)",
        },
    }
    write_report_json(out, "target.json")
    # persist the unified dataset (gitignored)
    ds.to_csv(OUT_DIR / ds_out, index=False)
    return ds


# ---------------------------------------------------------------------------
# Phase 6 — split integrity
# ---------------------------------------------------------------------------
def phase6_split(ds: pd.DataFrame | None = None) -> dict:
    if ds is None:
        ds = pd.read_csv(OUT_DIR / "field_target_dataset.csv",
                         low_memory=False)
    splits = {"train", "val", "test"}
    field_split = ds.groupby("field_id")["split"].nunique()
    cross = int((field_split > 1).sum())

    sets = {s: set(ds[ds["split"] == s]["field_id"]) for s in splits}
    inter = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }

    # near-duplicate / co-located audit: fields sharing the same ~50 m R5.7
    # cluster or the same DK grid cell must not straddle splits in a way that
    # only exists across splits (cells/clusters straddling train/test).
    cluster_split = ds.groupby(["field_cluster_id"])["split"].nunique()
    straddling_clusters = int((cluster_split > 1).sum())

    # approximate ~1 km DK cell index from rounded coords
    ds["cell_key"] = (ds["lat"].round(2).astype(str) + "|"
                      + ds["lon"].round(2).astype(str))
    cell_split = ds.groupby("cell_key")["split"].nunique()
    straddling_cells = int((cell_split > 1).sum())
    cell_train_test = ds[ds["cell_key"].isin(
        set(ds[ds["split"] == "train"]["cell_key"])
        & set(ds[ds["split"] == "test"]["cell_key"]))]

    rows = []
    for s in splits:
        sub = ds[ds["split"] == s]
        for crop in ["coconut", "pepper"]:
            rows.append({"split": s, "dominant_crop": crop,
                         "fields": int((sub["dominant_crop"] == crop).sum()),
                         "with_temporal": int(sub["n_valid_temporal_obs"].notna().sum())})
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT_DIR / "split_audit.csv", index=False)

    out = {
        "phase": "6",
        "rule": "grouped field splits by taluk (TALUK_SPLIT), inherited from "
                "R5.9; no new split logic introduced",
        "fields_crossing_splits": cross,
        "intersection_sizes": inter,
        "split_integrity": all(v == 0 for v in inter.values()) and cross == 0,
        "co_located_audit": {
            "r5_7_50m_clusters_straddling_splits": straddling_clusters,
            "approx_1km_cells_straddling_splits": straddling_cells,
            "fields_in_cells_straddling_train_test": int(len(cell_train_test)),
            "note": ("a field can never straddle splits (grouped by taluk); "
                     "co-located neighbours CAN sit in different splits when a "
                     "~1 km grid cell or 50 m cluster crosses a taluk boundary "
                     "- this is audited in the shortcut phase"),
        },
    }
    write_report_json(out, "split_design.json")
    return out


# ---------------------------------------------------------------------------
# Evaluation protocol (shared across all model sets)
# ---------------------------------------------------------------------------
def _models(seed: int = SEED):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    return [
        ("lr", LogisticRegression(max_iter=3000, random_state=seed,
                                  class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=250, random_state=seed,
                                      class_weight="balanced")),
        ("gb", GradientBoostingClassifier(n_estimators=250, random_state=seed,
                                          max_depth=4)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                              random_state=seed)),
        ("xgb", _xgb(seed)),
    ]


def _xgb(seed: int = SEED):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=seed,
        scale_pos_weight=None, verbosity=0,
    )


def _matrix_from_cols(df: pd.DataFrame, num_cols: list[str],
                      cat_cols: list[str], train_idx: np.ndarray) -> np.ndarray:
    """Numeric median-fill (train-only) + categorical factorise (train-only)."""
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    Xn = df[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float, copy=True) if num_cols else np.empty((len(df), 0))
    med = np.nanmedian(Xn[train_idx], axis=0)
    med = np.nan_to_num(med, nan=0.0)
    for j in range(Xn.shape[1]):
        col = Xn[:, j]
        if np.isnan(col).any():
            col[np.isnan(col)] = med[j]
    Xn = np.nan_to_num(Xn, nan=0.0)
    if cat_cols:
        Xc = df[cat_cols].fillna("__nan__").astype(str)
        # clean train-only factorisation (test/val levels mapped to 0)
        codes = np.zeros((len(df), len(cat_cols)), dtype=float)
        for j, c in enumerate(cat_cols):
            vals = Xc[c].to_numpy()
            train_vals = vals[train_idx]
            _, tr = pd.factorize(train_vals, sort=False)
            table = {v: i for i, v in enumerate(tr)}
            for i2, v in enumerate(vals):
                codes[i2, j] = table.get(v, -1)
            codes[codes[:, j] == -1, j] = 0.0
        out = np.hstack([Xn, codes]).astype(np.float64)
    else:
        out = Xn.astype(np.float64)
    return np.ascontiguousarray(out, dtype=np.float64)


def _metrics_full(y_true: np.ndarray, y_pred: np.ndarray,
                  y_prob: np.ndarray | None) -> dict:
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 f1_score, recall_score, roc_auc_score)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    per_class = recall_score(y_true, y_pred, labels=[0, 1], average=None)
    return {
        "accuracy": round(float((y_pred == y_true).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                         zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4)
        if y_prob is not None else None,
        "recall_coconut": round(float(per_class[0]), 4),
        "recall_pepper": round(float(per_class[1]), 4),
        "confusion_matrix": cm,
    }


def _run_pool(ds: pd.DataFrame, num_cols: list[str], cat_cols: list[str],
              tag: str, results: list[dict] | None = None,
              prob_store: dict | None = None,
              matrix: Callable[[pd.DataFrame], np.ndarray] | None = None) -> dict:
    """Shared evaluation protocol (identical for every feature set).

    ``matrix`` overrides the numeric/categorical column protocol (used ONLY for
    the bit-exact R5.9 SET-A reproduction); otherwise the train-only imputation
    protocol applies.
    """
    y = ds["target"].to_numpy()
    splits = ds["split"].to_numpy()
    idx = {s: np.where(splits == s)[0] for s in ("train", "val", "test")}
    if min(len(v) for v in idx.values()) == 0:
        return {"pool": tag, "error": "empty split"}
    X = (matrix(ds) if matrix is not None
         else _matrix_from_cols(ds, num_cols, cat_cols, idx["train"]))
    test_y = y[idx["test"]]
    majority = float(max((test_y == 0).mean(), (test_y == 1).mean()))
    majority_pred = np.zeros_like(test_y)
    majority_met = _metrics_full(test_y, majority_pred, None)

    per: dict[str, Any] = {}
    for name, clf in _models():
        clf.fit(X[idx["train"]], y[idx["train"]])
        row: dict[str, Any] = {"pool": tag, "model": name}
        for split in ("val", "test"):
            Xs, ys = X[idx[split]], y[idx[split]]
            pr = clf.predict(Xs)
            p = clf.predict_proba(Xs)[:, 1]
            met = _metrics_full(ys, pr, p)
            for k, v in met.items():
                row[f"{split}_{k}"] = v
            if split == "test" and prob_store is not None:
                prob_store[f"{tag}:{name}"] = {
                    "y_true": ys.tolist(), "y_prob": p.tolist()}
        per[name] = row
        if results is not None:
            results.append(row)

    best = max(per.values(),
               key=lambda r: (r.get("test_balanced_accuracy") or 0.0))
    out: dict[str, Any] = {
        "pool": tag,
        "n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])),
        "n_test": int(len(idx["test"])),
        "test_majority_accuracy": round(majority, 4),
        "test_majority_balanced_accuracy": round(
            majority_met["balanced_accuracy"], 4),
        "test_majority_macro_f1": round(majority_met["macro_f1"], 4),
        "best_test_balanced_accuracy": best["test_balanced_accuracy"],
        "best_model": best["model"],
        "best_test_macro_f1": best["test_macro_f1"],
        "best_test_roc_auc": best["test_roc_auc"],
        "best_test_recall_pepper": best["test_recall_pepper"],
        "best_test_confusion_matrix": best["test_confusion_matrix"],
        "per_model": to_py(per),
    }
    return out


def _set_a_pool(ds: pd.DataFrame) -> tuple[list[str], list[str]]:
    """R5.9 SET A feature columns (identical to R5.9's `_xy` column rule:
    R56 environmental numerics + R56 categoricals + the `sat_*` survey-frame
    satellite features)."""
    num = [c for c in ENV_NUMERIC if c in ds.columns
           and not c.startswith("year.") and c != "year"
           and c not in ("coconut_fraction", "pepper_fraction",
                         "lat", "lon", "spatial_match_distance_km")]
    num += [c for c in ds.columns if c.startswith("sat_")]
    cat = [c for c in R56_CATEGORICAL if c in ds.columns
           and c not in ("coconut_fraction", "pepper_fraction")]
    return num, cat


def _r59_xy(ds: pd.DataFrame) -> np.ndarray:
    """EXACT reproduction of R5.9's `_xy` matrix (full-pool median imputation
    and full-pool factorisation, env num + categorical + trailing sat_*),
    so the R5.9 static baseline is bit-reproducible here (0.5042 / 0.4781)."""
    num = [c for c in ENV_NUMERIC if c in ds.columns
           and not c.startswith("year.") and c != "year"
           and c not in ("coconut_fraction", "pepper_fraction",
                         "lat", "lon", "spatial_match_distance_km")]
    cat = [c for c in R56_CATEGORICAL if c in ds.columns
           and c not in ("coconut_fraction", "pepper_fraction")]
    sat = [c for c in ds.columns if c.startswith("sat_")]
    cols = num + cat + sat
    Xn = ds[cols].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float, copy=True)
    Xn = np.ascontiguousarray(Xn, dtype=float)
    med = np.nanmedian(Xn, axis=0)
    med = np.nan_to_num(med, nan=0.0)
    for j in range(Xn.shape[1]):
        col = Xn[:, j]
        if np.isnan(col).any():
            col[np.isnan(col)] = med[j]
    Xn = np.nan_to_num(Xn, nan=0.0)
    Xc = ds[cat].fillna("__nan__").astype(str)
    codes = np.stack([
        pd.factorize(Xc[c], sort=False)[0] for c in Xc.columns
    ], axis=1) if len(Xc.columns) else np.empty((len(ds), 0))
    return np.hstack([Xn, codes]).astype(np.float64)


def _set_b_pool(ds: pd.DataFrame) -> list[str]:
    return [c for c in ds.columns if c.endswith(("_tmean", "_tstd", "_tmin",
                                                 "_tmax", "_trange", "_tdelta",
                                                 "_tslope"))]


def _set_c_pool(ds: pd.DataFrame) -> list[str]:
    return [c for c in ds.columns
            if re.match(r".+_y(2018|2019|2020|2021)$", c)]


def _temporal_only_num(ds: pd.DataFrame) -> list[str]:
    return ([c for c in ds.columns if "_t" in c and c != "target"
             and c not in ("field_id", "quality_tier", "confidence_tier")]
            + _set_c_pool(ds) + ["n_valid_temporal_obs",
                                 "n_valid_s2_obs_total"])


# ---------------------------------------------------------------------------
# Phase 7 — cheap baselines
# ---------------------------------------------------------------------------
def phase7_cheap_baselines(ds: pd.DataFrame | None = None) -> dict:
    if ds is None:
        ds = pd.read_csv(OUT_DIR / "field_target_dataset.csv",
                         low_memory=False)
    results: list[dict] = []
    probs: dict[str, Any] = {}

    num_a, cat_a = _set_a_pool(ds)
    setA = _run_pool(ds, num_a, cat_a, "SET-A-static-r5.9", results, probs)
    setA_exact = _run_pool(ds, num_a, cat_a,
                           "SET-A-r5.9-exact-reproduction", results, probs,
                           matrix=_r59_xy)

    setBnum = _set_b_pool(ds)
    setB = _run_pool(ds, setBnum, [], "SET-B-temporal-aggregates", results,
                     probs)
    setB_with_count = _run_pool(
        ds, setBnum + ["n_valid_temporal_obs", "n_valid_s2_obs_total"], [],
        "SET-B-temporal-aggregates+obs-count", results, probs)

    setCnum = _set_c_pool(ds)
    setC = _run_pool(ds, setCnum, [], "SET-C-trajectory", results, probs)

    setAB = _run_pool(ds, list(dict.fromkeys(num_a + setBnum + [
        "n_valid_temporal_obs", "n_valid_s2_obs_total"])), cat_a,
        "SET-AB-static+temporal", results, probs)

    temporal_only = _temporal_only_num(ds)
    temp_only = _run_pool(ds, temporal_only, [], "SET-temporal-only", results,
                          probs)

    out = {
        "phase": "7",
        "reference_r5_9": {"balanced_accuracy": R5_9_BALANCED_ACCURACY,
                           "roc_auc": R5_9_ROC_AUC,
                           "model": R5_9_BEST_MODEL},
        "runs": {
            "SET_A_static_r5_9": setA,
            "SET_A_r5_9_exact_reproduction": setA_exact,
            "SET_B_temporal_aggregates": setB,
            "SET_B_temporal_aggregates_plus_obs_count": setB_with_count,
            "SET_C_trajectory": setC,
            "SET_AB_static_plus_temporal": setAB,
            "SET_temporal_only": temp_only,
        },
        "set_a_exact_note": (
            "SET_A_r5_9_exact_reproduction uses the bit-exact R5.9 _xy matrix "
            "(full-pool median + full-pool factorise + sat_* appended); it "
            "should reproduce R5.9's 0.5042 / 0.4781. SET_A_static_r5_9 is the "
            "same feature set under the shared train-only imputation protocol "
            "used for every R5.10 comparison set."),
    }
    write_report_json(out, "cheap_baselines.json")
    pd.DataFrame(results).to_csv(OUT_DIR / "baseline_results.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Phase 8 — ablation
# ---------------------------------------------------------------------------
def phase8_ablation(ds: pd.DataFrame | None = None) -> dict:
    if ds is None:
        ds = pd.read_csv(OUT_DIR / "field_target_dataset.csv",
                         low_memory=False)
    results: list[dict] = []
    probs: dict[str, Any] = {}

    num_a, cat_a = _set_a_pool(ds)
    setA = _run_pool(ds, num_a, cat_a, "ab-SET-A-static-only", results, probs)
    setB = _run_pool(ds, _set_b_pool(ds), [],
                     "ab-SET-B-temporal-only", results, probs)
    setAB = _run_pool(ds, list(dict.fromkeys(num_a + _set_b_pool(ds))), cat_a,
                      "ab-SET-A+B", results, probs)
    setC = _run_pool(ds, _set_c_pool(ds), [],
                     "ab-SET-C-trajectory-only", results, probs)
    setAll = _run_pool(
        ds, list(dict.fromkeys(num_a + _set_b_pool(ds) + _set_c_pool(ds))),
        cat_a, "ab-SET-all", results, probs)

    def _delta(b: dict, a: dict) -> dict:
        return {
            "delta_balanced_accuracy": round(
                (b.get("best_test_balanced_accuracy") or 0.0)
                - (a.get("best_test_balanced_accuracy") or 0.0), 4),
            "delta_macro_f1": round(
                (b.get("best_test_macro_f1") or 0.0)
                - (a.get("best_test_macro_f1") or 0.0), 4),
            "delta_roc_auc": round(
                ((b.get("best_test_roc_auc") or 0.0))
                - ((a.get("best_test_roc_auc") or 0.0)), 4)
            if a.get("best_test_roc_auc") is not None else None,
        }

    out = {
        "phase": "8",
        "purpose": "does temporal information add real discriminative signal "
                   "over the R5.9 static representation?",
        "against_r5_9_baseline": {
            "SET_B_vs_R5_9": _delta(setB, {"best_test_balanced_accuracy": R5_9_BALANCED_ACCURACY,
                                           "best_test_macro_f1": None,
                                           "best_test_roc_auc": R5_9_ROC_AUC}),
            "SET_AB_vs_R5_9": _delta(setAB, {"best_test_balanced_accuracy": R5_9_BALANCED_ACCURACY,
                                             "best_test_macro_f1": None,
                                             "best_test_roc_auc": R5_9_ROC_AUC}),
        },
        "against_inside_r5_10": {
            "SET_B_minus_SET_A": _delta(setB, setA),
            "SET_AB_minus_SET_A": _delta(setAB, setA),
            "SET_C_minus_SET_A": _delta(setC, setA),
            "SET_all_minus_SET_A": _delta(setAll, setA),
        },
        "runs": {"SET_A": setA, "SET_B": setB, "SET_AB": setAB,
                 "SET_C": setC, "SET_all": setAll},
    }
    write_report_json(out, "ablation_results.json")
    pd.DataFrame(results).to_csv(OUT_DIR / "ablation_results.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Phase 9 — shortcut audits
# ---------------------------------------------------------------------------
def phase9_shortcuts(ds: pd.DataFrame | None = None) -> dict:
    if ds is None:
        ds = pd.read_csv(OUT_DIR / "field_target_dataset.csv",
                         low_memory=False)
    results: list[dict] = []
    probs: dict[str, Any] = {}

    probes: dict[str, dict] = {}

    # 1. year shortcut
    probes["year_only"] = _run_pool(ds, ["year"], [], "short-year", results,
                                    probs)
    # 2. season shortcut
    probes["season_only"] = _run_pool(ds, [], ["season"], "short-season",
                                      results, probs)
    # 3. observation-count shortcut (s2_obs_count survey-year + valid obs)
    probes["obs_count_only"] = _run_pool(
        ds, ["n_valid_temporal_obs", "n_valid_s2_obs_total", "s2_obs_count"],
        [], "short-obs-count", results, probs)
    # 4. missingness shortcut: encode missing pattern of temporal slots
    n_valid = ds["n_valid_temporal_obs"].fillna(-1)
    miss = pd.DataFrame({"target": ds["target"], "split": ds["split"],
                         "n_valid": n_valid})
    probes["missingness_only"] = _run_pool(
        miss, ["n_valid"], [], "short-missingness", results, probs)
    # 5. location shortcut (lat/lon only)
    probes["location_only"] = _run_pool(ds, ["lat", "lon"], [],
                                        "short-location", results, probs)
    # 6. administrative-region shortcut (taluk only)
    probes["taluk_only"] = _run_pool(ds, [], ["taluk"], "short-taluk",
                                     results, probs)
    # 7. DK grid-cell shortcut: approximate 1 km cell id as categorical
    ds2 = ds.copy()
    ds2["cell_key"] = (ds2["lat"].round(2).astype(str) + "|"
                       + ds2["lon"].round(2).astype(str))
    probes["grid_cell_only"] = _run_pool(ds2, [], ["cell_key"],
                                         "short-grid-cell", results, probs)

    suspicious: list[str] = []
    for name, probe in probes.items():
        bal = probe.get("best_test_balanced_accuracy") or 0.0
        if bal >= 0.65:
            suspicious.append(name)

    rows = [{"probe": name, "best_test_balanced_accuracy":
             (probe.get("best_test_balanced_accuracy") or 0.0),
             "best_model": probe.get("best_model")}
            for name, probe in probes.items()]
    pd.DataFrame(rows).to_csv(OUT_DIR / "shortcut_audit.csv", index=False)

    out = {
        "phase": "9",
        "audits": probes,
        "suspicious_probes_ge_65pct": suspicious,
        "interpretation": (
            "a shortcut probe is a RAW feature (year/season/count/location/"
            "admin/cell) evaluated alone; high performance signals the feature "
            "encodes location/time shortcuts, NOT crop signal"),
    }
    write_report_json(out, "shortcut_audit.json")
    return out


# ---------------------------------------------------------------------------
# Phase 10 — robustness
# ---------------------------------------------------------------------------
def phase10_robustness(ds: pd.DataFrame | None = None,
                       grid: pd.DataFrame | None = None) -> dict:
    if ds is None:
        ds = pd.read_csv(OUT_DIR / "field_target_dataset.csv",
                         low_memory=False)
    results: list[dict] = []
    probs: dict[str, Any] = {}

    num_a, cat_a = _set_a_pool(ds)

    # (a) require >=2 valid temporal observations
    d2 = ds[ds["n_valid_temporal_obs"].fillna(0) >= 2]
    # (b) require >=3 valid temporal observations
    d3 = ds[ds["n_valid_temporal_obs"].fillna(0) >= 3]

    setA2 = _run_pool(d2, num_a, cat_a, "rob-A-min2", results, probs)
    setB2 = _run_pool(d2, _set_b_pool(d2), [], "rob-B-aggregates-min2",
                      results, probs)
    setAB2 = _run_pool(d2, list(dict.fromkeys(num_a + _set_b_pool(d2))),
                       cat_a, "rob-AB-min2", results, probs)
    setB3 = _run_pool(d3, _set_b_pool(d3), [], "rob-B-aggregates-min3",
                      results, probs)

    # (c) full-window 2018-2023 sensitivity (future years relative to the
    # survey year) — reported ONLY as a sensitivity, never as primary.
    full = _full_window_features()
    full_fuse = ds.merge(full, on="field_id", how="left") \
        if full is not None and len(full) else None
    full_run = None
    if full_fuse is not None and len(full_fuse):
        full_agg = [c for c in full_fuse.columns
                    if re.match(r"(full_.*_(mean|std|min|max))$", c)]
        full_run = _run_pool(full_fuse, full_agg, [],
                             "rob-full-window-2018-23-aggregates", results,
                             probs)

    out = {
        "phase": "10",
        "min_observation_variants": {
            "min2_fields": int(len(d2)),
            "min3_fields": int(len(d3)),
            "min2_A": setA2, "min2_B": setB2, "min2_AB": setAB2,
            "min3_B": setB3,
        },
        "full_window_sensitivity": full_run,
        "policy": ("min-observation thresholds and the 2018-2023 full-window "
                   "representation are sensitivity checks; the PRIMARY result "
                   "is the leading-window (<= survey year) representation with "
                   "all fields that have >=1 matched temporal slot"),
    }
    write_report_json(out, "robustness_results.json")
    pd.DataFrame(results).to_csv(OUT_DIR / "robustness_results.csv", index=False)
    return out


def _full_window_features() -> pd.DataFrame | None:
    p = OUT_DIR / "temporal_field_grid.csv"
    if not p.exists():
        return None
    grid = pd.read_csv(p, low_memory=False)
    grid = grid[grid["matched"]].copy()
    if not len(grid):
        return None
    out = pd.DataFrame({"field_id": grid["field_id"].unique()})
    for v in TEMPORAL_VARS:
        for stat in ["mean", "std", "min", "max"]:
            col = f"full_{v}_{stat}"
            out[col] = out["field_id"].map(grid.groupby("field_id")[v].agg(stat))
    return out


# ---------------------------------------------------------------------------
# Phase 11 — decision
# ---------------------------------------------------------------------------
def phase11_decision(baselines: dict, ablation: dict, shortcuts: dict,
                     robustness: dict) -> dict:
    # The headline best is chosen over PRIMARY runs only (the canonical
    # cheap-baseline feature sets). Ablation tag-duplicates and robustness
    # subsets are reported separately and never promoted to the headline.
    primary_runs = {k: v for k, v in baselines["runs"].items()
                    if not k.startswith("rob_")}

    runs_acc = {k: (v.get("best_test_balanced_accuracy") or 0.0)
                for k, v in primary_runs.items()}
    best_key = max(runs_acc, key=runs_acc.get)
    best_run = primary_runs[best_key]
    best_acc = float(best_run.get("best_test_balanced_accuracy") or 0.0)
    best_auc_vals = [v.get("best_test_roc_auc") for v in primary_runs.values()
                     if isinstance(v.get("best_test_roc_auc"), (int, float))]
    best_auc = max(best_auc_vals) if best_auc_vals else None

    key_temporal = baselines["runs"]["SET_B_temporal_aggregates"]
    key_ab = baselines["runs"]["SET_AB_static_plus_temporal"]
    setb_acc = key_temporal.get("best_test_balanced_accuracy") or 0.0
    setb_auc = key_temporal.get("best_test_roc_auc")
    setab_acc = key_ab.get("best_test_balanced_accuracy") or 0.0
    setab_auc = key_ab.get("best_test_roc_auc")

    suspicious = shortcuts.get("suspicious_probes_ge_65pct", [])
    delta_b = setb_acc - R5_9_BALANCED_ACCURACY
    delta_ab = setab_acc - R5_9_BALANCED_ACCURACY
    improvement_consistent = (delta_b > 0.005 and delta_ab > 0.005)

    if (setab_acc < GATE_NO_SIGNAL_HI or len(suspicious) > 0
            or (best_auc is not None and best_auc <= 0.52)
            or not improvement_consistent):
        verdict = "no_signal"
        rec = ("temporal satellite composites (leading window, 2018..survey "
               "year) do not recover a usable post-split coconut-vs-pepper "
               "field signal. The current data formulation does not provide "
               "sufficient evidence to justify full CropFusion training or "
               "claims of high-accuracy classification.")
    elif best_acc < GATE_WEAK_HI:
        verdict = "weak_signal"
        rec = "temporal information adds a small, unstable improvement; not enough for CropFusion."
    elif best_acc < GATE_PROMISING_HI:
        verdict = "promising_signal"
        rec = "temporal information shows a clear held-out improvement; a future R5.11 may evaluate CropFusion after the listed checks."
    else:
        verdict = "strong_signal"
        rec = "temporal information produces strong held-out discrimination; verify all leakage checks before any cropfusion step."

    out = {
        "phase": "11",
        "status": "COMPLETE",
        "best_model": best_run.get("best_model"),
        "best_feature_set": best_key,
        "best_selection_policy": (
            "best chosen over PRIMARY runs only (the canonical cheap-baseline "
            "feature sets); ablation tag-duplicates and robustness subsets "
            "are reported separately and never promoted to the headline"),
        "best_balanced_accuracy": round(best_acc, 4),
        "best_macro_f1": best_run.get("best_test_macro_f1"),
        "best_auc": round(best_auc, 4) if best_auc is not None else None,
        "R5.9_balanced_accuracy": R5_9_BALANCED_ACCURACY,
        "delta_vs_R5.9": round(best_acc - R5_9_BALANCED_ACCURACY, 4),
        "signal_decision": verdict,
        "leakage_status": {
            "suspicious_shortcut_probes": suspicious,
            "clean": len(suspicious) == 0,
        },
        "spatial_split_status": "grouped by taluk; no field crosses splits",
        "temporal_alignment_status": "leading-window grid years <= survey year",
        "crop_extent_unit_status": "UNKNOWN",
        "recommendation": rec,
        "gates": {
            "no_signal": "balanced accuracy < 55% OR a suspicious shortcut OR AUC <= 0.52",
            "weak": "55-60% with no shortcut",
            "promising": "60-70% with no shortcut",
            "strong": ">70%",
        },
        "evidence": {
            "SET_A_static_balanced_acc": round(
                baselines["runs"]["SET_A_static_r5_9"].get(
                    "best_test_balanced_accuracy") or 0.0, 4),
            "SET_B_temporal_balanced_acc": round(setb_acc, 4),
            "SET_B_temporal_auc": setb_auc,
            "SET_AB_balanced_acc": round(setab_acc, 4),
            "SET_AB_auc": setab_auc,
            "delta_SET_B_vs_R5_9": round(delta_b, 4),
            "delta_SET_AB_vs_R5_9": round(delta_ab, 4),
        },
    }
    write_report_json(out, "R5.10_decision.json")
    return out


# ---------------------------------------------------------------------------
# Phase 12 — report
# ---------------------------------------------------------------------------
def phase12_report(phase_outs: dict) -> dict:
    dec = phase_outs["d11"]
    tunit = phase_outs["d1"]
    alignment = phase_outs["d3"]
    target = phase_outs["d5"]
    split = phase_outs["d6"]

    lines = [
        "# R5.10 Temporal Field-Target Recoverability Experiment",
        "",
        "## Final status block",
        f"- **STATUS**: {dec['status']}",
        f"- **BEST_TEST_BALANCED_ACCURACY**: {dec['best_balanced_accuracy']}",
        f"- **R5.9_BALANCED_ACCURACY**: {dec['R5.9_balanced_accuracy']}",
        f"- **DELTA_VS_R5_9**: {dec['delta_vs_R5.9']}",
        f"- **BEST_TEST_AUC**: {dec['best_auc']}",
        f"- **BEST_MODEL**: {dec['best_model']}",
        f"- **SIGNAL_DECISION**: {dec['signal_decision']}",
        f"- **SPATIAL_SPLIT_STATUS**: {dec['spatial_split_status']}",
        f"- **TEMPORAL_ALIGNMENT_STATUS**: {dec['temporal_alignment_status']}",
        f"- **CROP_EXTENT_UNIT_STATUS**: {dec['crop_extent_unit_status']}",
        f"- **CROPFUSION_JUSTIFIED**: {dec['signal_decision'] in ('promising_signal', 'strong_signal')}",
        f"- **RECOMMENDATION**: {dec['recommendation']}",
        "",
        "## 1. Temporal unit",
        f"- Unique fields: **{tunit['unique_fields']:,}**",
        f"- Unique grid years, leading window: {tunit['unique_grid_years_available']}",
        f"- Temporal grid slots per field (median): {tunit['temporal_grid_slots_per_field']['median']}",
        f"- Fields with >=3 grid slots: {tunit['fields_with_n_grid_slots']['ge3']:,}",
        "",
        "## 2. Temporal alignment / coverage",
        f"- Usable fields with any leading slot: "
        f"**{alignment['usable_fields_with_any_leading_slot']:,}**",
        f"- Matched fraction (leading window): "
        f"**{alignment['matched_fraction_in_leading_window']}**",
        f"- Observations per field (slots, mean): "
        f"{alignment['observation_slots_per_field']['mean']}",
        "",
        "## 3. Satellite information actually available",
        "Temporally-varying DK grid composites (2018-2023) at each field's "
        "real GPS: NDVI, EVI, NDWI, NDRE, SAVI, S2_Obs_Count, seasonal "
        "Kharif/Rabi composites and per-year climate (rainfall, temperature, "
        "dewpoint, humidity). Static terrain/soil variables are not temporal.",
        "",
        "## 4. Target",
        f"- Coconut-dominant: **{target['raw_class_counts']['coconut_dominant']:,}**; "
        f"pepper-dominant: **{target['raw_class_counts']['pepper_dominant']:,}**",
        f"- Class ratio (pepper/coconut): {target['class_ratio_pepper_to_coconut']}",
        f"- by split: {json.dumps(target['by_split_class'])}",
        "",
        "## 5. Split integrity",
        f"- Fields crossing splits: **{split['fields_crossing_splits']}**",
        f"- Intersections: {json.dumps(split['intersection_sizes'])}",
        f"- Integrity: **{split['split_integrity']}**",
        "",
        "## 6. Headline result",
        f"- R5.9 static baseline balanced accuracy: "
        f"**{R5_9_BALANCED_ACCURACY}** (AUC {R5_9_ROC_AUC})",
        f"- SET B (temporal aggregates) balanced accuracy: "
        f"**{dec.get('evidence', {}).get('SET_B_temporal_balanced_acc')}** "
        f"(AUC {dec.get('evidence', {}).get('SET_B_temporal_auc')})",
        f"- SET A+B balanced accuracy: "
        f"**{dec.get('evidence', {}).get('SET_AB_balanced_acc')}** "
        f"(AUC {dec.get('evidence', {}).get('SET_AB_auc')})",
        f"- Δ vs R5.9 (best/overall): **{dec['delta_vs_R5.9']}**",
        "",
        "## 7. Conclusion",
        dec["recommendation"],
    ]
    (OUT_DIR / "R5.10_report.md").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")

    report = {
        "title": "R5.10 Temporal Field-Target Recoverability",
        "status": dec["status"],
        "answers": {
            "how_many_fields_have_enough_temporal_observations": (
                f"{tunit['unique_fields']:,} physical binary fields with "
                f"median {tunit['temporal_grid_slots_per_field']['median']} "
                f"leading-window grid slots"),
            "what_temporal_satellite_information_is_available": (
                "2018-2023 DK grid annual + seasonal vegetation composites "
                "(NDVI/EVI/NDWI/NDRE/SAVI, S2_Obs_Count) and per-year climate "
                "at each field's real GPS"),
            "does_temporal_information_improve_over_r5_9": (
                f"delta SET_B={dec.get('evidence', {}).get('delta_SET_B_vs_R5_9')}, "
                f"delta SET_AB={dec.get('evidence', {}).get('delta_SET_AB_vs_R5_9')}"),
            "best_held_out_balanced_accuracy": dec["best_balanced_accuracy"],
            "best_auc": dec["best_auc"],
            "does_improvement_survive_leakage_and_spatial_shortcut_audits": (
                f"suspicious shortcuts={dec['leakage_status']['suspicious_shortcut_probes']}"),
            "is_signal_strong_enough_to_justify_cropfusion": (
                dec["signal_decision"] in ("promising_signal", "strong_signal")),
            "can_current_data_scientifically_support_90": (
                dec["signal_decision"] == "strong_signal"),
            "next_recommended_experiment": dec["recommendation"],
        },
        "decision": dec,
        "temporal_unit": tunit,
        "temporal_alignment": alignment,
        "target": target,
        "split": split,
    }
    write_report_json(report, "R5.10_report.json")
    return report


# ---------------------------------------------------------------------------
# Phase 13 — provenance
# ---------------------------------------------------------------------------
def phase13_provenance() -> dict:
    files = {name: (SURVEY_DIR / name) for name in SURVEY_FILES}
    dk = {f"DK_Features_{y}.csv": sha(DK_DIR / f"DK_Features_{y}.csv")
          for y in GRID_YEARS_ALL}
    contract: dict[str, Any] = {
        "phase": "13",
        "schema_version": "r5.10.0",
        "seed": SEED,
        "source_fingerprint_survey": {name: sha(p) for name, p in files.items()},
        "source_fingerprint_dk_grids": dk,
        "r5_9_dataset": {"path": str(R59_DATASET),
                         "sha256": sha(R59_DATASET)},
        "r5_9_report": {"path": str(R59_REPORT),
                        "sha256": sha(R59_REPORT)},
        "master": {"path": str(MASTER_CSV), "sha256": sha(MASTER_CSV)},
        "target_construction": (
            "R5.9 dominant-crop composition target consumed unchanged from "
            "reports/R5.9/field_dataset_split.csv; dominant crop derived from "
            "the UNKNOWN-unit Crop_Extent via the R5.9 relative scalarization "
            "A + B/100 + C/10000 within each field only"),
        "crop_extent_unit_status": "UNKNOWN",
        "crop_extent_as_feature": False,
        "temporal_features_source": (
            "DK_Features_YYYY grids matched to each field's real survey GPS "
            "with the R5.2.9 SpatialTabularMatcher (K-NN IDW, radius 5 km, "
            "k=5, idw^2); Yield_Proxy_NPP excluded by the matcher "
            "contract"),
        "leading_window_policy": (
            "grid years <= composition survey year only in primary experiments; "
            "2018-2023 full window reported strictly as sensitivity"),
        "no_fabrication": True,
        "no_future_information_in_primary": True,
        "r5_6_r5_7_r5_8_r5_9_artifacts_untouched": True,
        "generated_artifacts": sorted(
            str(f.relative_to(REPO_ROOT)).replace("\\", "/")
            for f in OUT_DIR.iterdir() if f.is_file()),
    }
    write_report_json(contract, "provenance_contract.json")
    return contract


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_phase(phase: str, limit_fields: int | None = None) -> None:
    print(f"[r5.10] phase {phase} ({PHASES[phase]}) starting", flush=True)
    t0 = time.perf_counter()
    if phase == "0":
        phase0_source_schema()
    elif phase == "1":
        phase1_temporal_unit()
    elif phase == "2":
        phase2_extract_grid(limit_fields)
    elif phase == "3":
        phase3_temporal_alignment()
    elif phase == "4":
        phase4_temporal_features()
    elif phase == "5":
        phase5_target()
    elif phase == "6":
        phase6_split()
    elif phase == "7":
        phase7_cheap_baselines()
    elif phase == "8":
        phase8_ablation()
    elif phase == "9":
        phase9_shortcuts()
    elif phase == "10":
        phase10_robustness()
    elif phase == "11":
        base = json.load(open(OUT_DIR / "cheap_baselines.json", encoding="utf-8"))
        abl = json.load(open(OUT_DIR / "ablation_results.json", encoding="utf-8"))
        short = json.load(open(OUT_DIR / "shortcut_audit.json", encoding="utf-8"))
        rob = json.load(open(OUT_DIR / "robustness_results.json", encoding="utf-8"))
        phase11_decision(base, abl, short, rob)
    elif phase == "12":
        d1 = json.load(open(OUT_DIR / "temporal_unit.json", encoding="utf-8"))
        d3 = json.load(open(OUT_DIR / "temporal_alignment.json", encoding="utf-8"))
        d5 = json.load(open(OUT_DIR / "target.json", encoding="utf-8"))
        d6 = json.load(open(OUT_DIR / "split_design.json", encoding="utf-8"))
        d11 = json.load(open(OUT_DIR / "R5.10_decision.json", encoding="utf-8"))
        phase12_report({"d1": d1, "d3": d3, "d5": d5, "d6": d6, "d11": d11})
    elif phase == "13":
        phase13_provenance()
    else:
        raise ValueError(f"unknown phase {phase!r}")
    print(f"[r5.10] phase {phase} done in {time.perf_counter() - t0:.1f}s",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="R5.10 temporal field-target recoverability driver")
    ap.add_argument("--phases", default="all")
    ap.add_argument("--limit-fields", type=int, default=None,
                    help="restrict extraction to N random fields (dev)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = (sorted(PHASES, key=lambda x: int(x))
            if args.phases == "all"
            else [p.strip() for p in args.phases.split(",") if p.strip()])
    for phase in todo:
        run_phase(phase, args.limit_fields)


if __name__ == "__main__":
    main()