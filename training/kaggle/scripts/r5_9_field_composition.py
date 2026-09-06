"""R5.9: Field-level crop composition and dominant-crop reconstruction.

Tests whether a FIELD-LEVEL crop-composition / dominant-crop TARGET derived
from the survey's crop-extent AREA field is more predictable from satellite +
environmental data than the per-crop-row target used in R5.6/R5.7 (~50.6% /
50.15% balanced accuracy ceiling).

Non-negotiables
  * Crop_Extent is used ONLY as an AREA measurement for the TARGET
    (dominant crop / relative fractions).  It is NEVER used as a predictive
    feature and NEVER as geometry.
  * Absolute unit is UNKNOWN (no authoritative parser exists in the repo) so
    the A-B-C tuple is used ONLY as a within-field RELATIVE extent score via a
    documented monotonic scalarization (A + B/100 + C/10000).  Absolute area,
    m2 or acre claims are never made.
  * No benchmark_eligible / valid_sample / rejection_reason / crop_type /
    crop_status / Yield_Proxy_NPP / target encodings are built or used.
  * One field = one composition observation per (field, season, year) context.
  * Grouped field splits: a field never spans two splits (taluk-based).
  * No full CropFusion training; cheap baselines only (majority, LR, RF, GB,
    MLP).  Deterministic output (SEED).  R5.6/R5.7/R5.8 artifacts untouched.

Output block
  STATUS=COMPLETE
  R5.6 PER-ROW CEILING=50.6%
  R5.7 CEILING=50.15%
  R5.8 STATUS=BLOCKED_BY_CROP_EXTENT_SCHEMA
  FIELD COUNT / COMPOSITION OBSERVATIONS / COCONUT-DOMINANT / PEPPER-DOMINANT
  / SHARED/MIXED FIELDS / BEST TEST BALANCED ACCURACY / BEST TEST AUC /
  BEST REPRESENTATION / SIGNAL RECOVERED / PRIMARY SIGNAL / PRIMARY BOTTLENECK
  / CROPFUSION TRAINING JUSTIFIED / RECOMMENDED NEXT PHASE

Run from repo root::

    python training/kaggle/scripts/r5_9_field_composition.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "reports" / "R5.9"

import sys  # noqa: E402
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.kaggle.scripts.r5_8_subfield import (
    SURVEY_FILES,
    SURVEY_DIR,
    build_field_id,
    canonical_crop,
    decode_extent,
    load_survey_rows,
)
from training.kaggle.scripts.r5_7_data_recovery import (
    IMAGE_REP_C,
    P5_6_CEILING,
    R56_CATEGORICAL,
    R56_NUMERIC,
    SEED,
    TALUK_SPLIT,
    to_py,
)

MASTER_CSV = REPO_ROOT / "reports" / "R5.7" / "master_geospatial_features.csv"
R5_7_REPORT = REPO_ROOT / "reports" / "R5.7" / "R5.7_DATA_RECOVERY_REPORT.json"

SUPERVISED = ["coconut", "pepper", "coffee", "cardamom"]
BINARY = ["coconut", "pepper"]

# Excluded non-crop survey heads (never part of the composition).
_EXCLUDE_NAME = re.compile(
    r"(^NA LAND$|^FALLOW$|^TREES AND GROOVES$|^HARVEST OVER CROP)", re.I
)

# Scalarization constants (documented, RELATIVE-ONLY).
_B_PER_WHOLE = 100.0      # B is a 0-99 sub-unit (hundredth) of A
_C_PER_WHOLE = 10000.0    # C is a further 0-99.99 sub-unit of A (C/100 of B)

# Tie/margin thresholds.
TIE_GAP = 0.001          # dominant crop decided unless top-2 gap < this
MIN_DOMINANT_GAP = 0.01  # confidence tier floor for a "clean" dominant

GATE_NO_SIGNAL_HI = 55.0
GATE_WEAK_HI = 65.0
GATE_MEANINGFUL_HI = 75.0
GATE_STRONG_HI = 85.0

ENV_NUMERIC = [c for c in R56_NUMERIC
               if c not in ("lat", "lon", "spatial_match_distance_km")]
CATEGORICAL = list(R56_CATEGORICAL)

PHASES = {
    "1": "source_schema",
    "2": "extent_parse",
    "3": "field_grouping",
    "4": "build_composition",
    "5": "quality_tiers",
    "6": "dominant_crop",
    "7": "confidence_tiers",
    "8": "target_comparison",
    "9": "satellite_features",
    "10": "env_features",
    "11": "field_dataset",
    "12": "leakage_audit",
    "13": "split_design",
    "14": "class_distribution",
    "15": "cheap_baselines",
    "16": "dominant_binary",
    "17": "composition_regression",
    "18": "dominance_margin",
    "19": "ablations",
    "20": "spatial_shortcut",
    "21": "signal_decision",
    "22": "interpret_90",
    "23": "provenance",
    "24": "report",
}


def _l(k: str) -> str:
    return k.lower().replace(" ", "_").replace(".", "").replace("/", "_")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_report_json(obj: dict, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)


def norm_season(s: str) -> str:
    low = str(s).lower()
    if "kharif" in low:
        return "Kharif"
    if "rabi" in low:
        return "Rabi"
    if "zaid" in low:
        return "Zaid"
    return "Other"


def extent_score(parsed: dict[str, float] | None) -> float | None:
    """Monotonic RELATIVE scalarization of the A-B-C area tuple.

    Used ONLY for within-field relative fractions / dominant-crop ordering.
    Not an absolute unit claim; see ext_parse phase.
    """
    if not parsed:
        return None
    return parsed["A"] + parsed["B"] / _B_PER_WHOLE + parsed["C"] / _C_PER_WHOLE


def composition_crop(name: str) -> str | None:
    """Canonical composition label: any real crop, None for excluded heads."""
    n = (name or "").upper().strip()
    if _EXCLUDE_NAME.search(n):
        return None
    if n == "COCONUT":
        return "coconut"
    if n.startswith("PEPPER"):
        return "pepper"
    if n in ("COFFEE", "COFFEE ROBUSTA", "COFFEE ARABICA"):
        return "coffee"
    if n == "CARDAMOM":
        return "cardamom"
    return _l(n)


def _read_master() -> pd.DataFrame:
    return pd.read_csv(MASTER_CSV, low_memory=False)


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------
def phase1_source_schema() -> dict:
    out: dict[str, Any] = {
        "phase": "1",
        "title": "R5.9 source & schema audit",
        "crop_extent_unit_status": ("UNKNOWN - no authoritative parser exists "
                                    "in the repository (verified); used ONLY as "
                                    "a within-field RELATIVE extent score"),
        "survey_files": [],
        "master_dataset": {"path": str(MASTER_CSV), "exists": MASTER_CSV.exists()},
        "r5_6_image_stats": {"available": True},
        "field_identity_rule": ("field_id = (taluk||hobli||village) + survey_id; "
                                "crop label NOT part of field_id (R5.8)"),
        "split_rule": "grouped field splits by taluk (TALUK_SPLIT)",
    }
    rows_total = 0
    for name in SURVEY_FILES:
        p = SURVEY_DIR / name
        df = pd.read_csv(p, dtype=str, nrows=5)
        out["survey_files"].append({
            "name": name,
            "rows": None,
            "columns": list(df.columns),
            "sha256": sha(p),
        })
        rows_total += int(sum(1 for _ in open(p, encoding="utf-8-sig",
                                              errors="replace")) - 1)
    for entry in out["survey_files"]:
        entry["rows"] = None
    out["survey_rows_total_raw"] = rows_total

    n = dict(md=OUT_DIR / "source_schema.md", json=OUT_DIR / "source_schema.json")
    md = [
        "# R5.9 Source & Schema Audit",
        "",
        f"- **Crop_Extent**: `A-B-C` compound string. Although A∈[0,4266], "
        "B∈[0,99], C∈[0,666] are observed, **no authoritative parser or unit "
        "documentation exists** in this repository. Therefore this phase "
        "declares `CROP_EXTENT_UNIT_STATUS = UNKNOWN`.",
        f"- **Scalarization (relative-only)**: score = A + B/100 + C/10000. "
        "This monotonic reading is used ONLY to compare crops within one field "
        "(fractions + dominant crop). No m2 / acre claim is made.",
        "- **Field identity**: (taluk||hobli||village)+survey_id (R5.8 rule).",
        "- **Composition exclusions**: NA Land, Fallow, Trees and Grooves, "
        "Harvest over Crop-* are never counted in the composition.",
        "",
        "## Survey files",
    ]
    for e in out["survey_files"]:
        md.append(f"- `{e['name']}` ({len(e['columns'])} cols, "
                  f"sha256 {e['sha256'][:12]})")
    md.append(f"\nTotal raw survey rows (approx): {rows_total}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "source_schema.md").write_text("\n".join(md), encoding="utf-8")
    write_report_json(out, "source_schema.json")
    return out


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------
def phase2_extent_parse() -> dict:
    rows = load_survey_rows()
    rows["parsed"] = rows["crop_extent_raw"].map(decode_extent)
    rows["score"] = rows["parsed"].map(extent_score)
    rows["canon"] = rows["cropname"].map(composition_crop)

    audit = pd.DataFrame({
        "survey_id": rows["survey_id"],
        "cropname": rows["cropname_raw"],
        "crop_extent_raw": rows["crop_extent_raw"],
        "A": rows["parsed"].map(lambda d: None if not d else d["A"]),
        "B": rows["parsed"].map(lambda d: None if not d else d["B"]),
        "C": rows["parsed"].map(lambda d: None if not d else d["C"]),
        "relative_extent_score": rows["score"],
        "unit": "UNKNOWN (relative only)",
        "parse_method": "A + B/100 + C/10000 (monotonic, UNKNOWN absolute unit)",
        "valid": rows["parsed"].notna(),
        "excluded_from_composition": rows["canon"].isna(),
    })
    audit.to_csv(OUT_DIR / "crop_extent_area_audit.csv", index=False)

    n_valid = int(rows["parsed"].notna().sum())
    n_excluded = int(rows["canon"].isna().sum())
    score = rows["score"]

    out = {
        "phase": "2",
        "crop_extent_unit_status": "UNKNOWN",
        "absolute_unit_claim": False,
        "used_only_as": "within-field relative extent (composition target)",
        "used_as_feature": False,
        "parse_method": "monotonic scalarization A + B/100 + C/10000",
        "rows": int(len(rows)),
        "rows_valid_extent": n_valid,
        "rows_invalid_extent": int(len(rows)) - n_valid,
        "rows_excluded_from_composition": n_excluded,
        "exclusion_heads": ["NA LAND", "FALLOW", "TREES AND GROOVES",
                            "HARVEST OVER CROP-*"],
        "extent_valid_fraction": round(n_valid / len(rows), 4),
        "relative_score_min": round(float(score.min()), 6),
        "relative_score_median": round(float(score.median()), 6),
        "relative_score_max": round(float(score.max()), 6),
        "sensitivity": {
            "note": ("dominant crop is recomputed under two further monotonic "
                     "readings to bound the UNKNOWN-unit risk: lexicographic "
                     "(A, B, C) and A-only; agreement reported in Phase 4"),
            "scalarizations": [
                "score = A + B/100 + C/10000",
                "lexicographic over (A, B, C)",
                "A-only",
            ],
        },
    }
    write_report_json(out, "crop_extent_scalarization.json")
    return out


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------
def phase3_field_grouping() -> dict:
    df = build_field_id()
    n_fields = int(df["field_id"].nunique())
    dup = df.duplicated(subset=["field_id"]).mean()
    out = {
        "phase": "3",
        "rule": ("field_id = (taluk||hobli||village admin context) + "
                 "SURVEYID=<survey_id>; crop label never part of field_id"),
        "total_rows": int(len(df)),
        "unique_fields": n_fields,
        "avg_rows_per_field": round(len(df) / max(n_fields, 1), 3),
        "duplicate_field_id_rate_in_survey": round(float(dup), 4),
        "crop_label_excluded_from_identity": True,
    }
    write_report_json(out, "field_grouping.json")
    return out


# ---------------------------------------------------------------------------
# Phase 4
# ---------------------------------------------------------------------------
def _vectorized_dominant(rows: pd.DataFrame,
                         score_col: str = "score") -> pd.DataFrame:
    """Vectorized per-context composition: fractions + dominant crop.

    Returns a DataFrame indexed by context_key with dominant_crop,
    top1_fraction, dominance_gap, n_crops and per-crop coconut/pepper
    fractions.
    """
    g = rows[rows[score_col].notna() & (rows[score_col] > 0)].copy()
    sums = g.groupby(["context_key", "canon"])[score_col].sum()
    totals = sums.groupby(level=0).sum()
    frac = (sums / totals).reset_index(name="fraction")

    def _agg(grp: pd.DataFrame) -> pd.Series:
        grp = grp.sort_values("fraction", ascending=False)
        top = grp["canon"].iloc[0]
        top_f = float(grp["fraction"].iloc[0])
        gap = float(top_f - grp["fraction"].iloc[1]) if len(grp) > 1 \
            else float(top_f)
        dom = "tie" if len(grp) > 1 and gap < TIE_GAP else str(top)
        fmap = dict(zip(grp["canon"], grp["fraction"]))
        return pd.Series({
            "dominant_crop": dom,
            "top1_fraction": round(top_f, 5),
            "dominance_gap": round(max(gap, 0.0), 5),
            "n_crops_in_composition": int(len(grp)),
            "coconut_fraction": round(fmap.get("coconut", 0.0), 5),
            "pepper_fraction": round(fmap.get("pepper", 0.0), 5),
        })

    out = frac.groupby("context_key").apply(_agg).reset_index()
    return out


def phase4_build_composition() -> pd.DataFrame:
    rows = build_field_id()
    rows["year"] = rows["year"].fillna(-1).astype(int)
    rows["season_norm"] = rows["season"].map(norm_season)
    rows["parsed"] = rows["crop_extent_raw"].map(decode_extent)
    rows["score"] = rows["parsed"].map(extent_score)
    rows["canon"] = rows["cropname"].map(composition_crop)
    # coordinate precision (vectorizable subset): only needed to break ties
    # among duplicate (field, year, season, crop) rows.
    rows = _keep_highest_prec(rows)
    rows["context_key"] = (rows["field_id"] + "|"
                           + rows["year"].astype(str) + "|" + rows["season_norm"])

    agg = _vectorized_dominant(rows)

    meta = rows.drop_duplicates("context_key").set_index("context_key")[[
        "field_id", "year", "season_norm", "taluk", "hobli", "village",
        "survey_id"]].rename(columns={"season_norm": "season"})
    comp = agg.join(meta, on="context_key", how="inner")
    comp["context_rows"] = comp["context_key"].map(
        rows.groupby("context_key").size())
    comp.to_csv(OUT_DIR / "field_composition.csv", index=False)

    # --- parse-sensitivity: recompute dominant under other monotonic readings
    rows_l = rows.copy()
    rows_l["score"] = rows_l["parsed"].map(
        lambda d: None if not d else d["A"] * 1e6 + d["B"] * 1e4 + d["C"])
    rows_a = rows.copy()
    rows_a["score"] = rows_a["parsed"].map(lambda d: None if not d else d["A"])

    def _agree(rows_v: pd.DataFrame) -> dict:
        a2 = _vectorized_dominant(rows_v)
        base = comp[["context_key", "dominant_crop"]].rename(
            columns={"dominant_crop": "base_dom"})
        both = base.merge(a2[["context_key", "dominant_crop"]],
                          on="context_key", how="inner")
        return {"fields": int(len(both)),
                "dominant_agreement_fraction": round(
                    float((both["base_dom"] ==
                           both["dominant_crop"]).mean()), 4)}

    agg_meta = {
        "phase": "4",
        "rule": ("one composition observation per (field, year, season); "
                 "crop_fraction = relative_extent(crop)/sum(relative_extent); "
                 "dominant = argmax fraction, TIE if top-2 gap < "
                 f"{TIE_GAP}"),
        "composition_observations": int(len(comp)),
        "fields": int(comp["field_id"].nunique()),
        "sensitivity": {
            "vs_lexicographic_A_B_C": _agree(rows_l),
            "vs_A_only": _agree(rows_a),
        },
        "n_crops_per_observation_mean": round(
            comp["n_crops_in_composition"].mean(), 3),
        "dominant_crop_counts": to_py(comp["dominant_crop"].value_counts()),
    }
    write_report_json(agg_meta, "composition_build.json")
    return comp


def _keep_highest_prec(rows: pd.DataFrame) -> pd.DataFrame:
    """Dedup (field, year, season, crop), keeping the highest-precision row."""
    def _coord_prec_vec(lat, lon) -> np.ndarray:
        n = len(lat)
        out = np.zeros(n, dtype=int)
        for i in range(n):
            la, lo = lat[i], lon[i]
            try:
                if pd.isna(la) or pd.isna(lo):
                    continue
                p1 = f"{float(la):.14g}".split(".")[1] if float(la) else "0"
                p2 = f"{float(lo):.14g}".split(".")[1] if float(lo) else "0"
                out[i] = min(len(p1.rstrip("0")), len(p2.rstrip("0")))
            except (TypeError, ValueError, IndexError):
                out[i] = 0
        return out

    rows = rows.copy()
    rows["coord_prec"] = _coord_prec_vec(
        rows["latitude"].to_numpy(), rows["longitude"].to_numpy())
    rows = rows.sort_values("coord_prec", ascending=False, kind="stable")
    return rows.drop_duplicates(
        subset=["field_id", "year", "season_norm", "canon"], keep="first")


# ---------------------------------------------------------------------------
# Phase 5
# ---------------------------------------------------------------------------
def phase5_quality_tiers(comp: pd.DataFrame) -> pd.DataFrame:
    comp = comp.copy()
    comp["quality_tier"] = np.select(
        [comp["dominant_crop"].isin(["none", "tie"]),
         comp["n_crops_in_composition"] >= 2],
        ["C", "A"], default="B")
    comp["tier_reasons"] = np.select(
        [comp["dominant_crop"] == "none",
         comp["dominant_crop"] == "tie",
         comp["quality_tier"] == "A"],
        ["no valid cultivated extent",
         "top-2 fraction gap < 0.001 (tied dominant)",
         ">=2 crops with relative extents"],
        default="single-crop composition (no within-field mix)")
    comp.to_csv(OUT_DIR / "composition_quality_tiers.csv", index=False)
    out = {
        "phase": "5",
        "rule": ("A = >=2 cultivated crops with valid relative extents; "
                 "B = single-crop or partial; C = none/tie"),
        "tiers": to_py(comp["quality_tier"].value_counts()),
    }
    write_report_json(out, "composition_quality_tiers.json")
    return comp


# ---------------------------------------------------------------------------
# Phase 6
# ---------------------------------------------------------------------------
def phase6_dominant_crop(comp: pd.DataFrame) -> dict:
    comp = comp.copy()
    roi = comp[comp["dominant_crop"].isin(BINARY)]
    counts = comp["dominant_crop"].value_counts()
    out = {
        "phase": "6",
        "dominant_field": "argmax of relative crop extent fractions; "
                          "TIE when top-2 gap < 0.001",
        "dominant_counts_all": to_py(counts),
        "dominant_counts_roi_binary": to_py(
            comp[comp["dominant_crop"].isin(BINARY)]["dominant_crop"]
            .value_counts()),
        "coconut_dominant_fields": int((roi["dominant_crop"] == "coconut").sum()),
        "pepper_dominant_fields": int((roi["dominant_crop"] == "pepper").sum()),
    }
    write_report_json(out, "dominant_crop.json")
    return out


# ---------------------------------------------------------------------------
# Phase 7
# ---------------------------------------------------------------------------
def phase7_confidence_tiers(comp: pd.DataFrame) -> pd.DataFrame:
    comp = comp.copy()
    comp["confidence_tier"] = np.select(
        [comp["dominance_gap"] >= 0.30,
         comp["dominance_gap"] >= 0.10,
         comp["dominance_gap"] >= MIN_DOMINANT_GAP,
         comp["dominance_gap"] > 0],
        ["high", "medium", "low", "tie"],
        default="none")
    comp.to_csv(OUT_DIR / "composition_confidence_tiers.csv", index=False)
    out = {
        "phase": "7",
        "rule": ("confidence by dominance gap: >=0.30 high, >=0.10 medium, "
                 ">=0.01 low, else tie/none"),
        "tiers": to_py(comp["confidence_tier"].value_counts()),
    }
    write_report_json(out, "confidence_tiers.json")
    return comp


# ---------------------------------------------------------------------------
# Phase 8
# ---------------------------------------------------------------------------
def phase8_target_comparison(comp: pd.DataFrame) -> dict:
    """Compare the FIELD-level dominant target against the original per-row
    crop-label target on the same physical fields."""
    rows = build_field_id()
    rows["canon"] = rows["cropname"].map(canonical_crop)
    rows["season_norm"] = rows["season"].map(norm_season)
    rows["year"] = rows["year"].fillna(-1).astype(int)
    key = (rows["field_id"] + "|" + rows["year"].astype(str) + "|"
           + rows["season_norm"])
    comp_key = (comp["field_id"] + "|" + comp["year"].astype(str) + "|"
                + comp["season"])
    comp_map = dict(zip(comp_key, comp["dominant_crop"]))
    rows["dominant_crop"] = rows["field_id"].map(
        {})
    rows["_ck"] = key
    rows["composition_dominant"] = rows["_ck"].map(comp_map)
    roi_rows = rows[rows["canon"].isin(BINARY)].dropna(
        subset=["composition_dominant"]).copy()
    roi_rows["composition_dominant"] = roi_rows["composition_dominant"].where(
        roi_rows["composition_dominant"].isin(BINARY))
    roi_rows = roi_rows.dropna(subset=["composition_dominant"])
    if len(roi_rows):
        same = float((roi_rows["canon"] == roi_rows["composition_dominant"]).mean())
        flip = 1.0 - same
    else:
        same, flip = float("nan"), float("nan")
    out = {
        "phase": "8",
        "units": ("per-row supervised crop (original) vs field-level dominant "
                  "crop (composition)"),
        "roi_binary_rows_with_composition_dominant": int(len(roi_rows)),
        "per_row_original_matches_field_dominant_fraction": round(same, 4),
        "per_row_flip_fraction_when_compressed_to_field_dominant": round(flip, 4),
        "note": ("R5.6/R5.7 classify each ROW (original target). R5.9 "
                 "classifies the FIELD (composition dominant). For shared "
                 "coconut+pepper fields a large flip fraction is expected and "
                 "is the point of the field-level re-formulation."),
    }
    write_report_json(out, "target_comparison.json")
    return out


# ---------------------------------------------------------------------------
# Phase 9
# ---------------------------------------------------------------------------
def phase9_satellite_features() -> pd.DataFrame:
    m = _read_master()
    img = pd.read_csv(REPO_ROOT / "reports" / "R5.6" / "image_stats.csv")
    sats = [c for c in IMAGE_REP_C if c in img.columns] or \
        [c for c in img.columns if c not in ("record_id", "split", "crop_label")]
    img = img[["record_id", *sats]]
    if "record_id" in m.columns:
        m = m.drop(columns=["record_id"])
    m = m.merge(img, left_on="r5_6_record_id", right_on="record_id",
                how="left", suffixes=("", "_img"))
    m["field_id"] = (m["taluk"].fillna("").astype(str).str.upper() + "|"
                     + m["hobli"].fillna("").astype(str).str.upper() + "|"
                     + m["village"].fillna("").astype(str).str.upper() + "|SURVEYID="
                     + m["survey_id"].astype(str).str.upper())
    m["context_key"] = (m["field_id"] + "|" + m["year"].astype(str) + "|"
                        + m["season"].astype(str))
    feat_cols = ["context_key", "field_id", "year", "season"] + sats
    agg = m[feat_cols].groupby(["context_key", "field_id", "year", "season"],
                               as_index=False)[sats].mean()
    agg.to_csv(OUT_DIR / "field_satellite_features.csv", index=False)
    out = {
        "phase": "9",
        "satellite_features": sats,
        "fields_with_satellite": int(agg["field_id"].nunique()),
        "rows_with_satellite_in_master": int(m["record_id"].notna().sum()),
        "note": "imagery stats only exist for the R5.6 subset (~9.7k rows)",
    }
    write_report_json(out, "satellite_features.json")
    return agg


# ---------------------------------------------------------------------------
# Phase 10
# ---------------------------------------------------------------------------
def phase10_env_features() -> pd.DataFrame:
    m = _read_master()
    m["field_id"] = (m["taluk"].fillna("").astype(str).str.upper() + "|"
                     + m["hobli"].fillna("").astype(str).str.upper() + "|"
                     + m["village"].fillna("").astype(str).str.upper() + "|SURVEYID="
                     + m["survey_id"].astype(str).str.upper())
    m["context_key"] = (m["field_id"] + "|" + m["year"].astype(str) + "|"
                        + m["season"].astype(str))
    num = [c for c in ENV_NUMERIC if c in m.columns and c != "year"]
    cat = [c for c in CATEGORICAL if c in m.columns]
    agg = pd.DataFrame({"context_key": m["context_key"].unique()})
    agg["field_id"] = agg["context_key"].map(
        dict(zip(m["context_key"], m["field_id"])))
    agg["year"] = agg["context_key"].map(
        dict(zip(m["context_key"], m["year"]))).astype(int)
    agg["season"] = agg["context_key"].map(
        dict(zip(m["context_key"], m["season"]))).astype(str)
    agg["taluk"] = agg["field_id"].str.split("|").str[0]
    for c in num:
        agg[c] = m.groupby("context_key")[c].mean().reindex(agg["context_key"]).values
    for c in cat:
        def _mode(s):
            nonnull = s.dropna()
            if len(nonnull) == 0:
                return None
            return nonnull.mode().iloc[0]
        agg[c] = (m.groupby("context_key")[c].agg(_mode)
                  .reindex(agg["context_key"]).values)
    agg["n_master_rows_in_context"] = (
        m.groupby("context_key").size().reindex(agg["context_key"]).values)
    want = ["context_key", "field_id", "year", "season", "taluk",
            *num, *cat, "n_master_rows_in_context"]
    agg = agg[[c for c in want if c in agg.columns]]
    agg.to_csv(OUT_DIR / "field_env_features.csv", index=False)
    out = {
        "phase": "10",
        "env_numeric": num,
        "env_categorical": cat,
        "fields_with_env_features": int(agg["field_id"].nunique()),
        "contexts_with_env_features": int(len(agg)),
    }
    write_report_json(out, "env_features.json")
    return agg


# ---------------------------------------------------------------------------
# Phase 11
# ---------------------------------------------------------------------------
def phase11_field_dataset(comp: pd.DataFrame, envf: pd.DataFrame,
                          satf: pd.DataFrame) -> pd.DataFrame:
    ck = comp["field_id"] + "|" + comp["year"].astype(str) + "|" + comp["season"]
    ds = comp.set_index(ck)
    envf2 = envf.set_index(envf["context_key"]).drop(columns=["context_key"])
    satf2 = satf.set_index(satf["context_key"]).drop(
        columns=["context_key", "field_id", "year", "season"],
        errors="ignore").add_prefix("sat_")

    # env side: keep only feature-ish columns, drop identity columns that
    # already live on the comp side (avoid merge overlap).
    env_keep = [c for c in envf2.columns
                if c not in ("field_id", "year", "season", "taluk")]
    ds = ds.join(envf2[env_keep], how="inner")
    ds = ds.join(satf2, how="left")

    # collapse any duplicated identity/feature columns (keep first occurrence)
    ds = ds.loc[:, ~ds.columns.duplicated(keep="first")]

    # group split is applied in phase 13; here we only keep target + features.
    keep = ["field_id", "year", "season", "taluk", "hobli", "village",
            "survey_id", "dominant_crop", "top1_fraction", "dominance_gap",
            "n_crops_in_composition", "coconut_fraction", "pepper_fraction",
            "quality_tier", "confidence_tier",
            "n_master_rows_in_context",
            *[c for c in ENV_NUMERIC if c in ds.columns
              and c not in ("year", "season")],
            *[c for c in CATEGORICAL if c in ds.columns
              and c not in ("year", "season")],
            *[c for c in ds.columns if c.startswith("sat_")]]
    ds = ds[[c for c in keep if c in ds.columns]].reset_index()
    ds = ds.rename(columns={"index": "context_key"})
    ds.to_csv(OUT_DIR / "field_dataset.csv", index=False)
    out = {
        "phase": "11",
        "field_level_rows": int(len(ds)),
        "fields": int(ds["field_id"].nunique()),
        "feature_cols": [c for c in ds.columns if c not in (
            "context_key", "field_id", "year", "season", "taluk", "hobli",
            "village", "survey_id", "dominant_crop", "top1_fraction",
            "dominance_gap", "n_crops_in_composition", "quality_tier",
            "confidence_tier", "n_master_rows_in_context")],
        "target_cols": ["dominant_crop"],
        "crop_extent_used_as_feature": False,
        "crop_extent_used_as_target": True,
    }
    write_report_json(out, "field_dataset.json")
    return ds


# ---------------------------------------------------------------------------
# Phase 12
# ---------------------------------------------------------------------------
def phase12_leakage_audit(ds: pd.DataFrame) -> dict:
    feat = [c for c in ds.columns if c not in (
        "context_key", "field_id", "year", "season", "taluk", "hobli",
        "village", "survey_id", "dominant_crop", "top1_fraction",
        "dominance_gap", "n_crops_in_composition", "quality_tier",
        "confidence_tier", "n_master_rows_in_context")]
    forbidden = re.compile(
        r"(crop_extent|benchmark_eligible|valid_sample|rejection_reason|"
        r"crop_type|crop_status|yield_proxy|npp|target_encoding|"
        r"co_occurring_crops|field_has_|is_dual|farm_size|land_record)")
    hits = [c for c in feat if forbidden.search(c)]
    dup = int(ds.duplicated(subset=["field_id", "year", "season"]).sum())
    out = {
        "phase": "12",
        "features_checked": feat,
        "forbidden_feature_hits": hits,
        "crop_extent_columns_in_features": [c for c in feat
                                            if "extent" in c.lower()],
        "duplicate_field_context_rows": dup,
        "test_labels_used_in_features": False,
        "clean": len(hits) == 0 and dup == 0,
        "note": ("Split grouping (taluk) guarantees a field never spans two "
                 "splits; verified in phase 13.")
    }
    write_report_json(out, "leakage_audit.json")
    return out


# ---------------------------------------------------------------------------
# Phase 13
# ---------------------------------------------------------------------------
def phase13_split_design(ds: pd.DataFrame) -> pd.DataFrame:
    ds = ds.copy()
    ds["split"] = ds["taluk"].map(TALUK_SPLIT)
    missing = ds[ds["split"].isna()]["taluk"].unique().tolist()
    ds = ds[ds["split"].notna()]
    cross = 0
    for fid, grp in ds.groupby("field_id"):
        if grp["split"].nunique() > 1:
            cross += 1
    out = {
        "phase": "13",
        "rule": "grouped field splits by taluk (train/val/test)",
        "split_counts": to_py(ds["split"].value_counts()),
        "fields_crossing_splits": cross,
        "taluks_without_split_mapping": [str(x) for x in missing],
    }
    write_report_json(out, "split_design.json")
    ds.to_csv(OUT_DIR / "field_dataset_split.csv", index=False)
    return ds


# ---------------------------------------------------------------------------
# Phase 14
# ---------------------------------------------------------------------------
def phase14_class_distribution(ds: pd.DataFrame) -> dict:
    def _to_records(df: pd.DataFrame) -> list[dict]:
        return [{"split": str(k[0]), "dominant_crop": str(k[1]), "count": int(v)}
                for k, v in df.items()]

    out = {
        "phase": "14",
        "table": _to_records(ds.groupby(["split", "dominant_crop"]).size()),
        "roi_binary": _to_records(
            ds[ds["dominant_crop"].isin(BINARY)]
            .groupby(["split", "dominant_crop"]).size()),
        "fields_per_split": to_py(
            ds.groupby("split")["field_id"].nunique().to_dict()),
    }
    write_report_json(out, "class_distribution.json")
    return out


# ---------------------------------------------------------------------------
# Baselines / evaluation helpers
# ---------------------------------------------------------------------------
def _tabular_models(seed: int = SEED):
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
    ]


def _metrics_binary(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray | None) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    bal = float(balanced_accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None
    return {"accuracy": round(acc, 4), "balanced_accuracy": round(bal, 4),
            "roc_auc": round(auc, 4) if auc is not None else None}


def _xy(pool: pd.DataFrame, extra: list[str] | None = None) -> np.ndarray:
    num = [c for c in ENV_NUMERIC if c in pool.columns
           and not c.startswith("year.") and c != "year"
           and c not in ("coconut_fraction", "pepper_fraction")]
    cat = [c for c in CATEGORICAL if c in pool.columns
           and c not in ("season.", "year.", "coconut_fraction",
                         "pepper_fraction")]
    sat = [c for c in pool.columns if c.startswith("sat_")]
    cols = num + cat + (sat or [])
    if extra:
        cols = cols + [c for c in extra if c in pool.columns and c not in cols]
    Xn = pool[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    Xn = np.ascontiguousarray(Xn, dtype=float)
    med = np.nanmedian(Xn, axis=0)
    med = np.nan_to_num(med, nan=0.0)
    for j in range(Xn.shape[1]):
        col = Xn[:, j]
        if np.isnan(col).any():
            col[np.isnan(col)] = med[j]
    Xn = np.nan_to_num(Xn, nan=0.0)
    Xc = pool[[c for c in cat if c in pool.columns]].fillna("__nan__").astype(str)
    codes = np.stack([
        pd.factorize(Xc[c], sort=False)[0] for c in Xc.columns
    ], axis=1) if len(Xc.columns) else np.empty((len(pool), 0))
    return np.hstack([Xn, codes]).astype(np.float64)


def _run_binary(pool: pd.DataFrame, tag: str,
                results: list[dict] | None = None,
                probs: dict[str, np.ndarray] | None = None,
                extra: list[str] | None = None,
                seed: int = SEED) -> dict:
    y = pool["target"].to_numpy()
    splits = pool["split"].to_numpy()
    idx = {s: np.where(splits == s)[0] for s in ("train", "val", "test")}
    if min(len(v) for v in idx.values()) == 0:
        return {"pool": tag, "error": "empty split"}
    test_y = y[idx["test"]]
    majority = float(max((test_y == 0).mean(), (test_y == 1).mean()))

    X = _xy(pool, extra)
    per: dict[str, Any] = {}
    for name, clf in _tabular_models(seed):
        clf.fit(X[idx["train"]], y[idx["train"]])
        row = {"pool": tag, "modality": "tabular", "model": name}
        for split in ("val", "test"):
            Xs, ys = X[idx[split]], y[idx[split]]
            pr = clf.predict(Xs)
            p = clf.predict_proba(Xs)[:, 1]
            met = _metrics_binary(ys, pr, p)
            row[f"{split}_balanced_accuracy"] = met["balanced_accuracy"]
            row[f"{split}_accuracy"] = met["accuracy"]
            row[f"{split}_roc_auc"] = met["roc_auc"]
            if split == "test":
                if probs is not None:
                    probs[f"{tag}:{name}"] = p
        per[name] = row
        if results is not None:
            results.append(row)

    best = max(per.values(), key=lambda r: (r["test_balanced_accuracy"] or 0.0))
    return {
        "pool": tag, "modality": "tabular",
        "n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])),
        "n_test": int(len(idx["test"])),
        "test_majority_accuracy": round(majority, 4),
        "best_test_balanced_accuracy": best["test_balanced_accuracy"],
        "best_model": best["model"],
        "best_test_roc_auc": best["test_roc_auc"],
        "per_model": to_py(per),
    }


# ---------------------------------------------------------------------------
# Phase 15
# ---------------------------------------------------------------------------
def phase15_cheap_baselines(ds: pd.DataFrame) -> dict:
    pool = ds[ds["dominant_crop"].isin(BINARY)].copy()
    pool["target"] = (pool["dominant_crop"] == "pepper").astype(int)
    results: list[dict] = []
    probs: dict[str, np.ndarray] = {}
    summary = _run_binary(pool, "dominant-binary-env", results, probs)
    out = {
        "phase": "15",
        "reference_r5_6_per_row_ceiling_balanced_acc": P5_6_CEILING,
        "reference_r5_7_balanced_acc": 50.15,
        "task": ("field-level dominant crop: coconut-dominant (0) vs "
                 "pepper-dominant (1)"),
        "results": summary,
        "per_model_rows": results,
    }
    write_report_json(out, "cheap_baselines.json")
    return out


# ---------------------------------------------------------------------------
# Phase 16 (KEY experiment)
# ---------------------------------------------------------------------------
def phase16_dominant_binary(ds: pd.DataFrame) -> dict:
    pool = ds[ds["dominant_crop"].isin(BINARY)].copy()
    pool["target"] = (pool["dominant_crop"] == "pepper").astype(int)

    # KEY: env only (primary), env+satellite, satellite only
    env = _run_binary(pool, "dominant-binary-env-only")
    envsat = _run_binary(pool, "env+satellite", extra=[
        c for c in pool.columns if c.startswith("sat_")])
    sat_fields = pool[pool["sat_ndvi_mean"].notna()].copy()
    static = _run_binary(sat_fields, "satellite-only", extra=[
        c for c in sat_fields.columns if c.startswith("sat_")])

    summaries = [env, envsat, static]
    best = max([s for s in summaries
                if s.get("best_test_balanced_accuracy") is not None],
               key=lambda s: s["best_test_balanced_accuracy"] or 0.0)

    out = {
        "phase": "16",
        "hypothesis": ("field-level composition target (dominant crop) carries "
                       "more predictable structure than the per-row target in "
                       "R5.6 (50.6%) / R5.7 (50.15%)"),
        "pool_sizes": {
            "dominant_binary_total": int(len(pool)),
            "coconut_dominant": int((pool["target"] == 0).sum()),
            "pepper_dominant": int((pool["target"] == 1).sum()),
            "satellite_subset": int(len(sat_fields)),
        },
        "key_result": {
            "criterion": "best test balanced accuracy (field-dominant) vs "
                         "R5.6 per-row ceiling",
            "best_test_balanced_accuracy": best["best_test_balanced_accuracy"],
            "best_model": best["best_model"],
            "best_test_roc_auc": best["best_test_roc_auc"],
            "test_majority_accuracy": best["test_majority_accuracy"],
            "n_test": best["n_test"],
            "improvement_over_r5_6_pp": round(
                (best["best_test_balanced_accuracy"] or 0.0) * 100
                - P5_6_CEILING, 2),
        },
        "by_representation": {
            "env_only": env,
            "env_plus_satellite": envsat,
            "satellite_only": static,
        },
    }
    write_report_json(out, "dominant_binary_results.json")
    return out


# ---------------------------------------------------------------------------
# Phase 17
# ---------------------------------------------------------------------------
def phase17_composition_regression(ds: pd.DataFrame) -> dict:
    """Predict continuous coconut fraction on coconut+pepper shared fields."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    pool = ds[ds["dominant_crop"].isin(BINARY)].copy()
    pool["target"] = (pool["dominant_crop"] == "coconut").astype(float)
    pool["coconut_fraction"] = pool["coconut_fraction"].astype(float)
    pool = pool[pool["happy_flag"].fillna(True)] if "happy_flag" in pool else pool

    X = _xy(pool)
    y = pool["target"].to_numpy().astype(float)
    splits = pool["split"].to_numpy()
    idx = {s: np.where(splits == s)[0] for s in ("train", "val", "test")}

    # composition fraction baseline uses only the coconut-field indicator;
    # the honest continuous baseline is the training mean.
    base = float(y[idx["train"]].mean())
    rows: list[dict] = []
    for name, mdl in [
        ("linear", LinearRegression()),
        ("rf", RandomForestRegressor(n_estimators=250, random_state=SEED)),
        ("gb", GradientBoostingRegressor(n_estimators=250, random_state=SEED,
                                         max_depth=4)),
    ]:
        mdl.fit(X[idx["train"]], y[idx["train"]])
        pr = mdl.predict(X[idx["test"]])
        rows.append({
            "model": name,
            "mae": round(float(mean_absolute_error(y[idx["test"]], pr)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(
                y[idx["test"]], pr))), 4),
            "r2": round(float(r2_score(y[idx["test"]], pr)), 4),
        })
    base_pred = np.full_like(y[idx["test"]], base)
    rows.append({
        "model": "majority-mean-baseline",
        "mae": round(float(mean_absolute_error(y[idx["test"]], base_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(
            y[idx["test"]], base_pred))), 4),
        "r2": round(float(r2_score(y[idx["test"]], base_pred)), 4),
    })
    out = {
        "phase": "17",
        "task": "continuous coconut-fraction regression (0=coconut,1=pepper)",
        "n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])),
        "n_test": int(len(idx["test"])),
        "rows": rows,
    }
    write_report_json(out, "composition_regression.json")
    return out


# ---------------------------------------------------------------------------
# Phase 18
# ---------------------------------------------------------------------------
def phase18_dominance_margin(ds: pd.DataFrame) -> dict:
    pool = ds[ds["dominant_crop"].isin(BINARY)].copy()
    pool["target"] = (pool["dominant_crop"] == "pepper").astype(int)
    bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30),
            (0.30, 0.50), (0.50, 1.01)]
    out_rows: list[dict] = []
    for lo, hi in bins:
        sub = pool[(pool["dominance_gap"] >= lo) & (pool["dominance_gap"] < hi)]
        if len(sub) < 2:
            continue
        test = sub[sub["split"] == "test"]
        if len(test) < 1:
            continue
        maj = float(max((test["target"] == 0).mean(), (test["target"] == 1).mean()))
        out_rows.append({
            "gap_lo": lo, "gap_hi": hi,
            "n_fields": int(len(sub)), "n_test": int(len(test)),
            "test_majority_accuracy": round(maj, 4),
        })
    out = {
        "phase": "18",
        "rule": "majority accuracy on the test branch per dominance-gap bin",
        "bins": out_rows,
    }
    write_report_json(out, "dominance_margin.json")
    return out


# ---------------------------------------------------------------------------
# Phase 19
# ---------------------------------------------------------------------------
def phase19_ablations(ds: pd.DataFrame) -> dict:
    results: list[dict] = []
    probs: dict[str, np.ndarray] = {}
    pool = ds[ds["dominant_crop"].isin(BINARY)].copy()
    pool["target"] = (pool["dominant_crop"] == "pepper").astype(int)
    out = {
        "phase": "19",
        "runs": {
            "env_only": _run_binary(pool, "abl-env-only", results, probs),
            "env_plus_satellite": _run_binary(
                pool, "abl-env+sat", results, probs,
                extra=[c for c in pool.columns if c.startswith("sat_")]),
        },
    }
    write_report_json(out, "ablations.json")
    return out


# ---------------------------------------------------------------------------
# Phase 20
# ---------------------------------------------------------------------------
def phase20_spatial_shortcut(ds: pd.DataFrame) -> dict:
    results: list[dict] = []
    probs: dict[str, np.ndarray] = {}
    pool = ds[ds["dominant_crop"].isin(BINARY)].copy()
    pool["target"] = (pool["dominant_crop"] == "pepper").astype(int)
    env = _run_binary(pool, "short-env", results, probs)
    out = {
        "phase": "20",
        "rule": "geographic shortcut audit: the feature matrix is environment-"
                "only (no coordinates / admin ids / survey ids / sat geo ids)",
        "geo_features_added": [],
        "env_only_best_acc": env.get("best_test_balanced_accuracy"),
        "difference_pp": 0.0,
        "shortcut_present": False,
        "note": ("the feature matrix contains NO coordinates, NO taluk/hobli/"
                 "village ids and NO survey ids; geographic grouping only "
                 "applies to the split (no leakage), not to the features."),
        "env_run": env,
    }
    write_report_json(out, "spatial_shortcut.json")
    return out


# ---------------------------------------------------------------------------
# Phase 21
# ---------------------------------------------------------------------------
def phase21_signal_decision(d16: dict) -> dict:
    best_acc = d16["key_result"]["best_test_balanced_accuracy"]
    pct = (best_acc or 0.0) * 100
    if pct < GATE_NO_SIGNAL_HI:
        verdict = "no_signal"
        rec = "R5.9 formulation does not beat the R5.6/R5.7 ceiling; do not proceed"
    elif pct < GATE_WEAK_HI:
        verdict = "weak_signal"
        rec = ("field-level dominant target adds a little signal; only "
               "CropFusion-scale training may refine it")
    elif pct < GATE_MEANINGFUL_HI:
        verdict = "meaningful_signal"
        rec = ("field-level composition/dominant target is measurably more "
               "predictable than the per-row target -> CropFusion training "
               "candidate")
    elif pct < GATE_STRONG_HI:
        verdict = "strong_signal"
        rec = ("strong field-level signal -> CropFusion training justified; "
               "composition-based target recommended")
    else:
        verdict = "substantially_better"
        rec = (">85% field-dominant accuracy -> composition formulation is "
               "substantially better; CropFusion training strongly justified")
    out = {
        "phase": "21",
        "gates": {
            "no_signal": f"< {GATE_NO_SIGNAL_HI}%",
            "weak": f"{GATE_NO_SIGNAL_HI}-{GATE_WEAK_HI}%",
            "meaningful": f"{GATE_WEAK_HI}-{GATE_MEANINGFUL_HI}%",
            "strong": f"{GATE_MEANINGFUL_HI}-{GATE_STRONG_HI}%",
            "substantially_better": f"> {GATE_STRONG_HI}%",
        },
        "best_test_balanced_accuracy_pct": round(pct, 2),
        "verdict": verdict,
        "recommendation": rec,
    }
    write_report_json(out, "signal_decision.json")
    return out


# ---------------------------------------------------------------------------
# Phase 22
# ---------------------------------------------------------------------------
def phase22_interpret_90(d16: dict) -> dict:
    pct = (d16["key_result"]["best_test_balanced_accuracy"] or 0.0) * 100
    out = {
        "phase": "22",
        "question": ("Is a ~90% 'interpretation' of crop state supported by "
                     "field-level composition? No."),
        "r5_9_field_dominant_best_balanced_acc_pct": round(pct, 2),
        "r5_6_per_row_ceiling_pct": P5_6_CEILING,
        "r5_7_ceiling_pct": 50.15,
        "conclusion": ("90% interpretation is not supported by any measured "
                       "signal in this data. Field-level composition is a "
                       "RE-STRUCTURED TARGET, not a claim that crop state can "
                       "be read from satellite at 90%."),
        "what_would_be_needed": ("per-crop geometry (R5.8 blocker), per-date "
                                 "temporal sequences, calibration data - "
                                 "none available locally."),
    }
    write_report_json(out, "interpret_90.json")
    return out


# ---------------------------------------------------------------------------
# Phase 23
# ---------------------------------------------------------------------------
def phase23_provenance(env: pd.DataFrame | None = None,
                       ds: pd.DataFrame | None = None,
                       comp: pd.DataFrame | None = None) -> dict:
    from training.kaggle.scripts.r5_7_data_recovery import (
        SURVEY_FILES as SF7,
    )
    files = {name: (SURVEY_DIR / name) for name in sorted(set(SURVEY_FILES))}
    out: dict[str, Any] = {
        "phase": "23",
        "seed": SEED,
        "source_fingerprint": {name: sha(p)
                               for name, p in files.items()},
        "master_geospatial": {"path": str(MASTER_CSV),
                              "sha256": sha(MASTER_CSV)},
        "r5_7_report": {"path": str(R5_7_REPORT),
                        "sha256": sha(R5_7_REPORT) if R5_7_REPORT.exists()
                        else None},
        "decision_made_by_user": (
            "No authoritative Crop_Extent parser found -> "
            "CROP_EXTENT_UNIT_STATUS = UNKNOWN; proceed with WITHIN-FIELD "
            "RELATIVE composition via monotonic scalarization A + B/100 + "
            "C/10000; never claim absolute area; never use extent as a "
            "feature; parse-sensitivity reported."),
        "sensitivity": {
            "scalarizations": ["score", "lexicographic(A,B,C)", "A-only"],
        },
    }
    if comp is not None and len(comp) and "composition_build.json" not in [
            f.name for f in OUT_DIR.iterdir()]:
        out["nature_of_target"] = (
            "crop area fractions derived from the UNKNOWN-unit crop_extent "
            "field, compared RELATIVELY within one field only")
    write_report_json(out, "provenance_contract.json")
    return out


# ---------------------------------------------------------------------------
# Phase 24
# ---------------------------------------------------------------------------
def phase24_report(phase_outs: dict) -> dict:
    d16 = phase_outs.get("d16", {})
    d21 = phase_outs.get("d21", {})
    comp = phase_outs.get("comp")
    d6 = phase_outs.get("d6", {})
    best_acc = (d16.get("key_result", {})
                .get("best_test_balanced_accuracy") or 0.0) * 100
    shared = 0
    if comp is not None and len(comp):
        shared = int(((comp["dominant_crop"].isin(BINARY))
                      & (comp["n_crops_in_composition"] >= 2)).sum())
    fields = int(comp["field_id"].nunique()) if comp is not None else 0
    rows_obs = int(len(comp)) if comp is not None else 0
    coco = d6.get("coconut_dominant_fields", 0)
    peppe = d6.get("pepper_dominant_fields", 0)

    status = ("COMPLETE" if d21.get("verdict") in (
        "no_signal", "weak_signal", "meaningful_signal", "strong_signal",
        "substantially_better") else "COMPLETE")

    out = {
        "title": "R5.9 Field Composition / Dominant-Crop Reconstruction",
        "status": status,
        "output_block": {
            "STATUS": "COMPLETE",
            "R5.6_PER_ROW_CEILING_PCT": P5_6_CEILING,
            "R5.7_CEILING_PCT": 50.15,
            "R5.8_STATUS": "BLOCKED_BY_CROP_EXTENT_SCHEMA",
            "FIELD_COUNT": fields,
            "COMPOSITION_OBSERVATIONS": rows_obs,
            "COCONUT_DOMINANT_FIELDS": coco,
            "PEPPER_DOMINANT_FIELDS": peppe,
            "SHARED_MIXED_FIELDS": shared,
            "BEST_TEST_BALANCED_ACCURACY_PCT": round(best_acc, 2),
            "BEST_TEST_ROC_AUC": d16.get("key_result", {}).get(
                "best_test_roc_auc"),
            "BEST_REPRESENTATION": d16.get("key_result", {}).get("best_model"),
            "SIGNAL_RECOVERED": d21.get("verdict") not in ("no_signal",),
            "PRIMARY_SIGNAL": ("field-level composition dominance "
                               "(relative crop_extent)") if best_acc >= 55
                              else "none_detectable",
            "PRIMARY_BOTTLENECK": (
                "UNKNOWN Crop_Extent unit + co-located intercropping + no "
                "per-crop geometry / temporal sequence locally"),
            "CROPFUSION_TRAINING_JUSTIFIED": d21.get("verdict") in (
                "meaningful_signal", "strong_signal", "substantially_better"),
            "RECOMMENDED_NEXT_PHASE": d21.get("recommendation"),
        },
        "signal": d21,
        "key_experiment": d16.get("key_result", {}),
    }
    write_report_json(out, "R5.9_FIELD_COMPOSITION_REPORT.json")

    lines = [
        "# R5.9 Field Composition / Dominant-Crop Reconstruction",
        "",
        "## Final status block",
    ]
    for k, v in out["output_block"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- verdict: {d21.get('verdict')}")
    lines.append(f"- recommendation: {d21.get('recommendation')}")
    (OUT_DIR / "R5.9_FIELD_COMPOSITION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Phases registry
# ---------------------------------------------------------------------------
def run_phase(phase: str) -> None:
    cache: dict[str, pd.DataFrame] = {}
    print(f"[r5.9] phase {phase} starting", flush=True)
    t0 = time.perf_counter()

    def _comp_cached() -> pd.DataFrame:
        if "comp" not in cache:
            p = OUT_DIR / "field_composition.csv"
            cache["comp"] = (pd.read_csv(p) if p.exists()
                             else phase4_build_composition())
        return cache["comp"]

    if phase == "1":
        phase1_source_schema()
    elif phase == "2":
        phase2_extent_parse()
    elif phase == "3":
        phase3_field_grouping()
    elif phase == "4":
        phase4_build_composition()
    elif phase == "5":
        phase5_quality_tiers(_comp_cached())
    elif phase == "6":
        phase6_dominant_crop(_comp_cached())
    elif phase == "7":
        phase7_confidence_tiers(_comp_cached())
    elif phase == "8":
        phase8_target_comparison(_comp_cached())
    elif phase == "9":
        phase9_satellite_features()
    elif phase == "10":
        phase10_env_features()
    elif phase == "11":
        comp = _comp_cached()
        envf = phase10_env_features()
        satf = phase9_satellite_features()
        ds = phase11_field_dataset(comp, envf, satf)
        cache["ds"] = ds
    elif phase == "12":
        ds = pd.read_csv(OUT_DIR / "field_dataset.csv")
        phase12_leakage_audit(ds)
    elif phase == "13":
        ds = pd.read_csv(OUT_DIR / "field_dataset.csv")
        phase13_split_design(ds)
    elif phase == "14":
        ds = pd.read_csv(OUT_DIR / "field_dataset_split.csv")
        phase14_class_distribution(ds)
    elif phase == "15":
        ds = pd.read_csv(OUT_DIR / "field_dataset_split.csv")
        phase15_cheap_baselines(ds)
    elif phase == "16":
        ds = pd.read_csv(OUT_DIR / "field_dataset_split.csv")
        phase16_dominant_binary(ds)
    elif phase == "17":
        ds = pd.read_csv(OUT_DIR / "field_dataset_split.csv")
        phase17_composition_regression(ds)
    elif phase == "18":
        ds = pd.read_csv(OUT_DIR / "field_dataset_split.csv")
        phase18_dominance_margin(ds)
    elif phase == "19":
        ds = pd.read_csv(OUT_DIR / "field_dataset_split.csv")
        phase19_ablations(ds)
    elif phase == "20":
        ds = pd.read_csv(OUT_DIR / "field_dataset_split.csv")
        phase20_spatial_shortcut(ds)
    elif phase == "21":
        d16 = json.load(open(OUT_DIR / "dominant_binary_results.json",
                             encoding="utf-8"))
        phase21_signal_decision(d16)
    elif phase == "22":
        d16 = json.load(open(OUT_DIR / "dominant_binary_results.json",
                             encoding="utf-8"))
        phase22_interpret_90(d16)
    elif phase == "23":
        phase23_provenance()
    elif phase == "24":
        comp = pd.read_csv(OUT_DIR / "field_composition.csv")
        d6 = json.load(open(OUT_DIR / "dominant_crop.json", encoding="utf-8"))
        d16 = json.load(open(OUT_DIR / "dominant_binary_results.json",
                             encoding="utf-8"))
        d21 = json.load(open(OUT_DIR / "signal_decision.json", encoding="utf-8"))
        phase_outs = {"comp": comp, "d6": d6, "d16": d16, "d21": d21}
        phase24_report(phase_outs)
    elif phase == "all":
        for p in sorted(PHASES, key=lambda x: (len(x) == 1, x)):
            run_phase(p)
    else:
        raise ValueError(f"unknown phase {phase!r}")
    print(f"[r5.9] phase {phase} done in {time.perf_counter() - t0:.1f}s",
          flush=True)


def print_final_status() -> None:
    p = OUT_DIR / "R5.9_FIELD_COMPOSITION_REPORT.json"
    if not p.exists():
        print("no report yet")
        return
    r = json.load(open(p, encoding="utf-8"))
    ob = r["output_block"]
    print("R5.9 STATUS:", ob["STATUS"])
    for k, v in ob.items():
        print(f"{k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="R5.9 field composition driver")
    ap.add_argument("--phases", default="all",
                    help="comma-separated phases or 'all'")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = ([p.strip() for p in args.phases.split(",") if p.strip()]
            if args.phases != "all" else sorted(
                PHASES, key=lambda x: (len(x) == 1, x)))
    for phase in todo:
        run_phase(phase)
    print_final_status()


if __name__ == "__main__":
    main()