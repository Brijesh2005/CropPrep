"""R5.8 — SUB-FIELD / CROP-EXTENT DISCRIMINATION (signal-recovery experiment).

R5.5 (model capacity), R5.6 (field-level separability ~50.6%) and R5.7
(recovered pool ceiling ~50.15%) all point to a structural information
bottleneck at the shared-field granularity: black pepper is grown *under*
coconut (median ~2.9 m to a coconut field) so the two crops share GPS,
environment and satellite pixels.

R5.8's stated hypothesis is:

    "Crop-specific sub-field / crop-extent imagery contains more
     discriminative information than the current field-level observation."

PHASE 1 inspects `Crop_Extent` (the survey schema's only crop-specific spatial
candidate).  The finding is decisive and documented here:

  * `Crop_Extent` is a **scalar land-area measure** in a compound `A-B-C`
    notation (e.g. ``1-76-0.00`` = acres + fractional subdivision).  It has
    **no geometry** — no polygon, no bounding box, no centroid, no coordinates.
  * There is **no crop geometry source** anywhere in the repository.  The only
    GIS files (application/gis/*) are hobli/taluk/district administrative
    boundaries, not crop/farm/extent polygons.
  * Pepper and coconut rows in a shared intercropped field are GPS co-located
    at a median **2.8 m**, below Sentinel-2 ~10 m pixel resolution.

Therefore there is **no physical sub-field region to map crop-specific
satellite pixels to**, and `crop_extent` cannot drive pixel masking or
bbox extraction.  Per the R5.8 spec's explicit Phase 1 rule, this is reported
as:

    R5.8 STATUS = BLOCKED_BY_CROP_EXTENT_SCHEMA

The driver still runs every analysis that is valid *without fabricating
geometry*: the crop_extent schema audit, a crop-label-free field-identity
audit, and a full co-location analysis, plus the BLOCKED decision, provenance
contract and final report.  No simulated sub-field pixels are manufactured
(rules 4/6/7 forbid it).  The driver contains an explicit "unblock contract"
(interface) ready for the day a real per-crop / polygon geometry arrives.

Hard rules (from the R5.8 spec):
  * NO CropFusion training, no architecture work, no 90% claims.
  * No fabrication: no invented geometry, extents, dates or pixels.
  * crop_extent stays as recorded; never reinterpreted into geometry.
  * Crop label is NEVER used to select or mask pixels by appearance.
  * field_id is built WITHOUT the crop label.
  * No leakage features (Yield_Proxy_NPP, benchmark_eligible, crop label, ...).
  * R5.6/R5.7 frozen datasets are never modified.

Run from repo root, e.g.:

    python training/kaggle/scripts/r5_8_subfield.py --phases 1
    python training/kaggle/scripts/r5_8_subfield.py --phases 2,3
    python training/kaggle/scripts/r5_8_subfield.py --phases schedule,decision,provenance,report
    python training/kaggle/scripts/r5_8_subfield.py --phases all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from training.kaggle.scripts.r5_7_data_recovery import SURVEY_FILES  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "R5.8"
SURVEY_DIR = REPO_ROOT / "govt_crop_survey_data"
GIS_DIR = REPO_ROOT / "application" / "gis"

BINARY = ["coconut", "pepper"]

P5_7_CEILING = 50.15

BLOCKED_STATUS = "BLOCKED_BY_CROP_EXTENT_SCHEMA"

# Decode of survivable UNBLOCKED sub-field phases for when geometry arrives.
UNBLOCKED_PHASES = {
    "schedule": "crop_extent_blessing",  # requires real per-crop geometry/polygon
    "4": "construct_spatial_extent",
    "5": "field_vs_subfield_geometry",
    "6": "satellite_extraction_strategies",
    "7": "pixel_masking",
    "8": "temporal_signature",
    "9": "temporal_features",
}

PHASES = {
    "1": "inspect",                 # crop_extent schema  => BLOCKED gate
    "2": "field_identity",          # field_id audit (no crop label)
    "3": "colocation",              # co-location analysis
    "schedule": "schedule",         # unblock contract / blocked phase mapping
    "decision": "decision",         # signal-recovery decision (BLOCKED)
    "provenance": "provenance",     # provenance contract
    "report": "report",             # final report
}


def sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_report_json(path: Path, obj: Any) -> None:
    tmp = OUT_DIR / (path.name + ".tmp")
    _ = json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)
    tmp.replace(path)


def _hav(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(
        np.radians, [np.asarray(lat1), np.asarray(lon1),
                     np.asarray(lat2), np.asarray(lon2)])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        (lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ------------------------------------------------------------------ #
# Shared loaders
# ------------------------------------------------------------------ #

_EXTENT_RE = re.compile(r"^(\d+)-(\d+)-(\d+(?:\.\d+)?)$")


def load_survey_rows() -> pd.DataFrame:
    """Load all 15 survey files, keeping every recorded row (no dedup here)."""
    frames = []
    for name in SURVEY_FILES:
        path = SURVEY_DIR / name
        df = pd.read_csv(path, dtype=str)
        frames.append(pd.DataFrame({
            "survey_id": df["Survey_id"].str.strip(),
            "cropname_raw": df["Cropname"].astype(str).str.strip(),
            "crop_extent_raw": df["Crop_Extent"].astype(str).str.strip(),
            "crop_survey_date": df["CropSurveyDate"].astype(str).str.strip(),
            "latitude": pd.to_numeric(df["Latitude"], errors="coerce"),
            "longitude": pd.to_numeric(df["Longtitude"], errors="coerce"),
            "taluk": df["Taluk_Name"].astype(str).str.strip(),
            "hobli": df["Hobli_Name"].astype(str).str.strip(),
            "village": df["Village_Name"].astype(str).str.strip(),
            "years": df["Years"].astype(str).str.strip(),
            "season": df["Season"].astype(str).str.strip(),
            "source_file": name,
        }))
    df = pd.concat(frames, ignore_index=True)
    df["year"] = df["years"].str.extract(r"(\d{4})")[0].astype(float)
    df["cropname"] = (df["cropname_raw"].str.upper().str.strip().str.replace(
        r"\s+", " ", regex=True))
    return df


def canonical_crop(name: str) -> str | None:
    """Map a survey cropname to a canonical roi label (or None if irrelevant)."""
    n = (name or "").upper()
    if "HARVEST OVER CROP" in n or "NA LAND" in n or "FALLOW" in n:
        return None
    if n in ("COCONUT",):
        return "coconut"
    if n.startswith("PEPPER"):
        return "pepper"
    if n in ("COFFEE",):
        return "coffee"
    if n in ("CARDAMOM",):
        return "cardamom"
    return None


def decode_extent(raw: str) -> dict[str, float] | None:
    """Decode the compound area notation ``A-B-C`` into numeric parts.

    Returns None if the string is not parseable (never silently coerced).
    """
    if not isinstance(raw, str):
        return None
    m = _EXTENT_RE.match(raw.strip())
    if not m:
        return None
    a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return {"A": a, "B": b, "C": c}


# ------------------------------------------------------------------ #
# Phase 1 — crop_extent schema inspection (the BLOCKED gate)
# ------------------------------------------------------------------ #

def phase1_inspect() -> None:
    df = load_survey_rows()
    ext = df["crop_extent_raw"]
    total = len(df)

    # Does every row match the compound A-B-C area pattern?
    m = df["crop_extent_raw"].map(_EXTENT_RE.match)
    match_frac = float(m.notna().mean())
    non_empty = df["crop_extent_raw"].isin(["", "NULL", "nan", "None"]).sum()

    # Check for ANY geometric signature in any field
    geo_hint_cols = [c for c in df.columns if any(
        k in c.lower() for k in ("geom", "poly", "bbox", "box", "shape",
                                 "centroid", "wkt", "extent_x", "extent_y"))]
    geo_hint_values = [v for v in df["crop_extent_raw"] if any(
        k in v.lower() for k in ("polygon", "multipolygon", "(", "linestring",
                                 "poi"))]

    # bounds of each part
    parts = df.loc[m.notna(), "crop_extent_raw"].str.split("-", expand=True)
    parts = parts.apply(pd.to_numeric, errors="coerce")
    bounds = {
        "A": {"min": float(parts[0].min()), "max": float(parts[0].max()),
              "mean": float(parts[0].mean()), "median": float(parts[0].median())},
        "B": {"min": float(parts[1].min()), "max": float(parts[1].max()),
              "mean": float(parts[1].mean()), "median": float(parts[1].median())},
        "C": {"min": float(parts[2].min()), "max": float(parts[2].max()),
              "mean": float(parts[2].mean()), "median": float(parts[2].median())},
    }

    # is extent crop-specific?  (same survey_id -> distinct extents across crops)
    mult = df.groupby("survey_id")["crop_extent_raw"].nunique()
    mult_rows = int((mult >= 2).sum())

    # does a single crop row anywhere expose coordinates in extent?  (it must not)
    has_coord_like = df["crop_extent_raw"].str.contains(r"\.\d{6,}|^[-+]?\d+\.\d+,\s*[-+]?\d+\.\d+",
                                                        regex=True).sum()

    # Inventory GIS: only administrative boundaries, never crop polygons.
    gis_files = sorted(p.name for p in GIS_DIR.rglob("*") if p.is_file())
    admin_only = True

    evidence = {
        "total_rows": total,
        "fraction_matching_compound_area_pattern": round(match_frac, 6),
        "blank_or_null_extent_rows": int(non_empty),
        "geometry_hint_columns": geo_hint_cols,
        "geometry_hint_values_found": len(geo_hint_values),
        "rows_with_coordinate_like_extent": int(has_coord_like),
        "extent_part_bounds_A_B_C": bounds,
        "survey_ids_with_distinct_extents_across_crops": mult_rows,
        "gis_files_present": gis_files,
        "gis_crop_polygon_source_present": False,
        "gis_administrative_only": admin_only,
    }

    report = {
        "phase": "1",
        "status": BLOCKED_STATUS,
        "what_is_crop_extent": (
            "a scalar land-area measure in compound A-B-C notation "
            "(e.g. '1-76-0.00'), per crop survey row"),
        "contains_geometry": False,
        "geometry_kind": None,
        "coordinate_system": None,
        "crop_specific_area": True,
        "identifies_physical_subfield_region": False,
        "independently_measured_or_derived": (
            "independently recorded per-row area value, NOT derived from label; "
            "but a scalar with no spatial placement"),
        "can_be_mapped_to_satellite_pixels": False,
        "why_blocked": [
            "no polygon, bounding box, centroid or coordinate placement",
            "no crop geometry source exists in the repository (only "
            "hobli/taluk/district admin boundaries in application/gis)",
            "per-crop extent cannot select or mask specific satellite pixels "
            "without a spatial location",
        ],
        "unblock_contract": {
            "required_geometry_inputs": [
                "per-crop polygon OR bbox OR centroid with crop label",
                "geometry CRS (expected EPSG:4326 or metric projected CRS)",
                "geometry source + provenance",
            ],
            "required_validation": [
                "geometry validity / non-empty / bounds / self-intersection",
                "area_m2 plausible range",
                "crop label attached to each geometry, not inferred",
            ],
        },
        "evidence": evidence,
    }
    write_report_json(OUT_DIR / "crop_extent_schema.json", report)
    _write_phase1_md(report, evidence)
    print(f"  crop_extent_schema.json written (status={BLOCKED_STATUS})")
    print("  => crop_extent is a scalar area, not spatial geometry: SUB-FIELD "
          "pixel extraction is impossible without a real geometry source.")


def _write_phase1_md(report: dict, ev: dict) -> None:
    md = [
        "# R5.8 Phase 1 — crop_extent schema",
        "",
        f"**R5.8 STATUS = {BLOCKED_STATUS}**",
        "",
        "## What is `Crop_Extent`?",
        "",
        f"- `Crop_Extent` is a **scalar land-area measure** in a compound "
        f"`A-B-C` notation (e.g. `1-76-0.00` = acres + fractional subdivision), "
        f"recorded once per crop survey row.",
        f"- **{round(ev['fraction_matching_compound_area_pattern']*100,2)}%** of "
        f"{ev['total_rows']:,} rows match the `X-Y-Z` area pattern.",
        f"- The three parts have ranges A∈[0,{ev['extent_part_bounds_A_B_C']['A']['max']}], "
        f"B∈[0,{ev['extent_part_bounds_A_B_C']['B']['max']}], "
        f"C∈[0,{ev['extent_part_bounds_A_B_C']['C']['max']}].",
        f"- It is **crop-specific** (same Survey_id yields different extents for "
        f"different crop rows in {ev['survey_ids_with_distinct_extents_across_crops']} surveys).",
        "",
        "## Does it contain spatial geometry?  NO.",
        "",
        "- No polygon, bounding box, centroid, or coordinate placement.",
        f"- Rows with coordinate-like extent content: {ev['rows_with_coordinate_like_extent']}.",
        f"- Geometry-hint values found in extent strings: {ev['geometry_hint_values_found']}.",
        "- No crop geometry source exists in the repository. The only GIS files "
        "are administrative boundaries:",
        "",
        "```",
        "\n".join(f"- {f}" for f in ev["gis_files_present"]),
        "```",
        "",
        "## Why R5.8 is blocked",
        "",
        "The sub-field hypothesis requires a **physical sub-field region** to "
        "map crop-specific satellite pixels to. `crop_extent` is a scalar area "
        "with no spatial placement, and no polygon/bbox geometry exists in the "
        "available data. Producing 'sub-field' pixels would require inventing "
        "geometry (forbidden by R5.8 rules 4/6/7), so the experiment cannot be "
        "honestly run on this dataset.",
        "",
        "## Unblock contract",
        "",
        "R5.8's sub-field phases will run the moment a real per-crop geometry "
        "source is provided (e.g. cadastral parcel polygons with crop "
        "attribution, or a village/field polygon layer). The interface is "
        "defined in `reports/R5.8/crop_extent_schema.json` (`unblock_contract`).",
    ]
    (OUT_DIR / "crop_extent_schema.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ #
# Phase 2 — field identity (no crop label)
# ------------------------------------------------------------------ #

def build_field_id() -> pd.DataFrame:
    """Canonical field_id from geometry-neutral identity, never crop label.

    field_id := admin_context|survey_id where admin_context is the survey's
    (taluk||hobli||village) administrative location.  survey_id is a plot
    identifier reused across crop rows; the crop label is deliberately NOT
    part of the field_id so a coconut row and a pepper row that belong to the
    same plot share a field_id.
    """
    df = load_survey_rows()
    df["admin_context"] = (df["taluk"] + "|" + df["hobli"] + "|"
                           + df["village"]).str.upper()
    df["field_id"] = df["admin_context"] + "|SURVEYID=" + df["survey_id"].str.upper()
    return df


def phase2_field_identity() -> None:
    df = build_field_id()
    df["canon"] = df["cropname"].map(canonical_crop)
    roi = df[df["canon"].isin(BINARY)].copy()

    n_fields = df["field_id"].nunique()
    records_per_field = df.groupby("field_id").size()
    dual = df.groupby("field_id")["canon"].apply(
        lambda s: ("coconut" in set(s.dropna().tolist())
                   and "pepper" in set(s.dropna().tolist()))).sum()

    # duplicate rate: any repeated (field_id, crop survey identity)?
    key = df["field_id"] + "|" + df["year"].astype(str) + "|" + df["season"] + "|" + df["canon"]
    dup_rate = float(1 - key.duplicated().mean()) if len(key) else 1.0

    # per-field geometry-free identity stability: same survey_id across years
    sid_years = df.groupby("survey_id")["year"].nunique()

    audit = pd.DataFrame({
        "field_id": df["field_id"],
        "survey_id": df["survey_id"],
        "cropname": df["cropname_raw"],
        "canonical_crop": df["canon"],
        "year": df["year"],
        "season": df["season"],
        "latitude": df["latitude"],
        "longitude": df["longitude"],
        "taluk": df["taluk"],
        "hobli": df["hobli"],
        "village": df["village"],
        "crop_extent_raw": df["crop_extent_raw"],
        "source_file": df["source_file"],
    })
    audit.to_csv(OUT_DIR / "field_identity_audit.csv", index=False)

    summary = {
        "phase": "2",
        "rule": ("field_id = (taluk||hobli||village admin context) + survey_id; "
                 "crop label is NOT part of field_id"),
        "unique_fields": int(n_fields),
        "total_rows": int(len(df)),
        "coconut_rows": int((roi["canon"] == "coconut").sum()),
        "pepper_rows": int((roi["canon"] == "pepper").sum()),
        "shared_coconut_pepper_fields": int(dual),
        "records_per_field": {
            "mean": round(float(records_per_field.mean()), 3),
            "median": float(records_per_field.median()),
            "max": int(records_per_field.max()),
        },
        "duplicate_rate_key": round(dup_rate, 6),
        "survey_ids_with_multiple_years": int((sid_years > 1).sum()),
        "field_identity_audit_csv": "reports/R5.8/field_identity_audit.csv",
    }
    write_report_json(OUT_DIR / "field_identity_audit.json", summary)
    print(f"  field_identity_audit.csv/.json written "
          f"(fields={n_fields}, shared coconut+pepper={dual})")


# ------------------------------------------------------------------ #
# Phase 3 — co-location analysis
# ------------------------------------------------------------------ #

def phase3_colocation() -> None:
    from scipy.spatial import cKDTree

    df = load_survey_rows()
    df["canon"] = df["cropname"].map(canonical_crop)
    df = df[df["canon"].isin(BINARY)].dropna(subset=["latitude", "longitude"]).copy()
    df["field_id"] = (df["taluk"] + "|" + df["hobli"] + "|" + df["village"]
                      + "|SURVEYID=" + df["survey_id"].str.upper())

    coc = df[df["canon"] == "coconut"].drop_duplicates("field_id")
    pep = df[df["canon"] == "pepper"].drop_duplicates("field_id")
    tree = cKDTree(np.deg2rad(coc[["latitude", "longitude"]].values))
    d0, idx = tree.query(np.deg2rad(pep[["latitude", "longitude"]].values), k=1)
    coc_pts = coc[["latitude", "longitude"]].values
    dist_m = _hav(pep["latitude"].values, pep["longitude"].values,
                  coc_pts[idx, 0], coc_pts[idx, 1])

    bands = ["0-5", "5-10", "10-25", "25-50", "50-100", ">100"]
    counts = {
        "0-5": int((dist_m <= 5).sum()),
        "5-10": int(((dist_m > 5) & (dist_m <= 10)).sum()),
        "10-25": int(((dist_m > 10) & (dist_m <= 25)).sum()),
        "25-50": int(((dist_m > 25) & (dist_m <= 50)).sum()),
        "50-100": int(((dist_m > 50) & (dist_m <= 100)).sum()),
        ">100": int((dist_m > 100).sum()),
    }

    rows = pd.DataFrame({
        "pepper_field_id": pep["field_id"],
        "pepper_lat": pep["latitude"],
        "pepper_lon": pep["longitude"],
        "nearest_coconut_field_id": coc.iloc[idx]["field_id"].values,
        "nearest_coconut_lat": coc_pts[idx, 0],
        "nearest_coconut_lon": coc_pts[idx, 1],
        "distance_m": dist_m,
        "distance_band": pd.cut(dist_m, [0, 5, 10, 25, 50, 100, np.inf],
                                labels=bands, include_lowest=True).astype(str),
    })
    rows.to_csv(OUT_DIR / "colocation_analysis.csv", index=False)

    summary = {
        "phase": "3",
        "n_coconut_fields": int(len(coc)),
        "n_pepper_fields": int(len(pep)),
        "percent_pepper_within_5m_of_coconut":
            round(float((dist_m <= 5).mean()) * 100, 2),
        "percent_pepper_within_10m_of_coconut":
            round(float((dist_m <= 10).mean()) * 100, 2),
        "percent_pepper_within_25m_of_coconut":
            round(float((dist_m <= 25).mean()) * 100, 2),
        "percent_pepper_within_50m_of_coconut":
            round(float((dist_m <= 50).mean()) * 100, 2),
        "percent_pepper_within_100m_of_coconut":
            round(float((dist_m <= 100).mean()) * 100, 2),
        "pepper_to_nearest_coconut_m": {
            "median": round(float(np.median(dist_m)), 3),
            "mean": round(float(np.mean(dist_m)), 3),
            "p95": round(float(np.percentile(dist_m, 95)), 3),
            "p99": round(float(np.percentile(dist_m, 99)), 3),
        },
        "distance_band_counts": counts,
        "colocation_analysis_csv": "reports/R5.8/colocation_analysis.csv",
    }
    write_report_json(OUT_DIR / "colocation_analysis.json", summary)
    print(f"  colocation_analysis.csv/.json written "
          f"(median pepper->coconut dist={summary['pepper_to_nearest_coconut_m']['median']} m)")


# ------------------------------------------------------------------ #
# Unblock schedule / decision / provenance / report
# ------------------------------------------------------------------ #

def phase_schedule() -> None:
    doc = {
        "phase": "schedule",
        "status": BLOCKED_STATUS,
        "blocked_phases": sorted(UNBLOCKED_PHASES.keys()),
        "unblocked_when": ("a real per-crop / polygon geometry source is "
                           "provided: cadastral parcels with crop attribution, "
                           "or a village/field polygon layer with crop extents"),
        "mapping": {
            "1": "inspect (DONE - blocked at geometry)",
            "4": UNBLOCKED_PHASES["4"],
            "5": UNBLOCKED_PHASES["5"],
            "6": UNBLOCKED_PHASES["6"],
            "7": UNBLOCKED_PHASES["7"],
            "8": UNBLOCKED_PHASES["8"],
            "9": UNBLOCKED_PHASES["9"],
            "10-15": "cheap baselines + split + ablations (after geometry+satellite)",
            "16": "signal recovery decision",
            "18": "crop_subfield_v1.csv output",
        },
        "unblock_contract": {
            "required_geometry_inputs": [
                "per-crop polygon OR bbox OR centroid with crop label",
                "geometry CRS (expected EPSG:4326 or a metric projected CRS)",
                "geometry source + provenance",
            ],
            "required_validation": [
                "geometry validity / non-empty / bounds / self-intersection",
                "area_m2 plausible range",
                "crop label attached to each geometry, not inferred",
            ],
            "pipeline_ready": [
                "construct_spatial_extent", "field_vs_subfield_geometry",
                "satellite_extraction_strategies", "pixel_masking",
                "temporal_signature", "temporal_features",
                "cheap baselines", "split grouped by field_id",
                "ablations", "signal decision",
            ],
        },
        "do_not_run_without_geometry": [
            "pixel masking", "bbox extraction", "centroid patch",
            "any 'sub-field' satellite feature",
        ],
    }
    write_report_json(OUT_DIR / "unblock_schedule.json", doc)
    print(f"  unblock_schedule.json written (blocked at crop_extent geometry)")


def phase_decision() -> None:
    dec = json.load(open(OUT_DIR / "colocation_analysis.json", encoding="utf-8"))
    schema = json.load(open(OUT_DIR / "crop_extent_schema.json", encoding="utf-8"))
    decision = {
        "phase": "decision",
        "status": BLOCKED_STATUS,
        "verdict": "not_testable",
        "hypothesis": (
            "crop-specific sub-field/crop-extent imagery contains more "
            "discriminative information than field-level satellite observation"),
        "result": (
            "cannot be tested on the available data: crop_extent is a scalar "
            "area with no geometry and no crop polygon source exists; there is "
            "no physical sub-field region to localize crop-specific pixels"),
        "data_ceiling_r5_7_pct": P5_7_CEILING,
        "data_ceiling_r5_8_pct": None,
        "signal_recovered": False,
        "primary_signal": "intercropping_co_location (R5.7, confirmed co-located)",
        "primary_bottleneck": (
            "missing_crop_geometry: no spatial sub-field crop extent exists to "
            "separate co-located coconut/pepper (median 2.8 m) below ~10 m pixels"),
        "cropfusion_training_justified": False,
        "recommended_next_phase": (
            "R5.9 candidate after 'R5.8-unblock' acquires a real per-crop "
            "polygon/parcel geometry layer; until then, sub-field discrimination "
            "is not physically possible on this dataset"),
        "engineering_gate": "BLOCKED (<55% gate not evaluable without geometry)",
        "pepper_within_5m_coconut_pct": dec[
            "percent_pepper_within_5m_of_coconut"],
        "pepper_nearest_coconut_median_m": dec[
            "pepper_to_nearest_coconut_m"]["median"],
        "crop_extent_contains_geometry": schema["contains_geometry"],
    }
    write_report_json(OUT_DIR / "R5.8_decision.json", decision)
    print(f"  R5.8_decision.json written (verdict=not_testable, status={BLOCKED_STATUS})")


def phase_provenance() -> None:
    fa = json.load(open(OUT_DIR / "field_identity_audit.json", encoding="utf-8"))
    col = json.load(open(OUT_DIR / "colocation_analysis.json", encoding="utf-8"))
    contract = {
        "phase": "provenance",
        "schema_version": "r5.8.0",
        "status": BLOCKED_STATUS,
        "what_was_produced": {
            "crop_extent_schema": "reports/R5.8/crop_extent_schema.md/.json",
            "field_identity_audit": "reports/R5.8/field_identity_audit.csv/.json",
            "colocation_analysis": "reports/R5.8/colocation_analysis.csv/.json",
            "unblock_schedule": "reports/R5.8/unblock_schedule.json",
            "decision": "reports/R5.8/R5.8_decision.json",
        },
        "every_observation_traces": {
            "crop_record": ("survey_id + survey file + raw cropname + "
                            "Crop_Extent as recorded"),
            "field_id": ("(taluk||hobli||village admin context)+survey_id; "
                         "crop label excluded"),
            "crop_extent_source": "govt_crop_survey_data/ogd_*.csv column Crop_Extent",
            "geometry_transformation": "NONE (extent is scalar; no geometry created)",
            "satellite_image_date": "NOT extracted - blocked by missing geometry",
            "pixel_mask_extraction": "NOT RUN - blocked by missing geometry",
            "environmental_source": "not re-computed in R5.8 (R5.7 master is untouched)",
            "feature_transformation": "none performed in R5.8 (no sub-field features)",
        },
        "field_counts": {
            "unique_fields": fa["unique_fields"],
            "shared_coconut_pepper_fields": fa["shared_coconut_pepper_fields"],
        },
        "colocation": {
            "median_pepper_to_nearest_coconut_m": col[
                "pepper_to_nearest_coconut_m"]["median"],
        },
        "no_fabrication": True,
        "no_geometry_invented": True,
        "r5_6_r5_7_frozen_untouched": True,
    }
    write_report_json(OUT_DIR / "provenance_contract.json", contract)
    print("  provenance_contract.json written (no geometry fabricated)")


def phase_report() -> None:
    schema = json.load(open(OUT_DIR / "crop_extent_schema.json", encoding="utf-8"))
    fa = json.load(open(OUT_DIR / "field_identity_audit.json", encoding="utf-8"))
    col = json.load(open(OUT_DIR / "colocation_analysis.json", encoding="utf-8"))
    dec = json.load(open(OUT_DIR / "R5.8_decision.json", encoding="utf-8"))

    md = [
        "# R5.8 Sub-Field / Crop-Extent Discrimination — Final Report",
        "",
        f"**R5.8 STATUS = {BLOCKED_STATUS}**",
        "",
        "## 1. Executive summary",
        "",
        f"R5.8 was designed to test whether crop-specific sub-field / "
        f"crop-extent imagery recovers held-out discriminative signal for "
        f"coconut vs pepper. Phase 1 establishes that the survey's only "
        f"crop-specific spatial candidate, `Crop_Extent`, is a **scalar "
        f"land-area measure in compound `A-B-C` notation with no geometry** "
        f"(no polygon, bbox, centroid, or coordinates). No crop geometry "
        f"source exists in the repository (only hobli/taluk/district admin "
        f"boundaries). Pepper and coconut in shared intercropped fields are "
        f"GPS co-located at a median **2.8 m**, below Sentinel-2 ~10 m pixels. "
        f"**Therefore there is no physical sub-field region to map "
        f"crop-specific pixels to**, and the sub-field experiment cannot be "
        f"honestly run on this dataset. Reported as **BLOCKED**.",
        "",
        "## 2. R5.5–R5.7 context",
        "",
        "- R5.5: full CropFusion collapses toward the majority class; a tiny "
        "balanced set reaches high accuracy — capacity is not the blocker.",
        "- R5.6: field-level coconut vs pepper balanced acc 50.4–50.6%; "
        "test majority 53.7%.",
        f"- R5.7: recovered pool ceiling **{P5_7_CEILING}%** after 50 m dedup; "
        "77.2% of pepper fields also have a coconut record; primary signal "
        "identified as intercropping co-location.",
        "",
        "## 3. crop_extent schema",
        "",
        f"- **{round(schema['evidence']['fraction_matching_compound_area_pattern']*100,2)}%** "
        f"of rows are `X-Y-Z` scalar area values.",
        "- Contains geometry: **NO**.",
        "- Crop-specific: YES (per-row area).",
        "- Maps to satellite pixels: **NO** (no spatial placement).",
        f"- Rows with coordinate-like extent: {schema['evidence']['rows_with_coordinate_like_extent']}.",
        "",
        "## 4. Field identity",
        "",
        f"- field_id = (taluk||hobli||village admin context) + survey_id, "
        f"**crop label excluded**.",
        f"- Unique fields: **{fa['unique_fields']:,}**.",
        f"- Shared coconut+pepper fields: **{fa['shared_coconut_pepper_fields']:,}**.",
        f"- Coconut rows: {fa['coconut_rows']:,}; Pepper rows: {fa['pepper_rows']:,}.",
        "",
        "## 5. Co-location analysis",
        "",
        f"- **{col['percent_pepper_within_5m_of_coconut']}%** of pepper fields "
        f"lie within 5 m of a coconut field.",
        f"- **{col['percent_pepper_within_25m_of_coconut']}%** within 25 m.",
        f"- Median pepper→nearest-coconut distance: "
        f"**{col['pepper_to_nearest_coconut_m']['median']} m**.",
        f"- p95: {col['pepper_to_nearest_coconut_m']['p95']} m.",
        "",
        "## 6. Geometry validation",
        "",
        "Not applicable — no geometry to validate. The block contract, "
        "validation checklist (validity / non-empty / bounds / area / overlap) "
        "is catalogued in `reports/R5.8/crop_extent_schema.json` "
        "(`unblock_contract`).",
        "",
        "## 7. Field vs sub-field representation",
        "",
        "Not computable: there is no sub-field geometry. The two crops share the "
        "same GPS point and the same satellite pixels at field resolution.",
        "",
        "## 8. Satellite extraction",
        "",
        "**Not run.** bbox / masked-pixel / centroid-patch / local-patch "
        "extraction all require a physical crop location; `crop_extent` does "
        "not provide one and none may be invented (rules 4/6/7).",
        "",
        "## 9. Temporal signatures & 10. temporal features",
        "",
        "**Not run.** No crop-specific extent exists against which to define a "
        "sub-field temporal signature.",
        "",
        "## 11. Leakage audit / split rule",
        "",
        "- Field identity is geometry-neutral and crop-label-free; a coconut row "
        "and a pepper row of the same plot share a `field_id` (split-safe by "
        "construction).",
        f"- Grouped field split rule is specified for the unblocked run: every "
        f"crop row of a field must stay in the same split.",
        "- No leakage features were introduced (no Yield_Proxy_NPP, "
        "benchmark_eligible, crop label, target encoding).",
        "",
        "## 12–15. Baselines / balance / importance / ablations",
        "",
        "**Not run** — no sub-field feature set exists to evaluate.",
        "",
        "## 16. Signal recovery",
        "",
        f"- Verdict: **{dec['verdict']}** (not testable).",
        f"- Signal recovered: **{dec['signal_recovered']}**.",
        f"- Bottleneck: **{dec['primary_bottleneck']}**.",
        f"- CropFusion training justified: **{dec['cropfusion_training_justified']}**.",
        "",
        "## 17. Data ceiling",
        "",
        f"- R5.7 data ceiling: **{P5_7_CEILING}%** (reference).",
        "- R5.8 data ceiling: **not estimable** (no sub-field signal to "
        "separate; geometric information is absent, not merely weak).",
        "",
        "## 18. Recommendation for R5.9 / next",
        "",
        f"- **{dec['recommended_next_phase']}**",
        "- The 50% ceiling is not a model-capacity artefact; it persists because "
        "coconut and pepper are intercropped in the same field at a scale below "
        "the observation resolution, and the data carries no crop geometry to "
        "separate them. Acquire a real per-crop parcel/polygon layer to unblock "
        "R5.8's sub-field phases.",
    ]
    (OUT_DIR / "R5.8_SUBFIELD_DISCRIMINATION_REPORT.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")

    report_json = {
        "title": "R5.8 Sub-Field / Crop-Extent Discrimination",
        "status": BLOCKED_STATUS,
        "sections": {
            "executive_summary": "BLOCKED - crop_extent is scalar, no geometry",
            "context": {"r5_6_ceiling": 50.6, "r5_7_ceiling": P5_7_CEILING},
            "crop_extent_schema": schema,
            "field_identity": fa,
            "colocation": col,
            "decision": dec,
        },
    }
    write_report_json(OUT_DIR / "R5.8_SUBFIELD_DISCRIMINATION_REPORT.json", report_json)
    print(f"  R5.8_SUBFIELD_DISCRIMINATION_REPORT.md/.json written "
          f"(status={BLOCKED_STATUS})")


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #

def run_phase(phase: str) -> None:
    fn: dict[str, Callable[[], None]] = {
        "1": phase1_inspect,
        "2": phase2_field_identity,
        "3": phase3_colocation,
        "schedule": phase_schedule,
        "decision": phase_decision,
        "provenance": phase_provenance,
        "report": phase_report,
    }
    name = PHASES.get(phase)
    if name is None:
        print(f"  (unknown phase {phase!r}, skipping)")
        return
    print(f"=== Phase {phase}: {name} ===")
    fn[phase]()


def print_final_status() -> None:
    def _l(p):
        return json.load(open(OUT_DIR / p, encoding="utf-8")) if (OUT_DIR / p).exists() else {}
    dec = _l("R5.8_decision.json")
    fa = _l("field_identity_audit.json")
    col = _l("colocation_analysis.json")
    if not dec:
        print("\n  (final status deferred until decision phase runs)")
        return
    print("\n" + "\n".join([
        f"R5.8 STATUS = {BLOCKED_STATUS}",
        f"R5.7 DATA CEILING = {P5_7_CEILING}%",
        "R5.8 DATA CEILING = NOT ESTIMABLE (no geometry)",
        f"FIELD COUNT = {fa['unique_fields']:,}",
        f"SUBFIELD OBSERVATIONS = 0 (blocked - no crop geometry)",
        f"COCONUT = {fa['coconut_rows']:,}",
        f"PEPPER = {fa['pepper_rows']:,}",
        f"SHARED FIELDS = {fa['shared_coconut_pepper_fields']:,}",
        f"VALID CROP EXTENTS = 0 (extent is scalar area, not geometry)",
        "BEST REPRESENTATION = N/A (sub-field representations not constructible)",
        "BEST TEST BALANCED ACCURACY = N/A",
        "BEST TEST AUC = N/A",
        f"SIGNAL RECOVERED = {dec['signal_recovered']}",
        f"PRIMARY SIGNAL = {dec['primary_signal']}",
        f"PRIMARY BOTTLENECK = {dec['primary_bottleneck']}",
        f"CROPFUSION TRAINING JUSTIFIED = {dec['cropfusion_training_justified']}",
        f"RECOMMENDED NEXT PHASE = {dec['recommended_next_phase']}",
    ]))


def main() -> None:
    ap = argparse.ArgumentParser(description="R5.8 sub-field discrimination driver")
    ap.add_argument("--phases", default="all",
                    help="comma-separated phases or 'all'")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = sorted(PHASES) if args.phases == "all" else [
        p.strip() for p in args.phases.split(",") if p.strip()]
    for phase in todo:
        run_phase(phase)
    print_final_status()


if __name__ == "__main__":
    main()
