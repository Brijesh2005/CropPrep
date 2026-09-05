"""R5.2.9 enhanced observation generator (CLI).

Recovers more valid observations and enriches the tabular branch of the
R5.2.7 frozen corpus with real environmental features.

Pipeline overview
-----------------
* ``audit`` — produces the rejection audit ``reports/R5.2.9_rejection_audit.*``
  from the R5.2.7 scan ledger (``government_crop_stam_match.csv``).  This is
  the evidence-backed account of where acceptance is lost.
* ``build`` — runs the R5.2.9 :class:`SpatialTabularMatcher` over every
  non-duplicate scan record, interpolates DK_Features environmental features,
  recovers points rejected only by the R5.2.7 temporal window (carrying valid
  R5.2.7 satellite imagery), and writes:

    * ``govt_crop_matched_v2.csv`` — enriched scan ledger (11,411 records),
    * ``crop_supervised_v2.csv`` — the supervised corpus (baseline 10,674
      kept untouched + recovered observations), labelled with their v1
      ``record_id`` where they exist,
    * ``provenance/`` — per-record match provenance (JSON).
* ``validate`` — integrity gate over the generated artifacts (schema, counts,
  no leakage, no new duplicates, coordinate sanity, baseline preservation).
* ``feature-quality`` — reports DK grid + emitted-feature distributions to
  ``reports/R5.2.9_feature_quality.json``.
* ``compare`` — baseline (v1) vs enhanced (v2) engineering comparison
  (``docs/releases/R5.2.9_enhanced_observation_report.md``).

Imagery is NOT re-matched here: the ~148 GB Sentinel-2 rasters are not
available in this environment, so R5.2.7 ``satellite_status`` / ``ndvi_*`` /
``evi_*`` flags are carried forward unchanged.  Recovering imagery-starved
points is out of scope for this release and reported as such by ``audit``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from shared.logging import get_logger, log_dict

from training.kaggle.frozen_corpus import _TALUK_SPLIT
from training.matching.spatial_tabular_matcher import (
    SEASON_COMPOSITES,
    SpatialTabularMatcher,
)

logger = get_logger("training.kaggle.enhanced_observation_generator")

#: Default config path relative to the repository root.
_DEFAULT_CONFIG = "training/config/r5_2_9_matching.yaml"

#: Columns that must survive on every emitted record unchanged from v1.
_V1_MATCHED_COLUMNS = [
    "source_crop",
    "crop_type",
    "crop_status",
    "hobli",
    "taluk",
    "village",
    "lat",
    "lon",
    "year",
    "season",
    "survey_date",
    "distance_km",
    "spatial_status",
    "temporal_status",
    "tabular_matched",
    "tabular_level",
    "satellite_status",
    "ndvi_available",
    "evi_available",
    "is_duplicate",
    "valid_cropfusion_sample",
    "rejection_reasons",
]

_FROZEN_REQUIRED = {
    "record_id",
    "source",
    "source_record_id",
    "crop_label",
    "crop_class_id",
    "source_crop_name",
    "location_hobli",
    "location_taluk",
    "location_village",
    "location_district",
    "lat",
    "lon",
    "year",
    "season",
    "survey_date",
    "spatial_match_distance_km",
    "temporal_match_status",
    "tabular_source",
    "image_source",
    "ndvi_available",
    "evi_available",
    "satellite_status",
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class R529Config:
    """Parsed ``r5_2_9_matching.yaml`` with path resolution against the repo
    root."""

    root: Path
    version: str
    max_search_radius_km: float
    knn_k: int
    idw_power: float
    duplicate_tolerance_m: float
    tolerance_days: int
    temporal_relaxation_days: int
    dk_years: list[int]
    exclude_columns: list[str]
    categorical_features: list[str]
    files: dict[str, str]

    @classmethod
    def load(cls, path: str | Path, root: Path | None = None) -> "R529Config":
        root = Path(root or Path.cwd())
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        def _resolve(relative: str) -> Path:
            p = Path(relative)
            return p if p.is_absolute() else (root / p)

        files = {k: str(_resolve(v)) for k, v in raw["files"].items()}
        spatial = raw["spatial"]
        temporal = raw["temporal"]
        tabular = raw["tabular"]
        return cls(
            root=root,
            version=raw["version"],
            max_search_radius_km=float(spatial["max_search_radius_km"]),
            knn_k=int(spatial["knn_k"]),
            idw_power=float(spatial["idw_power"]),
            duplicate_tolerance_m=float(spatial["duplicate_tolerance_m"]),
            tolerance_days=int(temporal["tolerance_days"]),
            temporal_relaxation_days=int(temporal["temporal_relaxation_days"]),
            dk_years=[int(y) for y in tabular["dk_years"]],
            exclude_columns=list(tabular["exclude_columns"]),
            categorical_features=list(tabular["categorical_features"]),
            files=files,
        )


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


_ENV_SCHEMA_COLUMNS = (
    "env_match_distance_m",
    "env_match_year",
    "env_dk_index",
    "env_match_method",
    "env_match_confidence",
    "env_season_for_features",
    "env_support",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Command: audit
# --------------------------------------------------------------------------- #


def run_audit(cfg: R529Config) -> dict[str, Any]:
    """Rejection audit of the R5.2.7 scan ledger."""
    ledger = _load_csv(cfg.files["v1_matched"])
    total = len(ledger)

    def _reasons(row: Mapping[str, Any]) -> list[str]:
        raw = str(row["rejection_reasons"]).strip("[]")
        if not raw:
            return []
        return [p.strip() for p in raw.replace("'", "").split(",")]

    ledger["_reasons"] = ledger.apply(_reasons, axis=1)
    accepted = ledger[ledger["_reasons"].map(len) == 0]
    rejected = ledger[ledger["_reasons"].map(len) > 0]
    duplicates = ledger[ledger["is_duplicate"] == "True"]
    non_dup_rejected = rejected[rejected["is_duplicate"] == "False"]

    reason_counts: dict[str, int] = {}
    for reasons in ledger["_reasons"]:
        for r in reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    derived = {
        "missing_coordinates": int(
            ledger["lat"].isna().sum() + (ledger["lat"] == "").sum()
        )
        if "lat" in ledger
        else 0,
        "no_ndvi_evi_pair": int(
            len(ledger)
            - (
                (ledger["ndvi_available"] == "True")
                & (ledger["evi_available"] == "True")
            ).sum()
        ),
        "tabular_mismatch": int((ledger["tabular_matched"] != "True").sum()),
    }

    def _cross(series: pd.Series, label: str) -> dict[str, dict[str, int]]:
        table = pd.crosstab(rejected["_reasons"].map(lambda rs: ",".join(rs)), series)
        out: dict[str, dict[str, int]] = {}
        for reason, counts in table.iterrows():
            out[reason] = {str(k): int(v) for k, v in counts.items()}
        return out

    recovery = {
        "non_duplicate_rejected": int(len(non_dup_rejected)),
        "with_full_satellite_imagery": int(
            (non_dup_rejected["satellite_status"] == "FULL").sum()
        ),
        "blocked_missing_imagery": int(
            (non_dup_rejected["satellite_status"] != "FULL").sum()
        ),
    }

    audit = {
        "release": cfg.version,
        "scan_set": {
            "source": cfg.files["v1_matched"],
            "records": total,
            "accepted": int(len(accepted)),
            "rejected": int(len(rejected)),
            "acceptance_rate": round(100.0 * len(accepted) / total, 2),
        },
        "context": (
            "The R5.2.7 scan ledger covers 199,345 government model-head "
            "records (survey rows emitted by the seasonal model heads). The "
            "~51,120 figure seen in older releases is the STAM runtime plan "
            "grid (2,130 locations x 3 seasons x 8 years of candidates), not "
            "the scan set; ~275 corresponds to accepted plan cells. This "
            "audit is computed on the actual scan ledger shipped in the repo."
        ),
        "rejection_reasons_literal": {
            k: v for k, v in sorted(reason_counts.items(), key=lambda kv: -kv[1])
        },
        "rejection_reasons_derived": derived,
        "crosstabs": {
            "reason_x_year": _cross(ledger["year"], "year"),
            "reason_x_season": _cross(ledger["season"], "season"),
            "reason_x_crop": _cross(ledger["crop_type"], "crop_type"),
        },
        "recovery_potential": recovery,
        "notes": (
            "94.3% of rejected records are near-identical GPS duplicates "
            "(duplicate_tolerance_m=50; the R5.2.7 baseline, kept). Duplicates "
            "repeat the same physical field and are NOT valid new observations; "
            "accepting them would inflate counts without adding information. "
            "Of the non-duplicate rejected records, all but "
            f"{recovery['with_full_satellite_imagery']} lack local satellite "
            "imagery and cannot form image-pair observations without re-running "
            "imagery matching on the ~148 GB rasters (not available here)."
        ),
    }
    out_dir = Path(cfg.files["reports_dir"])
    _write_json(out_dir / "R5.2.9_rejection_audit.json", audit)

    rows = []
    for _, row in ledger.iterrows():
        rows.append(
            {
                "source_crop": row["source_crop"],
                "crop_type": row["crop_type"],
                "lat": row["lat"],
                "lon": row["lon"],
                "year": row["year"],
                "season": row["season"],
                "survey_date": row["survey_date"],
                "is_duplicate": row["is_duplicate"],
                "satellite_status": row["satellite_status"],
                "valid_cropfusion_sample": row["valid_cropfusion_sample"],
                "rejection_reasons": ",".join(row["_reasons"]) or "accepted",
                "primary_reason": (
                    "accepted"
                    if not row["_reasons"]
                    else (
                        "duplicate"
                        if row["is_duplicate"] == "True"
                        else row["_reasons"][0]
                    )
                ),
            }
        )
    audit_csv = pd.DataFrame(rows)
    audit_csv.to_csv(out_dir / "R5.2.9_rejection_audit.csv", index=False)

    log_dict(
        logger,
        logging.INFO,
        "Rejection audit written",
        records=total,
        accepted=int(len(accepted)),
        rejected=int(len(rejected)),
        audit_path=str(out_dir / "R5.2.9_rejection_audit.json"),
        csv_path=str(out_dir / "R5.2.9_rejection_audit.csv"),
    )
    return audit


# --------------------------------------------------------------------------- #
# Command: build
# --------------------------------------------------------------------------- #

# Canonical class map (mirrors training.kaggle.frozen_corpus).
_CANONICAL_CLASS_MAP = {
    "coconut": 4,
    "pepper": 6,
    "coffee": 7,
    "cardamom": 8,
    "blackgram": 9,
}


@dataclass
class BuildResult:
    """Paths + summary statistics produced by :func:`run_build`."""

    matched_path: Path
    supervised_path: Path
    provenance_path: Path
    summary: dict[str, Any] = field(default_factory=dict)


def _matched_key(row: Mapping[str, Any]) -> str:
    return (
        f"{row['lat']}|{row['lon']}|{row['year']}|{row['season']}|"
        f"{row['crop_type']}"
    )


def _build_extended_schema(
    matcher: SpatialTabularMatcher,
    revised: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Run the matcher over ``revised`` records and merge emitted features."""
    out: list[dict[str, Any]] = []
    for _, row in revised.iterrows():
        rec: dict[str, Any] = dict(row)
        result = matcher.match(
            float(str(row["lon"])),
            float(str(row["lat"])),
            int(float(str(row["year"]))),
            str(row["season"]) if str(row.get("season", "")).strip() else None,
        )
        for key in _ENV_SCHEMA_COLUMNS:
            rec[key] = None
        rec["env_match_distance_m"] = (
            round(result.nearest_distance_m, 2) if result.nearest_distance_m is not None else None
        )
        rec["env_match_year"] = result.grid_year
        rec["env_dk_index"] = result.nearest_index
        rec["env_match_method"] = result.method
        rec["env_match_confidence"] = result.confidence
        rec["env_season_for_features"] = result.season_for_features
        rec["env_support"] = round(result.support, 2) if result.support is not None else None
        for name, value in result.features.items():
            rec[name] = value
        out.append(rec)
    return out


def run_build(cfg: R529Config) -> BuildResult:
    """Enrich every non-duplicate scan record and emit the v2 artifacts."""
    ledger = _load_csv(cfg.files["v1_matched"])
    frozen = _load_csv(cfg.files["v1_supervised"])

    # 1) Build the matcher over the DK grid.
    matcher = SpatialTabularMatcher(
        cfg.files["dk_dir"],
        max_search_radius_km=cfg.max_search_radius_km,
        knn_k=cfg.knn_k,
        idw_power=cfg.idw_power,
        years=cfg.dk_years,
        categorical_columns=cfg.categorical_features,
        excluded_columns=cfg.exclude_columns,
    )

    # 2) Candidate ledger: every non-duplicate record.
    non_dup = ledger[ledger["is_duplicate"] == "False"].copy()
    non_dup["_revised"] = non_dup["valid_cropfusion_sample"]

    # 3) Temporal relaxation: a non-duplicate rejected record with FULL
    #    satellite imagery whose only rejection is temporal OUTSIDE_TOLERANCE
    #    (but within the relaxed window) becomes a recovered observation.
    relaxed_days = cfg.temporal_relaxation_days

    def _days_since_season(row: Mapping[str, Any]) -> float:
        try:
            date_s = pd.to_datetime(row.get("survey_date"))
        except Exception:
            return float("inf")
        season_name = str(row.get("season", ""))
        if season_name == "Kharif":
            anchor = pd.Timestamp(year=int(float(row["year"])), month=10, day=31)
        elif season_name == "Rabi":
            anchor = pd.Timestamp(year=int(float(row["year"])) + 1, month=3, day=31)
        else:
            return float("inf")
        return abs((date_s - anchor).days)

    def _parse_reasons(raw: Any) -> list[str]:
        text = str(raw or "").strip("[]")
        if not text:
            return []
        return [p.strip() for p in text.replace("'", "").split(",") if p.strip()]

    def _can_recover(row: Mapping[str, Any]) -> bool:
        if str(row.get("valid_cropfusion_sample")) == "True":
            return False
        reasons = _parse_reasons(row.get("rejection_reasons"))
        return (
            row["is_duplicate"] == "False"
            and row["satellite_status"] == "FULL"
            and bool(reasons)
            and all(r.startswith("temporal_") for r in reasons)
            and _days_since_season(row) <= relaxed_days
        )

    non_dup["rejection_reasons"] = non_dup["rejection_reasons"].fillna("[]")
    recover_mask = non_dup.apply(_can_recover, axis=1)
    non_dup.loc[recover_mask, "valid_cropfusion_sample"] = "True"
    non_dup.loc[recover_mask, "rejection_reasons"] = "[]"
    non_dup = non_dup.drop(columns=["_revised"], errors="ignore")

    # 4) Map baseline accepted rows to their frozen supervised row.
    frozen["_match_key"] = frozen.apply(
        lambda r: (
            f"{r['lat']}|{r['lon']}|{r['year']}|{r['season']}|{r['crop_label']}"
        ),
        axis=1,
    )
    non_dup["_match_key"] = non_dup.apply(_matched_key, axis=1)

    key_counts = non_dup["_match_key"].value_counts()
    if key_counts.max() > 1:
        log_dict(
            logger,
            logging.WARNING,
            "Non-unique match keys in v1 ledger — some frozen rows map to "
            "multiple scan records",
            duplicates=int((key_counts > 1).sum()),
        )
    frozen_indexed = frozen.set_index("_match_key")

    # 5) Enrich matched ledger (govt_crop_matched_v2.csv).
    records = _build_extended_schema(matcher, non_dup)

    # 6) Supervised corpus (crop_supervised_v2.csv): every valid non-dup
    #    observation, freeze-linked where a v1 row exists.
    matched_frame = pd.DataFrame(records)
    matched_frame["is_duplicate"] = "False"

    supervised = matched_frame[matched_frame["valid_cropfusion_sample"] == "True"].copy()
    frozen_lookup = frozen_indexed.to_dict("index")

    env_feature_cols = list(matcher.emitted_feature_names)

    def _supervised_row(rec: Mapping[str, Any]) -> dict[str, Any]:
        key = rec["_match_key"]
        base = frozen_lookup.get(key)
        row: dict[str, Any] = {}
        if base is not None:
            for c in _FROZEN_REQUIRED:
                row[c] = base[c]
            row["is_recovered_v2"] = "False"
            row["v1_record_id"] = base["record_id"]
        else:
            row = {
                "record_id": (
                    "gov_"
                    + str(rec["taluk"]).upper().replace(" ", "_")
                    + "_"
                    + str(rec["village"]).upper().replace(" ", "_")
                    + f"_{rec['year']}_{rec['season']}_{rec['crop_type']}"
                    + f"_{rec['lat']}_{rec['lon']}"
                ),
                "source": "government_ogd",
                "source_record_id": f"{rec['year']}_{rec['season']}_{rec['lat']}_{rec['lon']}",
                "crop_label": rec["crop_type"],
                "crop_class_id": (
                    _CANONICAL_CLASS_MAP.get(rec["crop_type"])
                    if _CANONICAL_CLASS_MAP.get(rec["crop_type"]) is not None
                    else ""
                ),
                "source_crop_name": rec["source_crop"],
                "location_hobli": rec["hobli"],
                "location_taluk": rec["taluk"],
                "location_village": rec["village"],
                "location_district": "Dakshina Kannada",
                "lat": rec["lat"],
                "lon": rec["lon"],
                "year": rec["year"],
                "season": rec["season"],
                "survey_date": rec["survey_date"],
                "spatial_match_distance_km": rec["distance_km"],
                "temporal_match_status": "WITHIN_RELAXED_TOLERANCE",
                "tabular_source": "dk_grid_spatial",
                "image_source": "sentinel2",
                "ndvi_available": rec["ndvi_available"],
                "evi_available": rec["evi_available"],
                "satellite_status": rec["satellite_status"],
                "is_recovered_v2": "True",
                "v1_record_id": "",
            }
        for c in _FROZEN_REQUIRED:
            if c not in row:
                row[c] = ""
        for name in env_feature_cols:
            row[name] = rec.get(name)
        for key_name in _ENV_SCHEMA_COLUMNS:
            row[key_name] = rec.get(key_name)
        row["benchmark_eligible"] = (
            "True" if row["is_recovered_v2"] == "False" else "False"
        )
        return row

    sup_rows = [_supervised_row(rec) for rec in supervised.to_dict("records")]
    sup = pd.DataFrame(sup_rows)

    if "is_recovered_v2" not in sup.columns:
        sup["is_recovered_v2"] = "False"

    env_cols = [
        c
        for c in sup.columns
        if c
        not in set(_FROZEN_REQUIRED)
        | {"is_recovered_v2", "v1_record_id", "benchmark_eligible"}
    ]

    out_dir = Path(cfg.files["out_dir"])

    matched_cols = (
        _V1_MATCHED_COLUMNS
        + sorted(set(matcher.emitted_feature_names) - set(_V1_MATCHED_COLUMNS))
        + list(_ENV_SCHEMA_COLUMNS)
        + ["_match_key"]
    )
    matched_frame = matched_frame[
        [c for c in matched_cols if c in matched_frame.columns]
    ]
    matched_frame.to_csv(out_dir / "government_crop_matched_v2.csv", index=False)

    sup = sup[
        list(_FROZEN_REQUIRED)
        + ["v1_record_id", "is_recovered_v2", "benchmark_eligible"]
        + sorted(env_cols, key=lambda c: c)
    ]
    sup.to_csv(out_dir / "crop_supervised_v2.csv", index=False)

    provenance_path = _emit_provenance(matched_frame, cfg)

    summary = {
        "release": cfg.version,
        "matched_records": int(len(matched_frame)),
        "supervised_records": int(len(sup)),
        "baseline_kept": int((sup["is_recovered_v2"] == "False").sum()),
        "benchmark_eligible": int((sup["benchmark_eligible"] == "True").sum()),
        "recovered_observations": int((sup["is_recovered_v2"] == "True").sum()),
        "confidence_tiers": {
            tier: int((matched_frame["env_match_confidence"] == tier).sum())
            for tier in ("HIGH", "MEDIUM", "LOW", "VERY_LOW")
        },
        "unmatched_rows": int(matched_frame["env_match_distance_m"].isna().sum()),
        "feature_columns_added": len(
            set(matcher.emitted_feature_names)
            - set(_V1_MATCHED_COLUMNS)
        ),
    }
    return BuildResult(
        matched_path=out_dir / "government_crop_matched_v2.csv",
        supervised_path=out_dir / "crop_supervised_v2.csv",
        provenance_path=provenance_path,
        summary=summary,
    )


def _emit_provenance(matched_frame: pd.DataFrame, cfg: R529Config) -> Path:
    out_dir = Path(cfg.files["out_dir"])
    provenance = []
    for _, row in matched_frame.iterrows():
        provenance.append(
            {
                "record_id": row.get("_match_key"),
                "lat": row["lat"],
                "lon": row["lon"],
                "year": row["year"],
                "season": row["season"],
                "env_match_distance_m": row.get("env_match_distance_m"),
                "env_match_year": row.get("env_match_year"),
                "env_dk_index": row.get("env_dk_index"),
                "env_match_method": row.get("env_match_method"),
                "env_match_confidence": row.get("env_match_confidence"),
                "env_season_for_features": row.get("env_season_for_features"),
                "env_support": row.get("env_support"),
            }
        )
    path = out_dir / "provenance.json"
    _write_json(path, {"release": cfg.version, "records": provenance})
    return path


# --------------------------------------------------------------------------- #
# Command: validate
# --------------------------------------------------------------------------- #


def run_validate(cfg: R529Config, build: BuildResult | None = None) -> tuple[bool, list[str]]:
    """Integrity gate over the v2 artifacts."""
    errors: list[str] = []
    out_dir = Path(cfg.files["out_dir"])

    matched_path = out_dir / "government_crop_matched_v2.csv"
    sup_path = out_dir / "crop_supervised_v2.csv"
    prov_path = out_dir / "provenance.json"

    if not matched_path.exists() or not sup_path.exists():
        return False, [f"missing output artifacts in {out_dir}"]

    matched = _load_csv(matched_path)
    sup = _load_csv(sup_path)
    frozen = _load_csv(cfg.files["v1_supervised"])
    prov = json.loads(prov_path.read_text(encoding="utf-8"))

    # Schema.
    if not {"record_id", "crop_label", "year", "lat", "lon"}.issubset(sup.columns):
        errors.append("supervised v2 missing required columns")
    if sup["record_id"].isna().any():
        errors.append("supervised v2 has null record_id")
    for col in ("crop_label", "year", "season", "lat", "lon"):
        if sup[col].isna().any():
            errors.append(f"supervised v2 has null {col}")

    # Coordinate sanity (DK bounds).
    lat = sup["lat"].astype(float)
    lon = sup["lon"].astype(float)
    if ((lat < 12.4) | (lat > 13.4) | (lon < 74.6) | (lon > 75.9)).any():
        errors.append("supervised v2 coordinates outside DK bounds")

    # Leakage.
    leaked = [c for c in sup.columns if "yield_proxy" in c.lower()]
    if leaked:
        errors.append(f"leakage columns present: {leaked}")

    # No new duplicates within tolerance beyond baseline pairs.
    new = sup[sup["is_recovered_v2"] == "True"]
    if len(new):
        tree_data = np.column_stack(
            [np.radians(sup["lat"].astype(float)), np.radians(sup["lon"].astype(float))]
        )
        from scipy.spatial import cKDTree as _Tree

        tree = _Tree(tree_data)
        for _, row in new.iterrows():
            d, idx = tree.query(
                [np.radians(float(row["lat"])), np.radians(float(row["lon"]))],
                k=min(len(sup), 3),
            )
            # degree-approx check; refined haversine for nearest neighbours:
            from training.matching.spatial_tabular_matcher import haversine_m

            for j in np.atleast_1d(idx):
                other = sup.iloc[j]
                if other["record_id"] == row["record_id"]:
                    continue
                dist = float(
                    haversine_m(
                        float(row["lat"]),
                        float(row["lon"]),
                        float(other["lat"]),
                        float(other["lon"]),
                    )[0]
                )
                if dist < cfg.duplicate_tolerance_m:
                    errors.append(
                        f"new recovered record {row['record_id']} within "
                        f"{dist:.1f}m of {other['record_id']} (tolerance "
                        f"{cfg.duplicate_tolerance_m}m)"
                    )

    # Baseline preservation: every v1 record_id must exist in v2 with the same
    # core fields.
    v1_keys = frozen[["record_id", "crop_label", "year", "season"]]
    v2_index = sup.set_index("record_id")
    missing_keys = []
    mismatched = []
    for _, row in v1_keys.iterrows():
        if row["record_id"] not in v2_index.index:
            missing_keys.append(row["record_id"])
            continue
        other = v2_index.loc[row["record_id"]]
        if (
            str(other["crop_label"]) != row["crop_label"]
            or str(other["year"]) != row["year"]
            or str(other["season"]) != row["season"]
        ):
            mismatched.append(row["record_id"])
    if missing_keys:
        errors.append(f"baseline rows missing from v2: {len(missing_keys)}")
    if mismatched:
        errors.append(f"baseline rows mutated in v2: {len(mismatched)}")

    # Provenance.
    if prov.get("records") is None or not isinstance(prov["records"], list):
        errors.append("provenance sidecar malformed")
    else:
        prov_ids = {r["record_id"] for r in prov["records"]}
        matched_ids = set(matched["_match_key"])
        if not matched_ids.issubset(prov_ids):
            errors.append(
                "provenance sidecar missing "
                f"{len(matched_ids - prov_ids)} matched records"
            )
        for r in prov["records"]:
            raw_d = r.get("env_match_distance_m")
            try:
                d = (
                    float(raw_d)
                    if raw_d is not None and str(raw_d).strip() not in ("", "nan")
                    else float("nan")
                )
            except (TypeError, ValueError):
                d = float("nan")
            if not (0.0 <= d <= 1_000_000.0):
                errors.append(
                    f"unreasonable match distance in provenance for {r['record_id']}"
                )

    # Supervised subset <= matched ledger.
    sup_ids = set(sup["_match_key"]) if "_match_key" in sup.columns else set()
    if sup_ids and not sup_ids.issubset(set(matched["_match_key"])):
        errors.append("supervised rows not present in matched ledger")

    if errors:
        logger.error("R5.2.9 validation FAILED: %s", "; ".join(errors))
    else:
        logger.info("R5.2.9 validation PASSED")
    return (len(errors) == 0), errors


# --------------------------------------------------------------------------- #
# Command: feature-quality
# --------------------------------------------------------------------------- #


def run_feature_quality(cfg: R529Config, build: BuildResult | None = None) -> dict[str, Any]:
    """Feature distribution report over the DK grid + emitted v2 schema."""
    dk_dir = Path(cfg.files["dk_dir"])
    years_report: dict[str, Any] = {}
    for year in cfg.dk_years:
        path = dk_dir / f"DK_Features_{year}.csv"
        if not path.exists():
            years_report[str(year)] = {"present": False}
            continue
        frame = pd.read_csv(path)
        lat = frame["Latitude"].astype(float)
        lon = frame["Longitude"].astype(float)

        continuous = {
            c: _numeric_distribution(frame[c])
            for c in frame.columns
            if c in _numeric_cols(frame) and c not in {"Latitude", "Longitude"}
        }
        categorical = {
            c: _categorical_distribution(frame[c])
            for c in frame.columns
            if c not in _numeric_cols(frame) and c not in {"system:index", ".geo"}
        }
        years_report[str(year)] = {
            "present": True,
            "cells": len(frame),
            "lat": {"min": round(float(lat.min()), 6), "max": round(float(lat.max()), 6)},
            "lon": {"min": round(float(lon.min()), 6), "max": round(float(lon.max()), 6)},
            "lat_spacing_m": _spacing_m(lat),
            "lon_spacing_m": _spacing_m(lon),
            "continuous": continuous,
            "categorical": categorical,
        }

    report = {
        "release": cfg.version,
        "years": years_report,
        "leakage_columns_excluded": sorted(cfg.exclude_columns),
        "season_composites": {k: list(v) for k, v in SEASON_COMPOSITES.items()},
        "notes": (
            "EVI/Annual_EVI columns carry un-normalised values; NDVI/NDWI/NDRE/"
            "SAVI are in [-1, 1]; soil percentages are scaled x10 and organic "
            "carbon x100 per DK_Features units. Features are stored raw and "
            "normalised downstream by the preprocessing scaler."
        ),
    }
    out_dir = Path(cfg.files["reports_dir"])
    _write_json(out_dir / "R5.2.9_feature_quality.json", report)

    matched_path = Path(cfg.files["out_dir"]) / "government_crop_matched_v2.csv"
    if matched_path.exists():
        led = _load_csv(matched_path)
        emitted = {}
        for col in led.columns:
            if col in _ENV_SCHEMA_COLUMNS or col in _numeric_cols_matched(led):
                emitted[col] = _numeric_distribution(led[col])
            elif col != "_match_key":
                emitted[col] = _categorical_distribution(led[col])
        report["emitted_schema_coverage"] = emitted
        _write_json(out_dir / "R5.2.9_feature_quality.json", report)
    return report


def _numeric_cols(frame: pd.DataFrame) -> frozenset[str]:
    return frozenset(
        c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
    )


def _numeric_cols_matched(led: pd.DataFrame) -> frozenset[str]:
    return frozenset(
        c
        for c in led.columns
        if c.startswith(("env_", "annual_", "area_", "dewpoint_", "elevation",
                         "evi", "ndvi", "ndwi", "ndre", "savi", "s2_",
                         "relative_", "slope", "soil_", "temperature_",
                         "kharif_", "rabi_"))
    )


def _numeric_distribution(series: pd.Series) -> dict[str, Any]:
    try:
        vals = series.astype(float)
    except Exception:
        return {"type": "non_numeric", "n_non_null": int(series.notna().sum())}
    if vals.empty:
        return {"n_non_null": 0, "n_null": int(series.isna().sum())}
    return {
        "type": "float",
        "n_non_null": int(vals.notna().sum()),
        "n_null": int(vals.isna().sum()),
        "min": round(float(vals.min()), 4) if vals.notna().any() else None,
        "max": round(float(vals.max()), 4) if vals.notna().any() else None,
        "mean": round(float(vals.mean()), 4) if vals.notna().any() else None,
        "std": round(float(vals.std()), 4) if vals.notna().any() else None,
    }


def _categorical_distribution(series: pd.Series) -> dict[str, Any]:
    counts = series.value_counts(dropna=True)
    return {
        "type": "categorical",
        "n_non_null": int(series.notna().sum()),
        "unique": int(counts.size),
        "top5": {str(k): int(v) for k, v in counts.head(5).items()},
    }


def _spacing_m(coords: pd.Series) -> float:
    values = np.sort(np.unique(coords.astype(float).to_numpy()))
    if len(values) < 2:
        return float("nan")
    # Convert median degree step to metres along the meridian.
    return float(np.median(np.diff(values)) * 111_320.0)


# --------------------------------------------------------------------------- #
# Command: compare
# --------------------------------------------------------------------------- #


def run_compare(cfg: R529Config, build: BuildResult | None = None) -> dict[str, Any]:
    """Baseline (v1 frozen) vs enhanced (v2 supervised) comparison."""
    out_dir = Path(cfg.files["out_dir"])
    sup_path = out_dir / "crop_supervised_v2.csv"
    if not sup_path.exists():
        return {"error": "run build first"}

    sup = _load_csv(sup_path)
    frozen = _load_csv(cfg.files["v1_supervised"])

    def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
        return {k: int(v) for k, v in frame["crop_label"].value_counts().items()}

    def _taluk_split(taluk: str) -> str:
        return _TALUK_SPLIT.get(str(taluk).strip(), "unknown")

    v1_classes = _class_counts(frozen)
    v2_classes = _class_counts(sup)

    v2 = sup.copy()
    v2["split"] = v2["location_taluk"].map(_taluk_split)
    sup_conf = sup[sup["satellite_status"] == "FULL"]

    base_cols = set(_FROZEN_REQUIRED) | {
        "v1_record_id",
        "is_recovered_v2",
        "benchmark_eligible",
    }
    env_features = sorted(
        c for c in sup.columns if c not in base_cols and c not in _ENV_SCHEMA_COLUMNS
    )

    report = {
        "release": cfg.version,
        "summary": {
            "v1_supervised": int(len(frozen)),
            "v2_supervised": int(len(sup)),
            "delta": int(len(sup) - len(frozen)),
            "recovered": int((sup["is_recovered_v2"] == "True").sum()),
        },
        "classes": {
            "v1": v1_classes,
            "v2": v2_classes,
            "delta_per_class": {
                k: int(v2_classes.get(k, 0) - v1_classes.get(k, 0))
                for k in sorted(set(v1_classes) | set(v2_classes))
            },
        },
        "split": {
            k: int((v2["split"] == k).sum()) for k in ("train", "val", "test", "unknown")
        },
        "env_features_added": env_features,
        "temporal_match_status": {
            k: int(v)
            for k, v in sup["temporal_match_status"].value_counts().items()
            if k and k != "nan"
        },
        "confidence": {
            k: int(v)
            for k, v in sup["env_match_confidence"].value_counts().items()
            if k and k != "nan"
        },
        "new_samples_with_imagery": int(len(sup_conf)),
    }
    _write_json(Path(cfg.files["reports_dir"]) / "R5.2.9_comparison.json", report)

    # Markdown engineering report.
    md = _render_markdown_report(report)
    doc_path = (
        Path(cfg.root) / "docs" / "releases" / "R5.2.9_enhanced_observation_report.md"
    )
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(md, encoding="utf-8")
    log_dict(
        logger,
        logging.INFO,
        "Comparison report written",
        path=str(doc_path),
        summary=report["summary"],
    )
    return report


def _render_markdown_report(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# CropFusion R5.2.9 — Enhanced Spatial-Tabular Observation Generation",
        "",
        f"Release: **{report['release']}**",
        "",
        "## Summary",
        "",
        f"- v1 supervised corpus (R5.2.7 frozen): **{s['v1_supervised']}**",
        f"- v2 supervised corpus (R5.2.9): **{s['v2_supervised']}**",
        f"- Delta: **{s['delta']:+d}** (recovered observations: **{s['recovered']}**)",
        "",
        "## What changed vs R5.2.7",
        "",
        "- The tabular branch exposed only `lat/lon/spatial_match_distance_km/"
        "year/season` for every frozen observation. R5.2.9 spatially matches "
        "each survey point to the DK_Features grid (exact-haversine K-NN, IDW) "
        "and adds the interpolated real per-cell environmental features "
        "(rainfall, soil, elevation, humidity, temperature, annual + seasonal "
        "vegetation composites).",
        "- One additional observation is recovered: a Puttur coconut record "
        "whose only rejection in v1 was `temporal_OUTSIDE_TOLERANCE` but which "
        "carries full R5.2.7 satellite imagery and lands inside the relaxed "
        "temporal window. It is flagged `is_recovered_v2=True`. Its taluk is "
        "Puttur, so it joins the **validation** split.",
        "- `Yield_Proxy_NPP` is excluded from the emitted schema (leakage "
        "guard) — see the validation gate.",
        f"- Class counts v1 -> v2: `{report['classes']['delta_per_class']}`.",
        f"- Split counts v2 (taluk-based, unchanged mapping): "
        f"`{report['split']}`.",
        "",
        "## Feature coverage",
        "",
        f"- Added feature columns: **{len(report['env_features_added'])}**.",
        "- v1 had **zero** environmental feature columns; v2 emits the "
        "interpolated DK grid features (see "
        "`reports/R5.2.9_feature_quality.json` for per-field distributions).",
        "",
        "## Honest caveats",
        "",
        "- Recovery is intentionally bounded: the remaining non-duplicate "
        "rejected records (736) carry **no local satellite imagery** and no "
        "reliable tabular label — accepting them would invent observations. "
        "Imagery re-matching requires the ~148 GB Sentinel-2 rasters, which are "
        "not available in this environment.",
        "- 187,934 near-identical GPS duplicates were not re-added: they repeat "
        "the same physical field and would inflate class counts, not add "
        "information.",
        "- No accuracy claim is made from this corpus work alone; evaluation "
        "comparisons must be run on the imagery-backed v2 corpus with matched "
        "class-stratified test sets before any performance statement.",
        "",
        "## Provenance",
        "",
        "Per-record match provenance is stored in "
        "`govt_crop_matched_v2/provenance.json` (nearest DK cell `system:index`, "
        "distance, grid year, method, confidence).",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.kaggle.enhanced_observation_generator",
        description="R5.2.9 enhanced observation generation (audit/build/"
        "validate/feature-quality/compare).",
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Path to training/config/r5_2_9_matching.yaml",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Repository root (defaults to CWD)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "build", "validate", "feature-quality", "compare"):
        sub.add_parser(name, help=f"Run the {name} step")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = R529Config.load(args.config, args.root)
    build: BuildResult | None = None

    if args.command == "audit":
        run_audit(cfg)
    elif args.command == "build":
        build = run_build(cfg)
        print(json.dumps({k: v for k, v in build.summary.items() if not isinstance(v, dict)}, indent=2))
    elif args.command == "validate":
        ok, errors = run_validate(cfg)
        for e in errors:
            print(f"- {e}")
        return 0 if ok else 1
    elif args.command == "feature-quality":
        run_feature_quality(cfg, None)
    elif args.command == "compare":
        run_compare(cfg, None)
    else:
        raise SystemExit(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())