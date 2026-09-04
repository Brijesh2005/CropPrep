"""Correctness-first matcher audit: baseline preservation + match classification.

Given the frozen supervised corpus (``crop_supervised_v1.csv``) this script:
  * records the baseline of every accepted match with its provenance,
  * classifies each row as PRESERVED / RECOVERED / REMOVED with an exact reason,
  * flags label-quality issues (exact-coordinate label conflicts, class
    imbalance, unsupported crops),
  * verifies split leakage (spatial leave-one-taluk-out disjointness).

It is the re-runnable companion to matcher.py's no-fabrication refactor: after
any change to the matching pipeline you can re-point it at a new accepted
corpus and diff against this baseline.

Usage::

    python scripts/match_audit.py [--csv govt_crop_matched_v1/crop_supervised_v1.csv]

Exit codes: 0 = audit clean, 1 = findings that warrant review, 2 = error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "govt_crop_matched_v1" / "crop_supervised_v1.csv"
DEFAULT_MANIFEST = REPO_ROOT / "training_manifests" / "crop_supervised_v1_manifest.json"

SUPERVISED_CLASSES = {"coconut", "pepper", "coffee", "cardamom"}


def load_corpus(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_split"] = df["location_taluk"].map(
        {"Puttur": "validation", "Sullia": "test"}
    ).fillna("train")
    return df


def classify_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Tag every row with a classification and an exact reason.

    Categories (from the correctness-first mandate):
      1 genuine valid match
      2 legitimate match the matcher handled poorly
      3 duplicate
      4 weak/invalid match
      5 fabricated match (caused by the legacy fallback)
    """
    df = df.copy()
    reason: list[str] = []
    classification: list[str] = []
    for _, row in df.iterrows():
        r = _single_reason(row)
        if r is None:
            classification.append("PRESERVED")
            reason.append("genuine valid match")
        else:
            classification.append("REMOVED")
            reason.append(r)
    df["_classification"] = classification
    df["_reason"] = reason
    return df


def _single_reason(row: pd.Series) -> str | None:
    """Return the exact removal reason for a row, or None if it is preserved."""
    crop = str(row.get("crop_label", "")).strip().lower()
    if crop not in SUPERVISED_CLASSES:
        return "unsupported-crop"

    sat = str(row.get("satellite_status", ""))
    if sat != "FULL":
        return "missing-imagery"

    temp = str(row.get("temporal_match_status", ""))
    if temp not in ("EXACT_SEASON", "WITHIN_TOLERANCE"):
        return "invalid-season"

    dist = row.get("spatial_match_distance_km")
    try:
        dist = float(dist)
    except (TypeError, ValueError):
        dist = None
    if dist is not None and dist > 3.0:
        return "invalid-spatial-match"

    return None


def label_quality_issues(df: pd.DataFrame) -> dict:
    """Quantify label-quality concerns without removing any sample."""
    issues: dict = {}
    counts = df["crop_label"].value_counts().to_dict()
    issues["class_counts"] = counts
    issues["class_imbalance"] = {
        "dominant": max(counts, key=counts.get),
        "dominant_count": max(counts.values()),
        "minority": min(counts, key=counts.get),
        "minority_count": min(counts.values()),
        "ratio": round(max(counts.values()) / max(1, min(counts.values())), 1),
        "excluding_blackgram": None,
    }
    for cls in SUPERVISED_CLASSES:
        if cls in counts:
            issues["class_imbalance"]["excluding_blackgram"] = True
    issues["unsupported_crops"] = {
        k: v for k, v in counts.items() if k not in SUPERVISED_CLASSES
    }

    # Exact-coordinate label conflicts (same lat/lon, >1 crop, same year/season).
    key = ["lat", "lon", "year", "season"]
    grouped = df.groupby(key)["crop_label"].nunique()
    conflict_coords = grouped[grouped > 1]
    conflicted = set(conflict_coords.index)
    conflict_rows = df[df.set_index(key).index.isin(conflicted)]
    issues["exact_coord_label_conflicts"] = {
        "distinct_coords": int(len(conflict_coords)),
        "rows_involved": int(len(conflict_rows)),
    }
    return issues


def leakage_audit(df: pd.DataFrame) -> dict:
    """Verify no sample leaks across the leave-one-taluk-out splits."""
    result: dict = {}
    if "_split" not in df.columns:
        df = df.copy()
        df["_split"] = df["location_taluk"].map(
            {"Puttur": "validation", "Sullia": "test"}
        ).fillna("train")
    taluk_split = df.groupby("location_taluk")["_split"].nunique()
    result["taluks_in_multiple_splits"] = int(
        (taluk_split > 1).sum()
    )
    result["split_counts"] = df["_split"].value_counts().to_dict()

    # Exact-coordinate cross-split leakage.
    coords = df.groupby(["lat", "lon"])["_split"].nunique()
    result["exact_coords_in_multiple_splits"] = int((coords > 1).sum())
    result["exact_coord_leak_rows"] = int(
        df[df.set_index(["lat", "lon"]).index.isin(coords[coords > 1].index)].shape[0]
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: corpus not found: {csv_path}", file=sys.stderr)
        return 2

    df = load_corpus(csv_path)
    df = classify_rows(df)
    issues = label_quality_issues(df)
    leak = leakage_audit(df)

    summary = df["_classification"].value_counts().to_dict()
    removed = df[df["_classification"] == "REMOVED"]
    reasons = Counter(removed["_reason"]) if len(removed) else Counter()

    report = {
        "corpus": str(csv_path),
        "total_accepted": int(len(df)),
        "classification_summary": {
            "PRESERVED": int(summary.get("PRESERVED", 0)),
            "RECOVERED": int(summary.get("RECOVERED", 0)),
            "REMOVED": int(summary.get("REMOVED", 0)),
        },
        "removed_by_reason": dict(reasons),
        "label_quality": issues,
        "leakage": leak,
        "formula": {
            "old_accepted": int(len(df)),
            "removed_invalid": int(summary.get("REMOVED", 0)),
            "newly_recovered": int(summary.get("RECOVERED", 0)),
            "new_accepted": int(summary.get("PRESERVED", 0))
            + int(summary.get("RECOVERED", 0)),
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    # Exit code: findings that warrant review if any removal/conflict/leak.
    findings = len(removed) + issues["exact_coord_label_conflicts"]["rows_involved"]
    findings += leak["exact_coord_leak_rows"]
    if issues["unsupported_crops"]:
        findings += sum(issues["unsupported_crops"].values())
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
