"""prepare_data.py — Build the clean master dataset from approved sources.

NEW active data flow (Phase 2):

  Karnataka OGD Crop Survey
  + Dakshina Kannada DK_Features
  + Sentinel-2 imagery (Kaggle)
  ↓
  master.csv + image_manifest.csv + split_manifest.json

Approved sources:
  1. govt_crop_survey_data/ogd_unified_all_hoblis.csv  (Belthangady, Mangalore, Bantwal, Sullia)
  2. govt_crop_survey_data/ogd_putturu_kharif_2020_21.csv  (Puttur)
  3. DK_Features (training/datasets/tabular/) — district-level summaries
  4. Sentinel-2 imagery (Kaggle: shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada)

Excluded datasets (must NOT participate):
  data_season.csv, ICRISAT-District Level Data.csv, cropdata_updated.csv,
  All-India crop datasets, dataset.csv, Yield_Proxy_NPP as supervised target.

Usage:
  python training/prepare_data.py --repo-root .
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import csv
from collections import Counter

# ---------------------------------------------------------------------------
# Crop name normalisation (OGD raw → standard label)
# ---------------------------------------------------------------------------

CROP_NAME_MAP: dict[str, str] = {
    "Coconut": "coconut",
    "Betel Nuts (Areca nuts)": "coconut",
    "Pepper (Black)": "pepper",
    "Coffee arabica": "coffee",
    "Coffee robusta": "coffee",
    "Cardamom": "cardamom",
}

# Supervised classes (learnable); Urad/blackgram excluded (only 2 samples, zero training support)
SUPERVISED_CLASSES = {"coconut", "pepper", "coffee", "cardamom"}

# Spatial split — frozen, do NOT change
SPLIT_TALUKS: dict[str, str] = {
    "Belthangady": "train",
    "Mangalore": "train",
    "Bantwal": "train",
    "Puttur": "val",
    "Sullia": "test",
}

# Dakshina Kannada bounding box (observed OGD extent + margin).
# Coordinates far outside DK (e.g. stray rows in the raw files) are dropped.
DK_LAT_RANGE = (12.4, 13.4)
DK_LON_RANGE = (74.5, 76.0)


def _in_dk_bounds(lat: str, lon: str) -> bool:
    try:
        la, lo = float(lat), float(lon)
    except ValueError:
        return False
    return DK_LAT_RANGE[0] <= la <= DK_LAT_RANGE[1] and DK_LON_RANGE[0] <= lo <= DK_LON_RANGE[1]


def _make_sample_id(
    lat: str,
    lon: str,
    year: str,
    season: str,
    crop_label: str,
    hobli: str,
    village: str,
) -> str:
    """Deterministic sample_id from observation fields."""
    raw = f"{lat}|{lon}|{year}|{season}|{crop_label}|{hobli}|{village}"
    short_hash = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"obs_{short_hash}"


def _parse_year(years_field: str) -> str:
    """Convert OGD Years field (e.g. '2020-2021') to a single year string."""
    parts = years_field.split("-")
    return parts[0].strip() if parts else years_field.strip()


def _parse_month(date_str: str) -> str:
    """Extract month number from CropSurveyDate (e.g. '2020-09-04' → '09')."""
    if not date_str or date_str == "NULL":
        return ""
    parts = date_str.split("-")
    return parts[1] if len(parts) >= 2 else ""


def _load_ogd_csv(path: Path) -> list[dict[str, str]]:
    """Load an OGD CSV and return list of row dicts."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _build_master_rows(repo_root: Path) -> list[dict[str, Any]]:
    """Load approved sources, filter to target crops, build master rows."""
    survey_dir = repo_root / "govt_crop_survey_data"
    unified_path = survey_dir / "ogd_unified_all_hoblis.csv"
    puttur_path = survey_dir / "ogd_putturu_kharif_2020_21.csv"

    # Load raw OGD data
    unified_rows = _load_ogd_csv(unified_path)
    puttur_rows = _load_ogd_csv(puttur_path)
    all_ogd = unified_rows + puttur_rows

    master_rows: list[dict[str, Any]] = []
    skipped_no_crop = 0
    skipped_no_taluk = 0
    skipped_no_coords = 0
    skipped_out_of_dk = 0
    skipped_unsupervised = 0

    for row in all_ogd:
        taluk = row.get("Taluk_Name", "").strip()
        if taluk not in SPLIT_TALUKS:
            skipped_no_taluk += 1
            continue

        raw_crop = row.get("Cropname", "").strip()
        crop_label = CROP_NAME_MAP.get(raw_crop)
        if crop_label is None:
            skipped_no_crop += 1
            continue

        if crop_label not in SUPERVISED_CLASSES:
            skipped_unsupervised += 1
            continue

        lat = row.get("Latitude", "").strip()
        lon = row.get("Longtitude", "").strip()
        if not lat or not lon or lat == "NULL" or lon == "NULL":
            skipped_no_coords += 1
            continue
        if not _in_dk_bounds(lat, lon):
            skipped_out_of_dk += 1
            continue

        year = _parse_year(row.get("Years", ""))
        season = row.get("Season", "").strip()
        hobli = row.get("Hobli_Name", "").strip()
        village = row.get("Village_Name", "").strip()
        month = _parse_month(row.get("CropSurveyDate", ""))
        image_url = row.get("Image_url", "").strip()
        crop_extent = row.get("Crop_Extent", "").strip()
        survey_id = row.get("Survey_id", "").strip()
        split = SPLIT_TALUKS[taluk]

        sample_id = _make_sample_id(lat, lon, year, season, crop_label, hobli, village)

        master_rows.append(
            {
                "sample_id": sample_id,
                "survey_id": survey_id,
                "latitude": float(lat),
                "longitude": float(lon),
                "taluk": taluk,
                "hobli": hobli,
                "village": village,
                "year": year,
                "season": season,
                "month": month,
                "crop_label": crop_label,
                "crop_extent_ogd": crop_extent,
                "image_url": image_url if image_url and image_url != "NULL" else "",
                "split": split,
                "yield_target": "",  # unresolved — see docs/YIELD_TARGET_ANALYSIS.md
            }
        )

    # Deduplicate: keep first observation per sample_id
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    dupes_removed = 0
    for row in master_rows:
        if row["sample_id"] not in seen_ids:
            seen_ids.add(row["sample_id"])
            deduped.append(row)
        else:
            dupes_removed += 1

    print(f"[prepare_data] Loaded {len(all_ogd)} raw OGD observations")
    print(f"[prepare_data] Kept {len(master_rows)} target-crop observations")
    print(f"[prepare_data] Removed {dupes_removed} duplicate sample_ids")
    print(f"[prepare_data] Final unique rows: {len(deduped)}")
    print(f"[prepare_data] Skipped: no_taluk={skipped_no_taluk}, no_crop_match={skipped_no_crop}, "
          f"unsupervised={skipped_unsupervised}, no_coords={skipped_no_coords}, "
          f"out_of_dk_bounds={skipped_out_of_dk}")

    return deduped


def _write_master_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Write master.csv."""
    fieldnames = [
        "sample_id",
        "survey_id",
        "latitude",
        "longitude",
        "taluk",
        "hobli",
        "village",
        "year",
        "season",
        "month",
        "crop_label",
        "crop_extent_ogd",
        "image_url",
        "split",
        "yield_target",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[prepare_data] Wrote {len(rows)} rows to {out_path}")


def _write_image_manifest(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Write image_manifest.csv.

    image_status semantics (per phase-2 spec):
      VALID   — an image link exists for this observation (OGD survey photo URL)
      MISSING — no image available for this observation
      INVALID — image exists but is unusable/corrupt (reserved; none today)
    """
    fieldnames = [
        "sample_id",
        "image_url",
        "year",
        "latitude",
        "longitude",
        "image_source",
        "image_status",
        "sentinel2_status",
    ]
    manifest_rows = []
    for row in rows:
        url = row["image_url"]
        if url:
            status = "VALID"
            source = "ogd_survey_photo"
        else:
            status = "MISSING"
            source = "none"
        manifest_rows.append(
            {
                "sample_id": row["sample_id"],
                "image_url": url,
                "year": row["year"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "image_source": source,
                "image_status": status,
                "sentinel2_status": "NOT_DOWNLOADED",
            }
        )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[prepare_data] Wrote {len(manifest_rows)} rows to {out_path}")


def _write_split_manifest(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Write split_manifest.json."""
    split_counts: dict[str, int] = Counter(r["split"] for r in rows)
    class_by_split: dict[str, dict[str, int]] = {}
    for split_name in ["train", "val", "test"]:
        split_rows = [r for r in rows if r["split"] == split_name]
        class_by_split[split_name] = dict(Counter(r["crop_label"] for r in split_rows))

    manifest = {
        "split_strategy": "spatial_leave_one_taluk_out",
        "split_groups": {
            "train_taluk": ["Belthangady", "Mangalore", "Bantwal"],
            "validation_taluk": "Puttur",
            "test_taluk": "Sullia",
        },
        "total_samples": len(rows),
        "split_counts": dict(split_counts),
        "class_by_split": class_by_split,
        "frozen_split_source": "crop_supervised_v2.0_manifest.json (reference only)",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prepare_data] Wrote split manifest to {out_path}")


def build_master_dataset(repo_root: Path) -> dict[str, Any]:
    """Build the complete master dataset from approved sources."""
    rows = _build_master_rows(repo_root)

    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    master_path = data_dir / "master.csv"
    image_path = data_dir / "image_manifest.csv"
    split_path = data_dir / "split_manifest.json"

    _write_master_csv(rows, master_path)
    _write_image_manifest(rows, image_path)
    _write_split_manifest(rows, split_path)

    # Summary
    crop_counts = Counter(r["crop_label"] for r in rows)
    split_counts = Counter(r["split"] for r in rows)
    img_status = Counter(
        "HAS_IMAGE" if r["image_url"] else "NO_IMAGE" for r in rows
    )
    unique_ids = len(set(r["sample_id"] for r in rows))

    report = {
        "total_rows": len(rows),
        "unique_sample_ids": unique_ids,
        "duplicates": len(rows) - unique_ids,
        "class_distribution": dict(crop_counts),
        "split_counts": dict(split_counts),
        "image_status": dict(img_status),
        "yield_target": "UNRESOLVED",
        "output_files": {
            "master_csv": str(master_path),
            "image_manifest": str(image_path),
            "split_manifest": str(split_path),
        },
    }

    print(f"\n[prepare_data] === SUMMARY ===")
    print(f"  Total rows: {report['total_rows']}")
    print(f"  Unique sample IDs: {report['unique_sample_ids']}")
    print(f"  Duplicates: {report['duplicates']}")
    print(f"  Class distribution: {report['class_distribution']}")
    print(f"  Split counts: {report['split_counts']}")
    print(f"  Image status: {report['image_status']}")
    print(f"  Yield target: {report['yield_target']}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build master dataset from approved sources.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    args = parser.parse_args()
    build_master_dataset(Path(args.repo_root).resolve())


if __name__ == "__main__":
    main()
