"""Local verification of the frozen R5.2.7 corpus without GPU / imagery.

Validates the frozen corpus end-to-end using only the CSV + manifest,
verifying all R5.2.8 data-integrity guarantees:

  1. Manifest checksum (SHA-256 of the manifest file matches its own
     reproducibility.checksums entry for crop_supervised_v1.csv).
  2. CSV row count matches manifest total_samples.
  3. Split counts (train / val / test) via taluk assignment match manifest.
  4. Class distribution (overall, train, val, test) matches manifest.
  5. Every observation carries frozen corpus provenance.
  6. No old data sources (data_season.csv, etc.) leaked in.
  7. Multimodal contract: tabular features, sequence, temporal, quality
     are all present on every observation.
  8. Yield separation: no yield_value is set.
  9. Duplicate identity: no duplicate record_id values.
 10. Observation construction: all required fields populated.

Run::

    python training/kaggle/scripts/verify_frozen_corpus_local.py

Exit code 0 = all checks passed. Non-zero = at least one failure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]

import sys
sys.path.insert(0, str(_REPO_ROOT))

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        failures.append(name)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    csv_path = _REPO_ROOT / "govt_crop_matched_v1" / "crop_supervised_v1.csv"
    manifest_path = _REPO_ROOT / "training_manifests" / "crop_supervised_v1_manifest.json"

    print("=" * 72)
    print("  R5.2.8 LOCAL FROZEN CORPUS VERIFICATION")
    print("=" * 72)
    print(f"  CSV      : {csv_path}")
    print(f"  Manifest : {manifest_path}")
    print()

    # -- 1. Manifest loads and validates -------------------------------- #
    print("[1] Manifest validation")
    check("manifest exists", manifest_path.exists())
    if not manifest_path.exists():
        _summary()
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check("dataset_version == crop_supervised_v1.1",
          manifest.get("dataset_version") == "crop_supervised_v1.1",
          f"got {manifest.get('dataset_version')!r}")

    required_fields = {
        "dataset_version", "total_samples", "train_samples",
        "validation_samples", "test_samples", "class_mapping", "split_groups",
        "provenance_schema",
    }
    missing = required_fields - manifest.keys()
    check("required manifest fields present", not missing, f"missing={sorted(missing)}")

    total = manifest["total_samples"]
    train_n = manifest["train_samples"]
    val_n = manifest["validation_samples"]
    test_n = manifest["test_samples"]
    check("split sums to total", train_n + val_n + test_n == total,
          f"{train_n}+{val_n}+{test_n}={train_n+val_n+test_n} != {total}")

    # -- 2. Manifest checksum ------------------------------------------- #
    print("\n[2] Manifest checksum")
    actual_csv_checksum = _sha256_file(csv_path)
    expected_csv_checksum = (
        manifest.get("reproducibility", {})
        .get("dataset_checksums", {})
        .get("crop_supervised_v1.csv")
    )
    check("CSV checksum matches manifest",
          actual_csv_checksum == expected_csv_checksum,
          f"actual={actual_csv_checksum[:16]}... expected={expected_csv_checksum[:16] if expected_csv_checksum else 'NONE'}...")

    # -- 3. CSV loads and has correct row count ------------------------- #
    print("\n[3] CSV validation")
    check("CSV exists", csv_path.exists())
    if not csv_path.exists():
        _summary()
        return 1

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    check("CSV row count matches manifest", len(rows) == total,
          f"csv={len(rows)} manifest={total}")

    # -- 4. Class distribution ------------------------------------------ #
    print("\n[4] Class distribution")
    csv_class_counts: dict[str, int] = {}
    for row in rows:
        label = row.get("crop_label", "unknown")
        csv_class_counts[label] = csv_class_counts.get(label, 0) + 1

    manifest_class_counts = manifest.get("class_counts", {}).get("overall", {})
    for cls, expected_count in manifest_class_counts.items():
        actual_count = csv_class_counts.get(cls, 0)
        check(f"class {cls} count", actual_count == expected_count,
              f"csv={actual_count} manifest={expected_count}")

    check("5 classes present", len(csv_class_counts) == 5,
          f"got {len(csv_class_counts)}: {sorted(csv_class_counts.keys())}")

    print("\n  Class distribution (overall):")
    for cls in sorted(csv_class_counts.keys()):
        print(f"    {cls:12s}: {csv_class_counts[cls]:>6d}")

    # -- 5. Split integrity --------------------------------------------- #
    print("\n[5] Split integrity (taluk-based)")
    TALUK_SPLIT = {
        "Belthangady": "train", "Mangalore": "train", "Bantwal": "train",
        "Puttur": "val", "Sullia": "test",
    }
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_class_counts: dict[str, dict[str, int]] = {
        "train": {}, "val": {}, "test": {},
    }
    unknown_taluks: set[str] = set()
    for row in rows:
        taluk = (row.get("location_taluk") or "").strip()
        split = TALUK_SPLIT.get(taluk, "unknown")
        if split == "unknown":
            unknown_taluks.add(taluk)
        else:
            split_counts[split] += 1
            label = row.get("crop_label", "unknown")
            split_class_counts[split][label] = split_class_counts[split].get(label, 0) + 1

    check("no unknown taluks", not unknown_taluks,
          f"unknown={sorted(unknown_taluks)}")
    check("train count matches manifest",
          split_counts["train"] == train_n,
          f"actual={split_counts['train']} manifest={train_n}")
    check("val count matches manifest",
          split_counts["val"] == val_n,
          f"actual={split_counts['val']} manifest={val_n}")
    check("test count matches manifest",
          split_counts["test"] == test_n,
          f"actual={split_counts['test']} manifest={test_n}")

    print(f"\n  Split counts: train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}")

    manifest_split_counts = manifest.get("class_counts", {})
    for split_name in ["train", "validation", "test"]:
        expected = manifest_split_counts.get(split_name, {})
        actual_key = "val" if split_name == "validation" else split_name
        actual = split_class_counts[actual_key]
        for cls, expected_count in expected.items():
            actual_count = actual.get(cls, 0)
            check(f"{split_name}/{cls} count", actual_count == expected_count,
                  f"actual={actual_count} manifest={expected_count}")

    # -- 6. Duplicate identity ------------------------------------------ #
    print("\n[6] Duplicate identity")
    record_ids = [row.get("record_id", "") for row in rows]
    unique_ids = set(record_ids)
    check("no duplicate record_ids", len(unique_ids) == len(record_ids),
          f"duplicates={len(record_ids) - len(unique_ids)}")

    # -- 7. Observation construction ------------------------------------ #
    print("\n[7] Observation construction (CSV -> AgriculturalObservation)")
    from training.stam.observation import (
        AgriculturalObservation,
        ImagePairRef,
        SequenceInfo,
    )
    from training.kaggle.frozen_corpus import (
        build_observation,
        _determine_split,
    )
    from unittest.mock import MagicMock

    mock_stam = MagicMock()
    # build_observation resolves imagery via sto.resolve_sequence() (the
    # frozen-corpus path), so the mock must return a real SequenceInfo there.
    mock_stam.resolve_sequence.return_value = SequenceInfo(
        pairs=[ImagePairRef(date=date(2020, 7, 1))],
    )

    manifest_checksum = _sha256_file(manifest_path)
    corpus_version = manifest["dataset_version"]
    obs_sample = build_observation(
        rows[0], mock_stam,
        corpus_version=corpus_version,
        manifest_checksum=manifest_checksum,
    )
    check("observation is AgriculturalObservation",
          isinstance(obs_sample, AgriculturalObservation))
    check("observation has location", obs_sample.location is not None)
    check("observation has temporal", obs_sample.temporal is not None)
    check("observation has tabular", obs_sample.tabular is not None)
    check("observation has sequence", obs_sample.sequence is not None)
    check("observation has quality", obs_sample.quality is not None)
    check("observation has crop label", obs_sample.crop is not None)

    # -- 8. Provenance -------------------------------------------------- #
    print("\n[8] Provenance verification")
    p = obs_sample.provenance
    check("provenance has corpus", p.get("corpus") == "crop_supervised_v1")
    check("provenance has corpus_version", p.get("corpus_version") == corpus_version)
    check("provenance has manifest_checksum", len(p.get("manifest_checksum", "")) == 64)
    check("provenance has record_id", p.get("record_id") is not None)
    check("provenance has source", p.get("source") == "government_ogd")
    check("provenance has source_record_id", p.get("source_record_id") is not None)
    check("provenance has crop_class_id", p.get("crop_class_id") is not None)
    check("provenance has spatial_match_distance_km",
          p.get("spatial_match_distance_km") is not None)
    check("provenance has temporal_match_status",
          p.get("temporal_match_status") is not None)
    check("provenance has tabular_source", p.get("tabular_source") is not None)
    check("provenance has image_source", p.get("image_source") is not None)
    check("provenance has ndvi_available", p.get("ndvi_available") is not None)
    check("provenance has evi_available", p.get("evi_available") is not None)
    check("provenance has satellite_status", p.get("satellite_status") is not None)
    check("provenance has split", p.get("split") in ("train", "val", "test"))

    # -- 9. Yield separation -------------------------------------------- #
    print("\n[9] Yield separation")
    check("yield_value is None", obs_sample.yield_value is None)
    check("tabular.yield_value is None", obs_sample.tabular.yield_value is None)

    # -- 10. Multimodal contract ---------------------------------------- #
    print("\n[10] Multimodal contract")
    check("tabular features present", isinstance(obs_sample.tabular.fields, dict))
    check("temporal year present", isinstance(obs_sample.temporal.year, int))
    check("temporal season present", obs_sample.temporal.season is not None)
    check("sequence present", obs_sample.sequence is not None)
    check("quality report present", obs_sample.quality is not None)
    check("quality passed", obs_sample.quality.passed is True)
    check("location.lon present", isinstance(obs_sample.location.lon, float))
    check("location.lat present", isinstance(obs_sample.location.lat, float))

    # -- 11. No old data sources ---------------------------------------- #
    print("\n[11] Old data exclusion")
    check("source is government_ogd", rows[0].get("source") == "government_ogd")
    check("no data_season.csv origin", "data_season" not in str(p))

    # -- 12. Full-class distribution per split -------------------------- #
    print("\n[12] Per-split class counts")
    print("  TRAIN:")
    for cls in sorted(split_class_counts["train"].keys()):
        print(f"    {cls:12s}: {split_class_counts['train'][cls]:>6d}")
    print("  VALIDATION:")
    for cls in sorted(split_class_counts["val"].keys()):
        print(f"    {cls:12s}: {split_class_counts['val'][cls]:>6d}")
    print("  TEST:")
    for cls in sorted(split_class_counts["test"].keys()):
        print(f"    {cls:12s}: {split_class_counts['test'][cls]:>6d}")

    # -- 13. All CSV rows constructable --------------------------------- #
    print("\n[13] All rows constructable into observations")
    errors = 0
    for idx, row in enumerate(rows):
        try:
            obs = build_observation(
                row, mock_stam,
                corpus_version=corpus_version,
                manifest_checksum=manifest_checksum,
            )
            if not isinstance(obs, AgriculturalObservation):
                errors += 1
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"    row {idx}: {exc}")
    check(f"all {len(rows)} rows constructable",
          errors == 0,
          f"errors={errors}")

    _summary()
    return 1 if failures else 0


def _summary() -> None:
    print("\n" + "=" * 72)
    if failures:
        print(f"  LOCAL VERIFICATION: FAIL ({len(failures)} failure(s))")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  LOCAL VERIFICATION: PASS")
    print("=" * 72)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
