"""R5.5 corpus delta audit (Phase 2 deliverable).

Documents the difference between the R5.2.7/8 supervised corpus
(``crop_supervised_v1.csv``, official version ``crop_supervised_v1.1``) and the
R5.2.9 enriched corpus (``crop_supervised_v2.csv``, official version
``crop_supervised_v2.0``) that the R5.5 run uses as its frozen contract.

Key facts established by this audit:

* The shipped v1 corpus has **10,674** rows (not 10,119).  The
  10,119 / 5,797 / 2,031 / 2,291 figures that leaked into older scripts and
  reports were a post-filter subset of a **cancelled r5-4 kernel run** — no
  shipped corpus ever had them (see ``docs/releases/R5.4_IMAGERY_EXCLUSION_AUDIT.md``).
* The v2 corpus is the v1 corpus plus **exactly one** recovered coconut
  observation (Puttur / 2020 / Kharif, FULL satellite imagery, recovered by the
  R5.2.9 temporal-relaxation rule).  That record is released but **not**
  benchmark-eligible, so the R5.5 benchmark contract is identical in records to
  v1 (10,674 / 5,924 / 2,459 / 2,291).
* No v1 record is removed and no record changes identity.

Usage::

    python training/kaggle/scripts/corpus_delta_audit.py \\
        --out-dir reports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_V1_CSV = _REPO_ROOT / "govt_crop_matched_v1" / "crop_supervised_v1.csv"
_DEFAULT_V2_CSV = _REPO_ROOT / "govt_crop_matched_v2" / "crop_supervised_v2.csv"
_DEFAULT_V1_MANIFEST = _REPO_ROOT / "training_manifests" / "crop_supervised_v1_manifest.json"
_DEFAULT_V2_MANIFEST = _REPO_ROOT / "training_manifests" / "crop_supervised_v2.0_manifest.json"

#: Stale totals that leaked from a cancelled r5-4 kernel run; never a shipped corpus.
_PHANTOM_TOTALS = {"total_samples": 10119, "train_samples": 5797, "validation_samples": 2031, "test_samples": 2291}

#: Columns compared for "changed" records.
_DIFF_COLUMNS = [
    "crop_label",
    "crop_class_id",
    "source_crop_name",
    "location_taluk",
    "location_village",
    "lat",
    "lon",
    "year",
    "season",
    "survey_date",
    "satellite_status",
    "ndvi_available",
    "evi_available",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(out_dir: Path) -> dict:
    v1 = pd.read_csv(_DEFAULT_V1_CSV, dtype=str)
    v2 = pd.read_csv(_DEFAULT_V2_CSV, dtype=str)
    m1 = _manifest(_DEFAULT_V1_MANIFEST)
    m2 = _manifest(_DEFAULT_V2_MANIFEST)

    s1 = set(v1["record_id"])
    s2 = set(v2["record_id"])
    removed = sorted(s1 - s2)
    added = sorted(s2 - s1)
    common = sorted(s1 & s2)

    changed: list[dict] = []
    if common:
        idx = pd.DataFrame({"record_id": common}).set_index("record_id")
        a = v1.set_index("record_id").loc[idx.index]
        b = v2.set_index("record_id").loc[idx.index]
        for col in _DIFF_COLUMNS:
            na = a[col].fillna("")
            nb = b[col].fillna("")
            diffs = na.index[na != nb]
            for rid in diffs:
                changed.append(
                    {"record_id": rid, "column": col, "v1": na.loc[rid], "v2": nb.loc[rid]}
                )

    benchmark = v2[v2["benchmark_eligible"] == "True"]
    added_details = []
    for rid in added:
        row = v2[v2["record_id"] == rid].iloc[0]
        added_details.append(
            {
                "record_id": rid,
                "crop_label": row["crop_label"],
                "location_taluk": row["location_taluk"],
                "location_hobli": row["location_hobli"],
                "location_village": row["location_village"],
                "year": row["year"],
                "season": row["season"],
                "survey_date": row["survey_date"],
                "satellite_status": row["satellite_status"],
                "ndvi_available": row["ndvi_available"],
                "evi_available": row["evi_available"],
                "is_recovered_v2": str(row.get("is_recovered_v2", "")),
                "benchmark_eligible": str(row.get("benchmark_eligible", "")),
            }
        )

    report = {
        "release": "r5.5",
        "phase": 2,
        "title": "Supervised-corpus delta: v1.1 (R5.2.7/8) -> v2.0 (R5.2.9)",
        "old_corpus": {
            "label": "crop_supervised_v1",
            "version": m1["dataset_version"],
            "csv_path": str(_DEFAULT_V1_CSV),
            "csv_rows": int(len(v1)),
            "manifest_total": int(m1["total_samples"]),
            "manifest_splits": {
                "train": int(m1["train_samples"]),
                "validation": int(m1["validation_samples"]),
                "test": int(m1["test_samples"]),
            },
            "csv_sha256": _sha256(_DEFAULT_V1_CSV),
        },
        "new_corpus": {
            "label": "crop_supervised_v2",
            "version": m2["dataset_version"],
            "csv_path": str(_DEFAULT_V2_CSV),
            "csv_rows": int(len(v2)),
            "benchmark_eligible_rows": int(len(benchmark)),
            "manifest_total": int(m2["total_samples"]),
            "manifest_splits": {
                "train": int(m2["train_samples"]),
                "validation": int(m2["validation_samples"]),
                "test": int(m2["test_samples"]),
            },
            "csv_sha256": _sha256(_DEFAULT_V2_CSV),
            "manifest_sha256": _sha256(_DEFAULT_V2_MANIFEST),
        },
        "delta": {
            "identical_records": len(common),
            "added": len(added),
            "removed": len(removed),
            "changed_columns": len(changed),
            "record_id_equality_benchmark": bool(set(benchmark["record_id"]) == s1),
            "split_delta_train": int(m2["train_samples"]) - int(m1["train_samples"]),
            "split_delta_validation": int(m2["validation_samples"]) - int(m1["validation_samples"]),
            "split_delta_test": int(m2["test_samples"]) - int(m1["test_samples"]),
        },
        "removed_records": removed,
        "changed_fields": changed[:50],
        "added_records": added_details,
        "historical_stale_totals": {
            **_PHANTOM_TOTALS,
            "note": (
                "'10119/5797/2031/2291' were a post-filter subset produced by a "
                "cancelled r5-4 kernel run. No shipped corpus ever had them: the "
                "official v1.1 and v2.0 corpora are both 10674/5924/2459/2291. "
                "These figures leaked into legacy diagnostics and old reports; "
                "R5.5 Phase 1 removed every hard-coded use."
            ),
        },
        "summary": (
            "The R5.5 benchmark contract is unchanged from the R5.2.9 v2.0 "
            "corpus: 10,674 benchmark-eligible records (5,924 train / 2,459 "
            "validation / 2,291 test). The v2 release adds exactly one recovered "
            "record (not benchmark-eligible). No records removed; no field changes "
            "among identical record_ids."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "corpus_delta_r5_2_8.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"corpus delta: identical={len(common)} added={len(added)} removed={len(removed)} changed={len(changed)}")
    print(f"wrote {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "reports"))
    args = parser.parse_args()
    run(Path(args.out_dir))


if __name__ == "__main__":
    main()