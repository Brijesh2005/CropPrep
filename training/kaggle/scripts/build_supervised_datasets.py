"""R5.2.2: Build supervised datasets, manifests, and evaluation reports.

Generates:
  - crop_dataset_manifest.json
  - yield_dataset_manifest.json
  - auxiliary_dataset_manifest.json
  - dataset_contract.json
  - crop_split_report.json
  - yield_split_report.json
  - leakage_report.json
  - crop_evaluation.json (placeholder — needs trained model)
  - yield_evaluation.json (placeholder — needs trained model)
  - readiness_report.json

Run from repo root::

    python training/kaggle/scripts/build_supervised_datasets.py \
        --corpus kaggle_runs/.../corpus.json \
        --output training/artifacts/supervised_contract
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from training.stam.observation import AgriculturalObservation  # noqa: E402
from training.preprocessing.supervised_contract import (  # noqa: E402
    SupervisedDataContract,
    DatasetDescriptor,
    build_contract,
)


def _load_accepted(path: Path) -> list[AgriculturalObservation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    obs = []
    for s in raw["samples"]:
        if s["status"] == "accepted" and s.get("observation") is not None:
            obs.append(AgriculturalObservation.model_validate(s["observation"]))
    return obs


def _write_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {path.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-build-supervised-datasets",
        description="R5.2.2: Build supervised data contract + all manifests",
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    obs = _load_accepted(Path(args.corpus))
    print(f"Loaded {len(obs)} accepted observations")

    # Build contract
    print("\nBuilding supervised data contract...")
    contract = build_contract(obs)

    # Print summary
    print(f"\n{'='*70}")
    print("SUPERVISED DATA CONTRACT SUMMARY")
    print(f"{'='*70}")
    print(f"  Crop dataset: {contract.crop_dataset.sample_count} samples")
    print(f"    classes: {contract.crop_dataset.class_counts}")
    print(f"    years: {contract.crop_dataset.years}")
    print(f"    locations: {len(contract.crop_dataset.locations)} unique")
    print(f"    split feasibility: {contract.crop_split_feasibility}")
    print(f"  Yield dataset: {contract.yield_dataset.sample_count} samples (kg/ha)")
    print(f"    yield stats: {contract.yield_dataset.yield_stats}")
    print(f"    years: {contract.yield_dataset.years}")
    print(f"  Auxiliary dataset: {contract.auxiliary_dataset.sample_count} samples (NPP proxy)")
    print(f"    years: {contract.auxiliary_dataset.years}")
    print(f"  Temporal crop generalization: {contract.temporal_crop_generalization}")
    print(f"  Overall valid: {contract.overall_validity}")
    if contract.errors:
        print(f"  ERRORS:")
        for e in contract.errors:
            print(f"    - {e}")
    if contract.warnings:
        print(f"  WARNINGS:")
        for w in contract.warnings:
            print(f"    - {w}")

    # --- Write manifests --- #
    print(f"\nWriting manifests to {out}...")

    # 1. Crop dataset manifest
    crop_manifest = contract.crop_dataset.to_dict()
    crop_manifest["split"] = {
        name: len(obs_list) for name, obs_list in contract.crop_split.items()
    }
    _write_json(crop_manifest, out / "crop_dataset_manifest.json")

    # 2. Yield dataset manifest
    yield_manifest = contract.yield_dataset.to_dict()
    yield_manifest["split"] = {
        name: len(obs_list) for name, obs_list in contract.yield_split.items()
    }
    _write_json(yield_manifest, out / "yield_dataset_manifest.json")

    # 3. Auxiliary dataset manifest
    _write_json(contract.auxiliary_dataset.to_dict(), out / "auxiliary_dataset_manifest.json")

    # 4. Dataset contract
    _write_json(contract.to_dict(), out / "dataset_contract.json")

    # 5. Crop split report
    crop_split_report = {
        "feasibility": contract.crop_split_feasibility,
        "splits": {
            name: {
                "count": len(obs_list),
                "classes": {
                    str(getattr(o, "crop", "unknown")): sum(
                        1 for o2 in obs_list if str(getattr(o2, "crop", "unknown")) == str(getattr(o, "crop", "unknown"))
                    )
                    for o in obs_list[:1]
                } if obs_list else {},
                "years": sorted(set(
                    int(getattr(getattr(o, "temporal", None), "year", 0))
                    for o in obs_list
                )),
                "locations": sorted(set(
                    _location_key_str(o) for o in obs_list
                )),
            }
            for name, obs_list in contract.crop_split.items()
        },
        "temporal_generalization": contract.temporal_crop_generalization,
    }
    _write_json(crop_split_report, out / "crop_split_report.json")

    # 6. Yield split report
    yield_split_report = {
        "splits": {
            name: {
                "count": len(obs_list),
                "yield_stats": _compute_stats(obs_list),
                "years": sorted(set(
                    int(getattr(getattr(o, "temporal", None), "year", 0))
                    for o in obs_list
                )),
            }
            for name, obs_list in contract.yield_split.items()
        },
    }
    _write_json(yield_split_report, out / "yield_split_report.json")

    # 7. Leakage report
    _write_json(contract.leakage_report, out / "leakage_report.json")

    # 8-9. Evaluation placeholders
    crop_eval = {
        "status": "PENDING",
        "reason": "Requires trained model. Placeholder for crop_evaluation.json",
        "metrics": {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "per_class_support": contract.crop_dataset.class_counts,
            "confusion_matrix": None,
        },
    }
    _write_json(crop_eval, out / "crop_evaluation.json")

    yield_eval = {
        "status": "PENDING",
        "reason": "Requires trained model. Placeholder for yield_evaluation.json",
        "metrics": {
            "mae": None,
            "rmse": None,
            "r2": None,
            "mape": None,
            "unit": "kg/ha",
        },
    }
    _write_json(yield_eval, out / "yield_evaluation.json")

    # 10. Readiness report
    readiness = _build_readiness(contract)
    _write_json(readiness, out / "readiness_report.json")

    # Final statement
    print(f"\n{'='*70}")
    print("READINESS STATEMENT")
    print(f"{'='*70}")
    for key in ["CROP_DATASET", "YIELD_DATASET", "MULTIMODAL_ALIGNMENT",
                "DATA_CONTRACT", "LEAKAGE_CHECK", "SMOKE_TEST"]:
        status = readiness.get(key, {}).get("status", "UNKNOWN")
        print(f"  {key:<30} {status}")

    overall = readiness.get("OVERALL", {}).get("status", "UNKNOWN")
    print(f"\n  OVERALL: {overall}")
    if overall == "FULL TRAINING NOT READY":
        print(f"  REASON: {readiness.get('OVERALL', {}).get('reason', 'unknown')}")

    return 0 if overall == "FULL TRAINING READY" else 1


def _location_key_str(o: Any) -> str:
    loc = getattr(o, "location", None)
    if loc is None:
        return "unknown"
    admin = getattr(loc, "admin", None)
    if admin is not None:
        village = getattr(admin, "village", None)
        if village:
            return str(village)
        district = getattr(admin, "district", None)
        if district:
            return str(district)
    return "unknown"


def _compute_stats(obs_list: list[Any]) -> dict[str, float]:
    import torch
    values = [
        float(getattr(o, "yield_value", 0))
        for o in obs_list
        if getattr(o, "yield_value", None) is not None
    ]
    if not values:
        return {}
    t = torch.tensor(values)
    return {
        "min": round(float(t.min().item()), 4),
        "max": round(float(t.max().item()), 4),
        "mean": round(float(t.mean().item()), 4),
        "std": round(float(t.std().item()), 4),
        "count": len(values),
    }


def _build_readiness(contract: SupervisedDataContract) -> dict[str, Any]:
    """Build the 6-criterion readiness report."""
    crop_ok = (
        contract.crop_dataset.sample_count > 0
        and contract.crop_split_feasibility != "CROP_DATA_INSUFFICIENT_FOR_CLASS_WISE_GENERALIZATION"
    )
    yield_ok = (
        contract.yield_dataset.sample_count > 0
        and contract.yield_dataset.yield_stats.get("std", 0) > 0.01
    )

    # Multimodal alignment: check all crop observations have required modalities
    crop_modalities_ok = (
        contract.crop_dataset.modalities.get("tabular", False)
        and contract.crop_dataset.modalities.get("temporal", False)
    )
    yield_modalities_ok = (
        contract.yield_dataset.modalities.get("tabular", False)
    )

    # Leakage
    crop_leakage = contract.leakage_report.get("crop", {})
    yield_leakage = contract.leakage_report.get("yield", {})
    leakage_ok = crop_leakage.get("passed", False) and yield_leakage.get("passed", False)

    # Data contract
    contract_ok = contract.overall_validity and len(contract.errors) == 0

    # Smoke test (requires Kaggle GPU — pending)
    smoke_ok = None

    results = {}

    results["CROP_DATASET"] = {
        "status": "READY" if crop_ok else "NOT READY",
        "reason": (
            f"{contract.crop_dataset.sample_count} crop-labeled samples, "
            f"{len(contract.crop_dataset.class_counts)} classes, "
            f"split: {contract.crop_split_feasibility}"
        ),
    }

    results["YIELD_DATASET"] = {
        "status": "READY" if yield_ok else "NOT READY",
        "reason": (
            f"{contract.yield_dataset.sample_count} kg/ha samples, "
            f"std={contract.yield_dataset.yield_stats.get('std', 0):.4f}"
        ),
    }

    results["MULTIMODAL_ALIGNMENT"] = {
        "status": "PASS" if (crop_modalities_ok and yield_modalities_ok) else "FAIL",
        "reason": (
            f"crop: {contract.crop_dataset.modalities}, "
            f"yield: {contract.yield_dataset.modalities}"
        ),
    }

    results["DATA_CONTRACT"] = {
        "status": "PASS" if contract_ok else "FAIL",
        "reason": "; ".join(contract.errors) if contract.errors else "All checks pass",
    }

    results["LEAKAGE_CHECK"] = {
        "status": "PASS" if leakage_ok else "FAIL",
        "reason": (
            "; ".join(crop_leakage.get("issues", []) + yield_leakage.get("issues", []))
            or "No leakage detected"
        ),
    }

    results["SMOKE_TEST"] = {
        "status": "PENDING",
        "reason": "Requires Kaggle GPU execution",
    }

    # Overall
    all_ok = crop_ok and yield_ok and contract_ok and leakage_ok
    if all_ok:
        overall_status = "FULL TRAINING READY"
        reason = "Both supervised tasks have valid evaluation support"
    else:
        overall_status = "FULL TRAINING NOT READY"
        blockers = []
        if not crop_ok:
            blockers.append("crop dataset")
        if not yield_ok:
            blockers.append("yield dataset")
        if not contract_ok:
            blockers.append("data contract violations")
        if not leakage_ok:
            blockers.append("data leakage detected")
        reason = f"Missing: {', '.join(blockers)}"

    results["OVERALL"] = {
        "status": overall_status,
        "reason": reason,
    }

    return results


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
