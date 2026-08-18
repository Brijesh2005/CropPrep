"""R5.2.1 Task D+E: Data contract + evaluation split verification.

Verifies:
  D1. Mixed yield units are detected (kg/ha + NPP proxy mixing)
  D2. Crop head without labels is caught
  D3. Training corpus is not empty
  E1. Temporal split: val/test have zero crop labels
  E2. Temporal split: all crop-labeled observations in train
  E3. Yield distribution constant across val/test
  E4. Split composition summary

Run from repo root::

    python training/kaggle/scripts/verify_data_contract_and_split.py \
        --corpus training/kaggle/outputs/reports/corpus.json \
        --output training/artifacts/contract_split
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from training.preprocessing.data_contract import (  # noqa: E402
    TrainingDataContract,
    assess_training_data_contract,
)
from training.preprocessing.dataset import split_observations  # noqa: E402
from training.preprocessing.master_pipeline import Preprocessor  # noqa: E402
from training.preprocessing.config import load_preprocessing_config  # noqa: E402
from training.stam.observation import AgriculturalObservation  # noqa: E402


def _load_accepted(path: Path) -> list[AgriculturalObservation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    obs = []
    for s in raw["samples"]:
        if s["status"] == "accepted" and s.get("observation") is not None:
            obs.append(AgriculturalObservation.model_validate(s["observation"]))
    return obs


def _verify_data_contract(obs: list[AgriculturalObservation]) -> dict[str, Any]:
    """Task D: Verify data contract catches issues."""
    print("\n=== TASK D: DATA CONTRACT VERIFICATION ===\n")

    # D1: Full corpus with crop head enabled
    report_full = assess_training_data_contract(obs, crop_head_enabled=True)
    print(f"D1. Full corpus ({len(obs)} obs):")
    print(f"    valid={report_full.valid}, errors={report_full.errors}")
    print(f"    warnings={report_full.warnings}")
    print(f"    crop_training_samples={report_full.crop_training_samples}")
    print(f"    yield_training_samples={report_full.yield_training_samples}")
    print(f"    yield_unit={report_full.yield_unit}")
    print(f"    yield_source={report_full.yield_source}")
    print()

    # D2: Empty corpus
    report_empty = assess_training_data_contract([], crop_head_enabled=True)
    print(f"D2. Empty corpus:")
    print(f"    valid={report_empty.valid}, errors={report_empty.errors}")
    print()

    # D3: Crop head without labels
    no_crop = [o for o in obs if getattr(o, "crop", None) is None]
    if no_crop:
        report_nolabel = assess_training_data_contract(no_crop, crop_head_enabled=True)
        print(f"D3. Unlabeled-only corpus ({len(no_crop)} obs):")
        print(f"    valid={report_nolabel.valid}, errors={report_nolabel.errors}")
        print(f"    crop_training_samples={report_nolabel.crop_training_samples}")
    else:
        print("D3. All observations have crop labels — skipping unlabeled-only test")
    print()

    return {
        "d1_full_corpus": report_full.to_dict(),
        "d2_empty_corpus": report_empty.to_dict(),
        "d3_unlabeled_only": report_nolabel.to_dict() if no_crop else None,
        "contract_valid": report_full.valid,
        "contract_errors": report_full.errors,
        "contract_warnings": report_full.warnings,
    }


def _verify_split_composition(
    obs: list[AgriculturalObservation],
) -> dict[str, Any]:
    """Task E: Verify evaluation split composition."""
    print("\n=== TASK E: EVALUATION SPLIT VERIFICATION ===\n")

    pre = Preprocessor(load_preprocessing_config())
    accepted, _ = pre.filter(obs)

    train, val, test = split_observations(accepted, pre.config.split)

    # Split composition
    splits_info = {}
    for name, split_obs in [("train", train), ("val", val), ("test", test)]:
        crop_labeled = [o for o in split_obs if getattr(o, "crop", None) is not None]
        yield_vals = [float(o.yield_value) for o in split_obs
                      if getattr(o, "yield_value", None) is not None]
        has_paired = sum(1 for o in split_obs
                         if getattr(o, "has_paired_images", False))

        splits_info[name] = {
            "total": len(split_obs),
            "crop_labeled": len(crop_labeled),
            "yield_values": len(yield_vals),
            "has_paired_images": has_paired,
            "yield_unique": len(set(round(v, 4) for v in yield_vals)) if yield_vals else 0,
            "years": sorted(set(
                int(getattr(o.temporal, "year", 0)) for o in split_obs
            )),
        }
        print(f"  {name}:")
        print(f"    total={len(split_obs)}")
        print(f"    crop_labeled={len(crop_labeled)}")
        print(f"    yield_values={len(yield_vals)}")
        print(f"    has_paired_images={has_paired}")
        print(f"    yield_unique_values={splits_info[name]['yield_unique']}")
        print(f"    years={splits_info[name]['years']}")

    # E1: val/test have zero crop labels
    val_no_crop = splits_info["val"]["crop_labeled"] == 0
    test_no_crop = splits_info["test"]["crop_labeled"] == 0
    print(f"\n  E1. Val has zero crop labels: {val_no_crop}")
    print(f"      Test has zero crop labels: {test_no_crop}")

    # E2: All crop-labeled in train
    total_crop = splits_info["train"]["crop_labeled"] + \
                 splits_info["val"]["crop_labeled"] + \
                 splits_info["test"]["crop_labeled"]
    train_has_all_crop = (splits_info["train"]["crop_labeled"] == total_crop)
    print(f"  E2. Train has ALL {total_crop} crop labels: {train_has_all_crop}")

    # E3: Yield distribution
    train_yields = [float(o.yield_value) for o in train
                    if getattr(o, "yield_value", None) is not None]
    if train_yields:
        yield_min = min(train_yields)
        yield_max = max(train_yields)
        yield_const = (yield_min == yield_max)
        print(f"  E3. Train yields range: [{yield_min:.4g}, {yield_max:.4g}]")
        print(f"      Constant yield: {yield_const}")
    else:
        print("  E3. No yield values in train split")
        yield_const = True

    # E4: Crop evaluation readiness
    crop_eval_ready = (splits_info["val"]["crop_labeled"] > 0 and
                       splits_info["test"]["crop_labeled"] > 0)
    print(f"\n  E4. Crop evaluation ready (val + test have labels): {crop_eval_ready}")
    if not crop_eval_ready:
        print(f"      ISSUE: Cannot evaluate crop classifier on val/test!")
        print(f"      Val crop labels: {splits_info['val']['crop_labeled']}")
        print(f"      Test crop labels: {splits_info['test']['crop_labeled']}")

    return {
        "split_strategy": pre.config.split.strategy,
        "split_config": {
            "train_ratio": pre.config.split.train_ratio,
            "val_ratio": pre.config.split.val_ratio,
            "test_ratio": pre.config.split.test_ratio,
        },
        "train": splits_info["train"],
        "val": splits_info["val"],
        "test": splits_info["test"],
        "val_no_crop_labels": val_no_crop,
        "test_no_crop_labels": test_no_crop,
        "train_has_all_crop_labels": train_has_all_crop,
        "yield_constant_in_train": yield_const if train_yields else None,
        "crop_eval_ready": crop_eval_ready,
        "total_crop_labeled": total_crop,
        "issues": [
            "val/test have zero crop labels" if val_no_crop or test_no_crop else None,
            "all crop labels in train" if train_has_all_crop else None,
            "yield is constant" if yield_const and train_yields else None,
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-verify-data-contract-and-split",
        description="R5.2.1 Tasks D+E: data contract + evaluation split verification",
    )
    parser.add_argument("--corpus", required=True,
                        help="corpus.json path (accepted samples)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    obs = _load_accepted(Path(args.corpus))
    print(f"Loaded {len(obs)} accepted observations")

    contract_result = _verify_data_contract(obs)
    split_result = _verify_split_composition(obs)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    issues = []
    if not contract_result["contract_valid"]:
        issues.append("Data contract INVALID")
    if split_result["val_no_crop_labels"] or split_result["test_no_crop_labels"]:
        issues.append("Val/test have zero crop labels")
    if split_result["yield_constant_in_train"]:
        issues.append("Yield values constant across training set")
    if not split_result["crop_eval_ready"]:
        issues.append("Cannot evaluate crop classifier on val/test")

    print(f"  Data contract valid: {contract_result['contract_valid']}")
    print(f"  Crop eval ready: {split_result['crop_eval_ready']}")
    print(f"  Issues found: {len(issues)}")
    for issue in issues:
        print(f"    - {issue}")

    report = {
        "task_d_contract": contract_result,
        "task_e_split": split_result,
        "issues": issues,
    }

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "contract_split_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {report_path}")

    return 1 if issues else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
