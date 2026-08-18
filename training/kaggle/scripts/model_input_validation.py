"""R5.2.2: Model input validation — tensor finite/valid checks.

For every supervised batch, verify:
  - tabular tensor finite
  - image tensor finite
  - temporal tensor finite
  - crop target valid when crop task active
  - yield target valid when yield task active

nan_policy remains: stop.

Run from Kaggle training kernel (needs GPU + data)::

    python training/kaggle/scripts/model_input_validation.py \
        --corpus training/kaggle/outputs/reports/corpus.json \
        --output training/artifacts/input_validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from training.preprocessing import (  # noqa: E402
    Preprocessor,
    load_preprocessing_config,
    split_observations,
)
from training.preprocessing.dataset import CropFusionDataset  # noqa: E402
from training.preprocessing.dataloader import DataloaderConfig, build_dataloader  # noqa: E402
from training.preprocessing.supervised_contract import build_contract  # noqa: E402
from training.stam.observation import AgriculturalObservation  # noqa: E402


def _load_accepted(path: Path) -> list[AgriculturalObservation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        AgriculturalObservation.model_validate(s["observation"])
        for s in raw["samples"]
        if s["status"] == "accepted" and s.get("observation")
    ]


def _validate_batch(
    batch: dict[str, Any],
    *,
    check_crop: bool = True,
    check_yield: bool = True,
) -> dict[str, Any]:
    """Validate a single batch for finite values and correct targets."""
    issues: list[str] = []

    # Tabular
    if "tabular" in batch:
        t = batch["tabular"]
        if not torch.isfinite(t).all():
            nan = int(torch.isnan(t).sum().item())
            inf = int(torch.isinf(t).sum().item())
            issues.append(f"tabular: {nan} NaN, {inf} Inf")

    # NDVI
    if "ndvi" in batch:
        t = batch["ndvi"]
        if not torch.isfinite(t).all():
            nan = int(torch.isnan(t).sum().item())
            inf = int(torch.isinf(t).sum().item())
            issues.append(f"ndvi: {nan} NaN, {inf} Inf")

    # EVI
    if "evi" in batch:
        t = batch["evi"]
        if not torch.isfinite(t).all():
            nan = int(torch.isnan(t).sum().item())
            inf = int(torch.isinf(t).sum().item())
            issues.append(f"evi: {nan} NaN, {inf} Inf")

    # Temporal mask
    if "temporal_mask" in batch:
        t = batch["temporal_mask"]
        if not torch.isfinite(t).all():
            issues.append("temporal_mask: non-finite values")

    # Crop target
    if check_crop and "crop_label" in batch:
        t = batch["crop_label"]
        if not t.is_floating_point():
            invalid = (t < 0).sum().item()
            if invalid > 0:
                issues.append(f"crop_label: {invalid} negative (invalid) labels")
        else:
            issues.append("crop_label: expected integer, got float")

    # Yield target
    if check_yield and "yield_label" in batch:
        t = batch["yield_label"]
        if not torch.isfinite(t).all():
            nan = int(torch.isnan(t).sum().item())
            inf = int(torch.isinf(t).sum().item())
            issues.append(f"yield_label: {nan} NaN, {inf} Inf")

    return {
        "issues": issues,
        "passed": len(issues) == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-model-input-validation",
        description="R5.2.2: Validate model inputs for finite values and correct targets",
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--config",
                        default=str(_REPO_ROOT / "training" / "config" / "preprocessing.yaml"))
    args = parser.parse_args(argv)

    obs = _load_accepted(Path(args.corpus))
    print(f"Loaded {len(obs)} accepted observations")

    # Build contract to separate datasets
    contract = build_contract(obs)
    crop_obs = contract.crop_dataset.observations
    yield_obs = contract.yield_dataset.observations

    print(f"  Crop dataset: {len(crop_obs)}")
    print(f"  Yield dataset: {len(yield_obs)}")

    # Preprocessor
    pre = Preprocessor(load_preprocessing_config(args.config))
    if crop_obs:
        accepted, _ = pre.filter(crop_obs)
        pre.fit(accepted)
    elif yield_obs:
        accepted, _ = pre.filter(yield_obs)
        pre.fit(accepted)
    else:
        print("ERROR: No observations to validate")
        return 1

    # Build loaders
    crop_split = contract.crop_split
    yield_split = contract.yield_split

    loader_cfg = DataloaderConfig(batch_size=args.batch_size, workers=0)

    results = {"batches_validated": 0, "issues_found": 0, "all_passed": True}

    # Validate crop batches
    if crop_split.get("train"):
        print("\n--- Validating crop train batches ---")
        dataset = CropFusionDataset.build(pre, crop_split["train"], split="train")
        loader = build_dataloader(dataset, loader_cfg, split="train")
        for i, batch in enumerate(loader):
            result = _validate_batch(batch, check_crop=True, check_yield=False)
            results["batches_validated"] += 1
            if not result["passed"]:
                results["issues_found"] += 1
                results["all_passed"] = False
                print(f"  batch {i}: ISSUES: {result['issues']}")
        print(f"  validated {results['batches_validated']} crop batches")

    # Validate yield batches
    if yield_split.get("train"):
        print("\n--- Validating yield train batches ---")
        n_before = results["batches_validated"]
        dataset = CropFusionDataset.build(pre, yield_split["train"], split="train")
        loader = build_dataloader(dataset, loader_cfg, split="train")
        for i, batch in enumerate(loader):
            result = _validate_batch(batch, check_crop=False, check_yield=True)
            results["batches_validated"] += 1
            if not result["passed"]:
                results["issues_found"] += 1
                results["all_passed"] = False
                print(f"  batch {i}: ISSUES: {result['issues']}")
        print(f"  validated {results['batches_validated'] - n_before} yield batches")

    print(f"\n=== RESULT: {'PASS' if results['all_passed'] else 'FAIL'} ===")
    print(f"  Batches validated: {results['batches_validated']}")
    print(f"  Issues found: {results['issues_found']}")

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "input_validation_report.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nWrote {out / 'input_validation_report.json'}")

    return 0 if results["all_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
