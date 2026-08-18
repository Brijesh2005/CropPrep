"""R5.2.1 Task H: Final readiness decision for CropFusion training.

Aggregates all R5.2.1 verification results and produces a pass/fail
readiness report with 7 acceptance criteria:

  A. Image branch verification (EfficientNetV2-S + NDVI/EVI)
  B. AMP stability (FP16/BF16/FP32 comparison)
  C. Numerical stability (no NaN/Inf in pipeline)
  D. Training data contract (mixed units, empty corpus, crop labels)
  E. Evaluation split quality (crop metrics evaluable on val/test)
  F. Multimodal tensor flow (all components, gradients, NaN-free)
  G. GPU smoke test (2 epochs on real data, no NaN, loss decreases)

Run from repo root::

    python training/kaggle/scripts/verify_readiness.py \
        --corpus training/kaggle/outputs/reports/corpus.json \
        --output training/artifacts/readiness_report
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
from training.preprocessing.data_contract import assess_training_data_contract  # noqa: E402
from training.preprocessing.dataset import split_observations  # noqa: E402
from training.preprocessing.master_pipeline import Preprocessor  # noqa: E402
from training.preprocessing.config import load_preprocessing_config  # noqa: E402


def _load_accepted(path: Path) -> list[AgriculturalObservation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        AgriculturalObservation.model_validate(s["observation"])
        for s in raw["samples"]
        if s["status"] == "accepted" and s.get("observation")
    ]


def _check_criteria(obs: list[AgriculturalObservation]) -> dict[str, Any]:
    """Evaluate all 7 readiness criteria."""
    criteria: dict[str, Any] = {}

    # --- A. Image branch verification --- #
    # Requires Kaggle GPU + imagery mount. Local assessment: check if
    # verify_image_tensors_full.py script exists and architecture supports it.
    img_script = _REPO_ROOT / "training" / "kaggle" / "scripts" / "verify_image_tensors_full.py"
    criteria["A"] = {
        "name": "Image Branch Verification",
        "status": "PENDING_KAGGLE",
        "reason": "Requires Kaggle GPU + imagery mount. Script exists: " + str(img_script.exists()),
        "acceptance": "Run verify_image_tensors_full.py on Kaggle",
        "pass": None,
    }

    # --- B. AMP stability --- #
    amp_script = _REPO_ROOT / "training" / "kaggle" / "scripts" / "amp_stability_test.py"
    criteria["B"] = {
        "name": "AMP Stability",
        "status": "PENDING_KAGGLE",
        "reason": "Requires GPU. Script exists: " + str(amp_script.exists()),
        "acceptance": "Run amp_stability_test.py on Kaggle; verify fp16_amp is stable",
        "pass": None,
    }

    # --- C. Numerical stability --- #
    # Check: training.yaml amp_dtype, performance.yaml dtype, nan_policy
    import yaml
    training_yaml = _REPO_ROOT / "training" / "config" / "training.yaml"
    perf_yaml = _REPO_ROOT / "training" / "config" / "performance.yaml"
    training_cfg = yaml.safe_load(training_yaml.read_text(encoding="utf-8"))
    perf_cfg = yaml.safe_load(perf_yaml.read_text(encoding="utf-8"))

    amp_dtype = training_cfg.get("general", {}).get("amp_dtype", "unknown")
    nan_policy = training_cfg.get("general", {}).get("nan_policy", "unknown")
    perf_dtype = perf_cfg.get("mixed_precision", {}).get("dtype", "unknown")
    grad_clip = training_cfg.get("general", {}).get("gradient_clip", None)

    c_issues = []
    if amp_dtype == "bf16":
        c_issues.append("amp_dtype=bf16 in training.yaml (P100 incompatible)")
    if perf_dtype == "bf16":
        c_issues.append("dtype=bf16 in performance.yaml (P100 incompatible)")
    if nan_policy != "stop":
        c_issues.append(f"nan_policy={nan_policy} (should be 'stop')")

    criteria["C"] = {
        "name": "Numerical Stability",
        "status": "PARTIAL_PASS" if c_issues else "PASS",
        "reason": "Config check on local machine",
        "details": {
            "training.yaml.amp_dtype": amp_dtype,
            "performance.yaml.dtype": perf_dtype,
            "nan_policy": nan_policy,
            "gradient_clip": grad_clip,
        },
        "issues": c_issues,
        "pass": len(c_issues) == 0,
    }

    # --- D. Training data contract --- #
    contract = assess_training_data_contract(obs, crop_head_enabled=True)
    d_issues = list(contract.errors)
    criteria["D"] = {
        "name": "Training Data Contract",
        "status": "PASS" if contract.valid else "FAIL",
        "crop_training_samples": contract.crop_training_samples,
        "yield_training_samples": contract.yield_training_samples,
        "yield_unit": contract.yield_unit,
        "yield_source": contract.yield_source,
        "errors": contract.errors,
        "warnings": contract.warnings,
        "pass": contract.valid,
    }

    # --- E. Evaluation split quality --- #
    pre = Preprocessor(load_preprocessing_config())
    accepted, _ = pre.filter(obs)
    train, val, test = split_observations(accepted, pre.config.split)

    val_crop = sum(1 for o in val if getattr(o, "crop", None) is not None)
    test_crop = sum(1 for o in test if getattr(o, "crop", None) is not None)
    train_crop = sum(1 for o in train if getattr(o, "crop", None) is not None)
    total_crop = train_crop + val_crop + test_crop

    e_issues = []
    if val_crop == 0:
        e_issues.append("val split has ZERO crop-labeled samples")
    if test_crop == 0:
        e_issues.append("test split has ZERO crop-labeled samples")
    if train_crop == total_crop and total_crop > 0:
        e_issues.append("ALL crop-labeled samples in train (val/test have none)")

    criteria["E"] = {
        "name": "Evaluation Split Quality",
        "status": "FAIL" if e_issues else "PASS",
        "split_strategy": pre.config.split.strategy,
        "train_crop_labeled": train_crop,
        "val_crop_labeled": val_crop,
        "test_crop_labeled": test_crop,
        "total_crop_labeled": total_crop,
        "issues": e_issues,
        "pass": len(e_issues) == 0,
    }

    # --- F. Multimodal tensor flow --- #
    multimodal_script = _REPO_ROOT / "training" / "kaggle" / "scripts" / "verify_multimodal_tensors.py"
    criteria["F"] = {
        "name": "Multimodal Tensor Flow",
        "status": "PENDING_KAGGLE",
        "reason": "Requires GPU. Script exists: " + str(multimodal_script.exists()),
        "acceptance": "Run verify_multimodal_tensors.py on Kaggle",
        "pass": None,
    }

    # --- G. GPU smoke test --- #
    smoke_script = _REPO_ROOT / "training" / "kaggle" / "scripts" / "gpu_smoke_test.py"
    criteria["G"] = {
        "name": "GPU Smoke Test",
        "status": "PENDING_KAGGLE",
        "reason": "Requires GPU. Script exists: " + str(smoke_script.exists()),
        "acceptance": "Run gpu_smoke_test.py on Kaggle (2 epochs, batch_size=4)",
        "pass": None,
    }

    return criteria


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-readiness-report",
        description="R5.2.1 Task H: Final readiness decision",
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    obs = _load_accepted(Path(args.corpus))
    print(f"=== R5.2.1 READINESS REPORT ===")
    print(f"Accepted observations: {len(obs)}")

    criteria = _check_criteria(obs)

    # Summary
    print(f"\n{'='*70}")
    print(f"{'CRITERION':<40} {'STATUS':<20} {'PASS'}")
    print(f"{'='*70}")
    for key in "ABCDEFG":
        c = criteria[key]
        status = c["status"]
        p = c.get("pass")
        p_str = "YES" if p is True else ("NO" if p is False else "PENDING")
        print(f"  {key}. {c['name']:<36} {status:<20} {p_str}")

    # Final decision
    decided = {k: v for k, v in criteria.items() if v.get("pass") is not None}
    passed = {k: v for k, v in decided.items() if v["pass"]}
    failed = {k: v for k, v in decided.items() if not v["pass"]}
    pending = {k: v for k, v in criteria.items() if v.get("pass") is None}

    print(f"\n{'='*70}")
    print(f"DECISION SUMMARY")
    print(f"{'='*70}")
    print(f"  Decided: {len(decided)}/7 ({len(passed)} PASS, {len(failed)} FAIL)")
    print(f"  Pending (require Kaggle GPU): {len(pending)}")
    if pending:
        print(f"    {', '.join(f'{k}. {v['name']}' for k, v in pending.items())}")

    # Local blockers
    local_blockers = [f"{k}. {v['name']}: {v.get('issues', v.get('errors', ['unknown']))}"
                      for k, v in failed.items()]
    if local_blockers:
        print(f"\n  LOCAL BLOCKERS:")
        for b in local_blockers:
            print(f"    - {b}")

    # Overall status
    if failed:
        overall = "BLOCKED"
        reason = "local criteria failed"
    elif pending:
        overall = "PROVISIONAL"
        reason = f"{len(pending)} criteria pending Kaggle execution"
    else:
        overall = "READY"
        reason = "all criteria pass"

    print(f"\n  OVERALL: {overall} ({reason})")

    # Action items
    print(f"\n  ACTION ITEMS:")
    if pending:
        print(f"  1. Upload scripts to Kaggle kernel:")
        for k in "ABFG":
            if criteria[k].get("pass") is None:
                script_name = {
                    "A": "verify_image_tensors_full.py",
                    "B": "amp_stability_test.py",
                    "F": "verify_multimodal_tensors.py",
                    "G": "gpu_smoke_test.py",
                }.get(k)
                print(f"     - training/kaggle/scripts/{script_name}")
        print(f"  2. Run each script on Kaggle P100 GPU")
        print(f"  3. Check all results pass")
    if failed:
        print(f"  1. Fix local blockers before proceeding:")
        for k, v in failed.items():
            for issue in v.get("issues", []):
                print(f"     - {k}: {issue}")

    report = {
        "criteria": criteria,
        "overall": overall,
        "reason": reason,
        "decided_count": len(decided),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "pending_count": len(pending),
        "local_blockers": local_blockers,
    }

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "readiness_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nWrote {out / 'readiness_report.json'}")

    return 0 if overall == "READY" else (2 if overall == "BLOCKED" else 1)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
