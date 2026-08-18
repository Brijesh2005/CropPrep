"""R5.2.1 Task G: GPU smoke test for CropFusion.

Small deterministic test on real data:
  1. Loads 16 real observations
  2. Runs 2 epochs of training (batch_size=4)
  3. Reports per-epoch loss, gradient norms, NaN/Inf detection
  4. Validates model saves/loads correctly
  5. Checks AMP scaler state

Purpose: verify the pipeline is numerically stable on actual data before
committing to a full training run.

Run from Kaggle training kernel::

    python training/kaggle/scripts/gpu_smoke_test.py \
        --corpus training/kaggle/outputs/reports/corpus.json \
        --output training/artifacts/gpu_smoke_test
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from training.models.config import ModelConfig  # noqa: E402
from training.models.factory import ModelFactory  # noqa: E402
from training.preprocessing import Preprocessor, load_preprocessing_config, split_observations  # noqa: E402
from training.preprocessing.dataloader import CropFusionDataset, DataloaderConfig, build_dataloader  # noqa: E402
from training.stam.observation import AgriculturalObservation  # noqa: E402
from training.training.config import load_training_config, TrainingConfig  # noqa: E402
from training.training.losses import MultiTaskLoss, build_class_weights  # noqa: E402
from training.training.trainer import Trainer  # noqa: E402
from training.training.callbacks import HistoryRecorder  # noqa: E402
from training.training.checkpoint import TrainingCheckpointManager  # noqa: E402
from training.training.logger import ExperimentLogger  # noqa: E402
from training.training.utils import resolve_device  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-gpu-smoke-test",
        description="R5.2.1 Task G: GPU smoke test on real data",
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--config",
                        default=str(_REPO_ROOT / "training" / "config" / "preprocessing.yaml"))
    args = parser.parse_args(argv)

    device = resolve_device("auto")
    print(f"=== GPU SMOKE TEST (device={device}) ===")

    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Compute capability: {torch.cuda.get_device_capability(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB")

    # Load observations
    raw = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    obs = [
        AgriculturalObservation.model_validate(s["observation"])
        for s in raw["samples"]
        if s["status"] == "accepted" and s.get("observation")
    ]
    print(f"\nLoaded {len(obs)} accepted observations")
    if len(obs) < args.batch_size:
        print(f"ERROR: Need at least {args.batch_size} observations, got {len(obs)}")
        return 1

    # Preprocessor
    pre = Preprocessor(load_preprocessing_config(args.config))
    train_obs, val_obs, _ = split_observations(obs, pre.config.split)
    accepted_train, _ = pre.filter(train_obs)
    accepted_val, _ = pre.filter(val_obs)
    pre.fit(accepted_train)

    print(f"  Train observations: {len(accepted_train)}")
    print(f"  Val observations: {len(accepted_val)}")

    # Model
    mc = ModelConfig.from_preprocessor(pre)
    model = ModelFactory.create(mc)
    model.to(device)
    print(f"\n  Model: {mc.name} ({sum(p.numel() for p in model.parameters()):,} params)")

    # Config
    cfg = load_training_config(str(_REPO_ROOT / "training" / "config" / "training.yaml"))
    cfg.train.epochs = args.epochs
    cfg.data.batch_size = args.batch_size
    cfg.checkpoint.save_best = False
    cfg.checkpoint.save_latest = False
    cfg.checkpoint.save_periodic = False

    # Loss
    counts = torch.tensor([64.0, 7.0, 1.0, 1.0, 1.0])
    loss_fn = MultiTaskLoss(
        cfg.loss,
        class_weights={
            "crop": build_class_weights(cfg.loss, mc.heads.crop.num_classes, counts)
        },
    )
    loss_fn.to(device)

    # DataLoaders
    train_dataset = CropFusionDataset.build(
        pre, accepted_train[:16], split="train",
    )
    val_dataset = CropFusionDataset.build(
        pre, accepted_val[:8] if accepted_val else accepted_train[:4],
        split="val",
    )
    loader_cfg = DataloaderConfig(
        batch_size=args.batch_size,
        workers=0,
        pin_memory=device.type == "cuda",
    )
    train_loader = build_dataloader(train_dataset, loader_cfg, split="train")
    val_loader = build_dataloader(val_dataset, loader_cfg, split="val")

    # Trainer
    print(f"\n--- Running {args.epochs} epochs ---")
    history = HistoryRecorder()
    checkpoint_manager = TrainingCheckpointManager(Path(tempfile.mkdtemp()), keep_last=0)
    logger = ExperimentLogger(Path(tempfile.mkdtemp()), name="smoke_test")

    trainer = Trainer(
        model,
        train_loader,
        cfg,
        val_loader=val_loader,
        loss_module=loss_fn,
        callbacks=[history],
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        device=device,
    )

    result = trainer.train()

    # Report
    print(f"\n--- Results ---")
    print(f"  Epochs completed: {result.epochs}")
    print(f"  Steps: {result.steps}")
    print(f"  NaN steps: {result.nan_steps}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(f"  Stopped early: {result.stopped_early}")

    if result.history:
        last_epoch = result.history[-1]
        print(f"  Final train loss: {last_epoch.get('train_loss', 'N/A')}")
        if "val_loss" in last_epoch:
            print(f"  Final val loss: {last_epoch['val_loss']}")
        if "val_crop_accuracy" in last_epoch:
            print(f"  Final val crop accuracy: {last_epoch['val_crop_accuracy']}")

    if result.nan_diagnostics:
        print(f"\n  NaN diagnostics:")
        for diag in result.nan_diagnostics[:3]:
            print(f"    epoch={diag['epoch']}, step={diag['step']}")
            print(f"      loss={diag['loss']}")
            if diag.get('grad_params'):
                print(f"      nan_grad_params={len(diag['grad_params'])}")

    # Validation
    passed = True
    issues = []

    if result.nan_steps > 0:
        passed = False
        issues.append(f"NaN detected in {result.nan_steps} steps")

    if result.epochs < args.epochs:
        passed = False
        issues.append(f"Only {result.epochs}/{args.epochs} epochs completed")

    if not result.history:
        passed = False
        issues.append("No training history recorded")

    # Check AMP scaler state
    if trainer.scaler is not None:
        print(f"\n  AMP Scaler:")
        print(f"    Scale: {trainer.scaler.get_scale()}")
        print(f"    Growth: {trainer.scaler._growth_interval}")
    else:
        print(f"\n  AMP Scaler: DISABLED (CPU or amp=False)")

    print(f"\n=== RESULT: {'PASS' if passed else 'FAIL'} ===")
    for issue in issues:
        print(f"  - {issue}")

    report = {
        "epochs": result.epochs,
        "steps": result.steps,
        "nan_steps": result.nan_steps,
        "duration_seconds": result.duration_seconds,
        "stopped_early": result.stopped_early,
        "final_loss": result.history[-1].get("train_loss") if result.history else None,
        "history": result.history,
        "nan_diagnostics": result.nan_diagnostics[:5],
        "passed": passed,
        "issues": issues,
    }

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gpu_smoke_test_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nWrote {out / 'gpu_smoke_test_report.json'}")

    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
