"""Post-training diagnostic: per-class metrics, confusion matrix, prediction distribution.

Loads a frozen corpus, refits the preprocessor, rebuilds the model from a
checkpoint, runs inference on a chosen split, and emits a structured JSON
report plus a human-readable summary table.

Requires imagery (Kaggle mount). Exits 0 with a clear message if imagery
is unavailable.

Run from repo root (Kaggle kernel, after ``run_pipeline.py``)::

    python training/kaggle/scripts/diagnose_model.py \\
        --checkpoint training/kaggle/outputs/cropfusion_v5/checkpoints/best.pt \\
        --corpus training/kaggle/outputs/reports/frozen_corpus.json \\
        --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_repo_root(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    root = str(repo_root)
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    repo_training = (repo_root / "training").resolve()
    for entry in list(sys.path):
        if entry == root or entry == "":
            continue
        shadow = Path(entry) / "training"
        if shadow.exists() and shadow.resolve() != repo_training:
            sys.path.remove(entry)


_add_repo_root(_REPO_ROOT)

from training.dataset_manager import DatasetManager, load_settings  # noqa: E402
from training.models.factory import ModelFactory  # noqa: E402
from training.preprocessing import Preprocessor, load_preprocessing_config  # noqa: E402
from training.preprocessing.dataloader import build_dataloader  # noqa: E402
from training.preprocessing.dataset import CropFusionDataset, split_observations  # noqa: E402
from training.stam import STAM  # noqa: E402
from training.stam.config import load_stam_config  # noqa: E402


def _load_observations(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [s for s in raw["samples"] if s["status"] == "accepted" and s.get("observation") is not None]


def _per_class_metrics(
    all_preds: list[int], all_targets: list[int], class_names: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for idx, name in enumerate(class_names):
        tp = sum(1 for p, t in zip(all_preds, all_targets) if p == idx and t == idx)
        fp = sum(1 for p, t in zip(all_preds, all_targets) if p == idx and t != idx)
        fn = sum(1 for p, t in zip(all_preds, all_targets) if p != idx and t == idx)
        support = sum(1 for t in all_targets if t == idx)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            "class": name,
            "index": idx,
            "support": support,
            "predicted": sum(1 for p in all_preds if p == idx),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-diagnose-model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--preprocessing-config", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--stam-config", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    corpus_path = Path(args.corpus)
    ckpt_path = Path(args.checkpoint)
    output_dir = Path(args.output) if args.output else _REPO_ROOT / "training/kaggle/outputs/reports"
    pre_config = Path(args.preprocessing_config) if args.preprocessing_config else _REPO_ROOT / "training/config/preprocessing.yaml"
    ds_config = Path(args.dataset_config) if args.dataset_config else _REPO_ROOT / "training/config/dataset.yaml"
    stam_config = Path(args.stam_config) if args.stam_config else _REPO_ROOT / "training/config/stam.yaml"

    print("=" * 66)
    print("  POST-TRAINING DIAGNOSTIC")
    print("=" * 66)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    # ── Load observations ──────────────────────────────────────────────
    print("  loading frozen corpus...")
    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    from training.stam.observation import AgriculturalObservation
    all_obs: list[AgriculturalObservation] = []
    for sample in raw["samples"]:
        if sample["status"] == "accepted" and sample.get("observation") is not None:
            obs = AgriculturalObservation.model_validate(sample["observation"])
            obs.provenance = dict(sample.get("provenance") or obs.provenance or {})
            all_obs.append(obs)
    print(f"  {len(all_obs)} observations loaded")

    # ── Resolve imagery ────────────────────────────────────────────────
    print("  resolving imagery (STAM)...")
    try:
        manager = DatasetManager(load_settings(ds_config))
        manifest = manager.provider_manifests().get("kaggle_hub_image", {})
        if not manifest.get("available"):
            raise RuntimeError("imagery catalog not available")
        stam = STAM(manager, load_stam_config(stam_config))
        stam.initialize()
        extractor = stam.get_patch
    except Exception as exc:
        print(f"  SKIPPED: {exc}")
        return 0

    # ── Preprocessor fit ───────────────────────────────────────────────
    print("  fitting preprocessor on train split...")
    preprocessing_cfg = load_preprocessing_config(pre_config)
    train_obs, val_obs, test_obs = split_observations(all_obs, preprocessing_cfg.split)
    if args.split == "val":
        eval_obs = val_obs
    else:
        eval_obs = test_obs
    print(f"  train={len(train_obs)}  val={len(val_obs)}  test={len(test_obs)}  eval={len(eval_obs)}")

    pre = Preprocessor(preprocessing_cfg)
    try:
        pre.fit(train_obs, extractor=extractor)
    except Exception as exc:
        print(f"  preprocessor fit failed: {exc}")
        return 1

    # ── Build eval DataLoader ──────────────────────────────────────────
    print("  building eval DataLoader...")
    ds = CropFusionDataset.build(pre, eval_obs, split=args.split, extractor=extractor)
    loader = build_dataloader(ds, config=preprocessing_cfg, split=args.split, batch_size=args.batch_size)
    print(f"  eval samples: {len(eval_obs)}")

    # ── Load model from checkpoint ─────────────────────────────────────
    print(f"  loading model from {ckpt_path}...")
    try:
        model = ModelFactory.from_checkpoint(str(ckpt_path))
    except Exception as exc:
        print(f"  failed to load checkpoint: {exc}")
        return 1
    model.to(device)
    model.eval()

    # ── Class names from label encoder ─────────────────────────────────
    class_names = list(pre.label.crop_encoder.classes_)
    n_classes = len(class_names)
    print(f"  classes ({n_classes}): {class_names}")

    # ── Inference ──────────────────────────────────────────────────────
    print("  running inference...")
    all_preds_np: list[np.ndarray] = []
    all_targets_np: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            inputs = {k: batch_dev[k] for k in ("tabular", "ndvi", "evi", "temporal_mask") if k in batch_dev}
            out = model(inputs)
            logits = out.crop_logits if hasattr(out, "crop_logits") else out.get("crop_logits")
            if logits is None:
                continue
            all_preds_np.append(logits.argmax(dim=-1).cpu().numpy())
            all_targets_np.append(batch_dev["crop_label"].cpu().numpy())

    if not all_preds_np:
        print("  no predictions collected")
        return 1

    all_preds = np.concatenate(all_preds_np)
    all_targets = np.concatenate(all_targets_np)
    assert len(all_preds) == len(all_targets)

    # ── Overall metrics ────────────────────────────────────────────────
    correct = int((all_preds == all_targets).sum())
    total = len(all_preds)
    accuracy = correct / total if total else 0.0

    macro_f1 = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    per_class = _per_class_metrics(all_preds.tolist(), all_targets.tolist(), class_names)
    non_zero = [r for r in per_class if r["support"] > 0]
    if non_zero:
        macro_f1 = sum(r["f1"] for r in non_zero) / len(non_zero)
        macro_precision = sum(r["precision"] for r in non_zero) / len(non_zero)
        macro_recall = sum(r["recall"] for r in non_zero) / len(non_zero)

    # ── Prediction distribution ────────────────────────────────────────
    pred_dist = {name: int((all_preds == i).sum()) for i, name in enumerate(class_names)}
    target_dist = {name: int((all_targets == i).sum()) for i, name in enumerate(class_names)}

    # ── Confusion matrix ───────────────────────────────────────────────
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(all_preds, all_targets):
        cm[int(t), int(p)] += 1

    # ── Build report ───────────────────────────────────────────────────
    report: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "n_eval_samples": total,
        "class_names": class_names,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "prediction_distribution": pred_dist,
        "target_distribution": target_dist,
        "confusion_matrix": cm.tolist(),
    }

    # ── Print summary ──────────────────────────────────────────────────
    print()
    print(f"  ---- OVERALL  split={args.split}  n={total} ----")
    print(f"  accuracy:       {accuracy:.4f}")
    print(f"  macro F1:       {macro_f1:.4f}")
    print(f"  macro precision:{macro_precision:.4f}")
    print(f"  macro recall:   {macro_recall:.4f}")

    print()
    print("  ---- PER-CLASS METRICS ----")
    hdr = f"  {'class':<12}{'support':>9}{'pred':>7}{'prec':>7}{'recall':>7}{'f1':>7}"
    print(hdr)
    for r in per_class:
        print(f"  {r['class']:<12}{r['support']:>9}{r['predicted']:>7}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}")

    print()
    print("  ---- PREDICTION DISTRIBUTION ----")
    for name in class_names:
        n_pred = pred_dist.get(name, 0)
        n_true = target_dist.get(name, 0)
        pct = n_pred / total * 100 if total else 0.0
        print(f"  {name:<12} predicted={n_pred:>6} ({pct:5.1f}%)  actual={n_true:>6}")

    print()
    print("  ---- CONFUSION MATRIX (rows=actual, cols=predicted) ----")
    header = "  " + "".join(f"{name[:6]:>8}" for name in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = cm[i]
        print(f"  {name[:6]:<6}" + "".join(f"{v:>8}" for v in row))

    # ── Emit JSON ──────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "diagnostic_report.json"
    target.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote report -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
