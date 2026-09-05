"""Post-training diagnostic: per-class metrics, confusion matrix, prediction distribution.

Loads a frozen corpus, respects its provenance-based train/val/test split,
refits the preprocessor on TRAIN ONLY, rebuilds the model from a checkpoint,
runs inference on the requested split, and emits a structured JSON report
plus a human-readable summary table.

Uses the FROZEN spatial split (taluk-based) — never calls split_observations().

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
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]

# ── Frozen contract (from training_manifests/crop_supervised_v2.0_manifest.json) ──
_FROZEN_MANIFEST = _REPO_ROOT / "training_manifests" / "crop_supervised_v2.0_manifest.json"
_SUPERVISED_CLASSES = ["coconut", "pepper", "coffee", "cardamom"]
_EXCLUDED_CLASSES = ["blackgram"]


def _load_frozen_contract(manifest_path: Path) -> dict[str, Any]:
    """Read the immutable frozen-corpus contract from the v2.0 manifest.

    The assertions stay strict: these values ARE the contract
    (total 10674 = train 5924 + val 2459 + test 2291).  Any future corpus
    change must ship a new manifest rather than silently bump a constant.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    total = int(data["total_samples"])
    train = int(data["train_samples"])
    val = int(data["validation_samples"])
    test = int(data["test_samples"])
    assert total == train + val + test, f"manifest total inconsistent: {total} != {train}+{val}+{test}"
    supervised = list(data.get("supervised_classes") or _SUPERVISED_CLASSES)
    excluded = list(data.get("excluded_classes") or _EXCLUDED_CLASSES)
    sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {
        "total": total,
        "train": train,
        "val": val,
        "test": test,
        "supervised_classes": supervised,
        "excluded_classes": excluded,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256,
        "dataset_checksums": data.get("reproducibility", {}).get("dataset_checksums", {}),
        "spatial_split": data.get("split_groups", {}),
    }


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
from training.preprocessing.dataset import CropFusionDataset  # noqa: E402
from training.stam import STAM  # noqa: E402
from training.stam.config import load_stam_config  # noqa: E402
from training.stam.observation import AgriculturalObservation  # noqa: E402


# ── Frozen-provenance split ────────────────────────────────────────────

def _split_by_frozen_provenance(
    observations: list[AgriculturalObservation],
) -> tuple[list[AgriculturalObservation], list[AgriculturalObservation], list[AgriculturalObservation]]:
    """Partition observations by their frozen provenance split.

    The frozen corpus stamps each observation with provenance["split"]
    determined by the taluk-to-split mapping.  This function replicates
    the partitioning logic of FrozenCorpusLoader.build() without calling
    the generic split_observations() recommuter.
    """
    train: list[AgriculturalObservation] = []
    val: list[AgriculturalObservation] = []
    test: list[AgriculturalObservation] = []
    for obs in observations:
        split = (obs.provenance or {}).get("split", "unknown")
        if split == "train":
            train.append(obs)
        elif split == "val":
            val.append(obs)
        elif split == "test":
            test.append(obs)
        else:
            train.append(obs)  # frozen fallback: unknown -> train
    return train, val, test


def _load_frozen_corpus(path: Path) -> list[AgriculturalObservation]:
    """Load accepted observations from the frozen corpus JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    obs_list: list[AgriculturalObservation] = []
    for sample in raw["samples"]:
        if sample.get("status") != "accepted":
            continue
        obs_data = sample.get("observation")
        if obs_data is None:
            continue
        obs = AgriculturalObservation.model_validate(obs_data)
        # Ensure provenance is populated (may come from sample-level or obs-level)
        if not obs.provenance:
            obs.provenance = dict(sample.get("provenance") or {})
        obs_list.append(obs)
    return obs_list


def _validate_frozen_contract(
    train: list[AgriculturalObservation],
    val: list[AgriculturalObservation],
    test: list[AgriculturalObservation],
    expected: dict[str, Any],
) -> None:
    """Assert frozen corpus split counts match the manifest contract."""
    total = len(train) + len(val) + len(test)
    assert total == expected["total"], (
        f"Frozen corpus total mismatch: expected {expected['total']}, got {total}"
    )
    assert len(train) == expected["train"], (
        f"Frozen train split mismatch: expected {expected['train']}, got {len(train)}"
    )
    assert len(val) == expected["val"], (
        f"Frozen val split mismatch: expected {expected['val']}, got {len(val)}"
    )
    assert len(test) == expected["test"], (
        f"Frozen test split mismatch: expected {expected['test']}, got {len(test)}"
    )
    # Verify no ID overlap
    train_ids = {str(o.observation_id) for o in train}
    val_ids = {str(o.observation_id) for o in val}
    test_ids = {str(o.observation_id) for o in test}
    assert not (train_ids & val_ids), "train/val ID overlap detected"
    assert not (train_ids & test_ids), "train/test ID overlap detected"
    assert not (val_ids & test_ids), "val/test ID overlap detected"


def _validate_class_vocabulary(observations: list[AgriculturalObservation]) -> Counter:
    """Verify supervised class vocabulary and return crop distribution."""
    crops = Counter(o.crop for o in observations)
    all_labels = set(crops.keys())
    supervised = set(_SUPERVISED_CLASSES)
    excluded = set(_EXCLUDED_CLASSES)
    unexpected = all_labels - supervised - excluded
    assert not unexpected, f"Unexpected crop labels: {unexpected}"
    return crops


def _per_class_metrics(
    all_preds: list[int], all_targets: list[int], class_names: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for idx, name in enumerate(class_names):
        tp = sum(1 for p, t in zip(all_preds, all_targets) if p == idx and t == idx)
        fp = sum(1 for p, t in zip(all_preds, all_targets) if p == idx and t != idx)
        fn = sum(1 for p, t in zip(all_preds, all_targets) if p != idx and t == idx)
        tp_sum = tp + fp
        fn_sum = tp + fn
        precision = tp / tp_sum if tp_sum > 0 else 0.0
        recall = tp / fn_sum if fn_sum > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for t in all_targets if t == idx)
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


def _compute_weighted_f1(per_class: list[dict[str, Any]]) -> float:
    total_support = sum(r["support"] for r in per_class)
    if total_support == 0:
        return 0.0
    return sum(r["f1"] * r["support"] for r in per_class) / total_support


def _evaluate_split(
    model: Any,
    loader: Any,
    class_names: list[str],
    device: str,
    split_name: str,
) -> dict[str, Any]:
    """Run inference on a single split and return structured metrics."""
    all_preds_np: list[np.ndarray] = []
    all_targets_np: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            inputs = {k: batch_dev[k] for k in ("tabular", "ndvi", "evi", "temporal_mask") if k in batch_dev}
            out = model(inputs)
            logits = out.crop_logits if hasattr(out, "crop_logits") else out.get("crop_logits")
            if logits is None:
                continue
            all_preds_np.append(logits.argmax(dim=-1).cpu().numpy())
            all_targets_np.append(batch_dev["crop_label"].cpu().numpy())

    if not all_preds_np:
        print(f"  WARNING: no predictions collected for {split_name}")
        return {}

    all_preds = np.concatenate(all_preds_np)
    all_targets = np.concatenate(all_targets_np)
    total = len(all_preds)
    correct = int((all_preds == all_targets).sum())
    accuracy = correct / total if total else 0.0

    per_class = _per_class_metrics(all_preds.tolist(), all_targets.tolist(), class_names)
    non_zero = [r for r in per_class if r["support"] > 0]
    macro_f1 = sum(r["f1"] for r in non_zero) / len(non_zero) if non_zero else 0.0
    macro_precision = sum(r["precision"] for r in non_zero) / len(non_zero) if non_zero else 0.0
    macro_recall = sum(r["recall"] for r in non_zero) / len(non_zero) if non_zero else 0.0
    weighted_f1 = _compute_weighted_f1(per_class)

    pred_dist = {name: int((all_preds == i).sum()) for i, name in enumerate(class_names)}
    target_dist = {name: int((all_targets == i).sum()) for i, name in enumerate(class_names)}

    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(all_preds, all_targets):
        cm[int(t), int(p)] += 1

    # Print summary
    print()
    print(f"  ---- {split_name.upper()}  n={total} ----")
    print(f"  accuracy:        {accuracy:.4f}")
    print(f"  macro F1:        {macro_f1:.4f}")
    print(f"  weighted F1:     {weighted_f1:.4f}")
    print(f"  macro precision: {macro_precision:.4f}")
    print(f"  macro recall:    {macro_recall:.4f}")

    print()
    print("  ---- PER-CLASS METRICS ----")
    hdr = f"  {'class':<12}{'support':>9}{'pred':>7}{'prec':>7}{'recall':>7}{'f1':>7}"
    print(hdr)
    for r in per_class:
        print(f"  {r['class']:<12}{r['support']:>9}{r['predicted']:>7}"
              f"{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}")

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

    return {
        "split": split_name,
        "n_samples": total,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "per_class": per_class,
        "prediction_distribution": pred_dist,
        "target_distribution": target_dist,
        "confusion_matrix": cm.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-diagnose-model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default=None,
                        help="Frozen manifest defining the corpus contract (default: repo v2.0 manifest)")
    parser.add_argument("--preprocessing-config", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--stam-config", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    corpus_path = Path(args.corpus)
    ckpt_path = Path(args.checkpoint)
    manifest_path = Path(args.manifest) if args.manifest else _FROZEN_MANIFEST
    output_dir = Path(args.output) if args.output else _REPO_ROOT / "training/kaggle/outputs/reports"
    pre_config = Path(args.preprocessing_config) if args.preprocessing_config else _REPO_ROOT / "training/config/preprocessing.yaml"
    ds_config = Path(args.dataset_config) if args.dataset_config else _REPO_ROOT / "training/config/dataset.yaml"
    stam_config = Path(args.stam_config) if args.stam_config else _REPO_ROOT / "training/config/stam.yaml"

    print("=" * 66)
    print("  POST-TRAINING DIAGNOSTIC (frozen-provenance split)")
    print("=" * 66)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    # ── Load frozen contract from the manifest (NOT hard-coded counts) ──
    print("  loading frozen contract from manifest...")
    contract = _load_frozen_contract(manifest_path)
    print(f"  contract: total={contract['total']} train={contract['train']} "
          f"val={contract['val']} test={contract['test']}")
    print(f"  manifest sha256: {contract['manifest_sha256']}")

    # ── Load frozen corpus ─────────────────────────────────────────────
    print("  loading frozen corpus...")
    all_obs = _load_frozen_corpus(corpus_path)
    print(f"  {len(all_obs)} observations loaded")

    # ── Split by frozen provenance (NOT split_observations) ────────────
    print("  using frozen provenance split...")
    train_obs, val_obs, test_obs = _split_by_frozen_provenance(all_obs)

    # ── Validate frozen contract ───────────────────────────────────────
    print("  validating frozen contract...")
    _validate_frozen_contract(train_obs, val_obs, test_obs, contract)
    crop_dist = _validate_class_vocabulary(all_obs)

    print(f"  Using frozen provenance split")
    print(f"  train={len(train_obs)}  val={len(val_obs)}  test={len(test_obs)}")
    print(f"  supervised classes: {_SUPERVISED_CLASSES}")
    print(f"  excluded classes: {_EXCLUDED_CLASSES}")
    print(f"  crop distribution: {dict(crop_dist)}")

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

    # ── Preprocessor fit (TRAIN ONLY) ──────────────────────────────────
    print("  fitting preprocessor on train split (TRAIN ONLY)...")
    preprocessing_cfg = load_preprocessing_config(pre_config)
    pre = Preprocessor(preprocessing_cfg)
    try:
        pre.fit(train_obs, extractor=extractor)
    except Exception as exc:
        print(f"  preprocessor fit failed: {exc}")
        return 1

    # ── Class names from label encoder (supervised 4-class) ────────────
    class_names = list(pre.label.crop_encoder.classes_)
    n_classes = len(class_names)
    print(f"  classes ({n_classes}): {class_names}")
    assert class_names == _SUPERVISED_CLASSES, (
        f"Label encoder classes {class_names} != frozen contract {_SUPERVISED_CLASSES}"
    )

    # ── Feature shape diagnostics ──────────────────────────────────────
    print("  feature shapes:")
    sample_ds = CropFusionDataset.build(pre, train_obs[:2], split="train", extractor=extractor)
    sample_batch = next(iter(build_dataloader(sample_ds, config=preprocessing_cfg, split="train", batch_size=2)))
    for key in ("tabular", "ndvi", "evi", "temporal_mask"):
        if key in sample_batch:
            t = sample_batch[key]
            print(f"    {key}: {list(t.shape)}")
    print(f"    tabular feature names: {pre.tabular.feature_names}")

    # ── Load model from checkpoint ─────────────────────────────────────
    print(f"  loading model from {ckpt_path}...")
    try:
        model = ModelFactory.from_checkpoint(str(ckpt_path))
    except Exception as exc:
        print(f"  failed to load checkpoint: {exc}")
        return 1
    model.to(device)
    model.eval()

    # ── Evaluate requested split ───────────────────────────────────────
    print(f"  building {args.split} DataLoader...")
    eval_obs = val_obs if args.split == "val" else test_obs
    assert len(eval_obs) > 0, (
        f"Frozen {args.split} split unexpectedly empty. "
        f"Expected {contract['val'] if args.split == 'val' else contract['test']} "
        f"observations from frozen provenance."
    )
    ds = CropFusionDataset.build(pre, eval_obs, split=args.split, extractor=extractor)
    loader = build_dataloader(ds, config=preprocessing_cfg, split=args.split, batch_size=args.batch_size)

    eval_result = _evaluate_split(model, loader, class_names, device, args.split)
    if not eval_result:
        return 1

    # ── Build report ───────────────────────────────────────────────────
    report: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "corpus": str(corpus_path),
        "frozen_contract": contract,
        "actual_split_sizes": {
            "train": len(train_obs),
            "val": len(val_obs),
            "test": len(test_obs),
        },
        "classes": class_names,
        "device": device,
        **eval_result,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "diagnostic_report.json"
    target.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote report -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
