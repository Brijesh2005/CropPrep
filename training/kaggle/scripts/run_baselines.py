"""Baseline experiments: tabular-only, imagery-only, and combined classifiers.

Trains lightweight sklearn classifiers on extracted features to establish
upper/lower bounds for the CropFusion model performance.

Uses the FROZEN spatial split (taluk-based) — never calls split_observations().

Requires imagery (Kaggle mount). Exits 0 with a clear message if unavailable.

Run from repo root (Kaggle kernel, after ``run_pipeline.py``)::

    python training/kaggle/scripts/run_baselines.py \\
        --corpus training/kaggle/outputs/reports/frozen_corpus.json \\
        --output training/kaggle/outputs/reports
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
from training.preprocessing import Preprocessor, load_preprocessing_config  # noqa: E402
from training.preprocessing.dataset import CropFusionDataset  # noqa: E402
from training.preprocessing.dataloader import build_dataloader  # noqa: E402
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


def _extract_tabular_features(
    loader: torch.utils.data.DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract tabular features and labels from a DataLoader."""
    all_tab = []
    all_labels = []
    for batch in loader:
        all_tab.append(batch["tabular"].numpy())
        all_labels.append(batch["crop_label"].numpy())
    return np.concatenate(all_tab), np.concatenate(all_labels)


def _extract_image_stats(
    loader: torch.utils.data.DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-sample mean NDVI/EVI from real frames only."""
    all_features = []
    all_labels = []
    for batch in loader:
        ndvi = batch["ndvi"]  # [B, T, 1, H, W]
        evi = batch["evi"]
        mask = batch["temporal_mask"]  # [B, T]
        B = ndvi.size(0)

        # Get real-frame mask: [B, T] -> [B, T, 1, 1, 1] for broadcasting
        real_mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Mean over real frames only (skip zero-padded frames)
        ndvi_real = (ndvi * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1)
        evi_real = (evi * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1)

        # Flatten and compute statistics
        ndvi_flat = ndvi_real.view(B, -1)
        evi_flat = evi_real.view(B, -1)

        # Per-pixel statistics across spatial dims
        features = torch.cat([
            ndvi_flat.mean(dim=-1, keepdim=True),
            ndvi_flat.std(dim=-1, keepdim=True),
            ndvi_flat.min(dim=-1, keepdim=True).values,
            ndvi_flat.max(dim=-1, keepdim=True).values,
            evi_flat.mean(dim=-1, keepdim=True),
            evi_flat.std(dim=-1, keepdim=True),
            evi_flat.min(dim=-1, keepdim=True).values,
            evi_flat.max(dim=-1, keepdim=True).values,
            # Cross-band correlation
            (ndvi_flat * evi_flat).mean(dim=-1, keepdim=True),
        ], dim=-1)  # [B, 9]

        all_features.append(features.numpy())
        all_labels.append(batch["crop_label"].numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def _per_class_metrics(
    all_preds: np.ndarray, all_targets: np.ndarray, class_names: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for idx, name in enumerate(class_names):
        tp = int(((all_preds == idx) & (all_targets == idx)).sum())
        fp = int(((all_preds == idx) & (all_targets != idx)).sum())
        fn = int(((all_preds != idx) & (all_targets == idx)).sum())
        support = int((all_targets == idx).sum())
        predicted = int((all_preds == idx).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            "class": name,
            "index": idx,
            "support": support,
            "predicted": predicted,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    return rows


def _evaluate_classifier(
    clf: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    name: str,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    per_class = _per_class_metrics(y_pred, y_test, class_names)
    non_zero = [r for r in per_class if r["support"] > 0]
    macro_f1 = sum(r["f1"] for r in non_zero) / len(non_zero) if non_zero else 0.0

    # Prediction distribution
    pred_dist = {name_: int((y_pred == i).sum()) for i, name_ in enumerate(class_names)}
    target_dist = {name_: int((y_test == i).sum()) for i, name_ in enumerate(class_names)}

    # Confusion matrix
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(y_pred, y_test):
        cm[int(t), int(p)] += 1

    return {
        "name": name,
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "prediction_distribution": pred_dist,
        "target_distribution": target_dist,
        "confusion_matrix": cm.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-run-baselines")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default=None,
                        help="Frozen manifest defining the corpus contract (default: repo v2.0 manifest)")
    parser.add_argument("--preprocessing-config", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--stam-config", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    corpus_path = Path(args.corpus)
    manifest_path = Path(args.manifest) if args.manifest else _FROZEN_MANIFEST
    output_dir = Path(args.output) if args.output else _REPO_ROOT / "training/kaggle/outputs/reports"
    pre_config = Path(args.preprocessing_config) if args.preprocessing_config else _REPO_ROOT / "training/config/preprocessing.yaml"
    ds_config = Path(args.dataset_config) if args.dataset_config else _REPO_ROOT / "training/config/dataset.yaml"
    stam_config = Path(args.stam_config) if args.stam_config else _REPO_ROOT / "training/config/stam.yaml"

    print("=" * 66)
    print("  BASELINE EXPERIMENTS (frozen-provenance split)")
    print("=" * 66)

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

    class_names = list(pre.label.crop_encoder.classes_)
    n_classes = len(class_names)
    print(f"  classes ({n_classes}): {class_names}")
    assert class_names == _SUPERVISED_CLASSES, (
        f"Label encoder classes {class_names} != frozen contract {_SUPERVISED_CLASSES}"
    )

    # ── Build DataLoaders ──────────────────────────────────────────────
    print("  building DataLoaders...")
    train_ds = CropFusionDataset.build(pre, train_obs, split="train", extractor=extractor)
    val_ds = CropFusionDataset.build(pre, val_obs, split="val", extractor=extractor)
    train_loader = build_dataloader(train_ds, config=preprocessing_cfg, split="train", batch_size=args.batch_size)
    val_loader = build_dataloader(val_ds, config=preprocessing_cfg, split="val", batch_size=args.batch_size)

    # ── Extract features ───────────────────────────────────────────────
    print("  extracting tabular features...")
    X_train_tab, y_train = _extract_tabular_features(train_loader)
    X_val_tab, y_val = _extract_tabular_features(val_loader)
    print(f"  tabular: train={X_train_tab.shape}  val={X_val_tab.shape}")

    print("  extracting image statistics...")
    X_train_img, _ = _extract_image_stats(train_loader)
    X_val_img, _ = _extract_image_stats(val_loader)
    print(f"  imagery: train={X_train_img.shape}  val={X_val_img.shape}")

    X_train_combined = np.hstack([X_train_tab, X_train_img])
    X_val_combined = np.hstack([X_val_tab, X_val_img])
    print(f"  combined: train={X_train_combined.shape}  val={X_val_combined.shape}")

    # ── Train and evaluate classifiers ─────────────────────────────────
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight

    results: list[dict[str, Any]] = []

    class_weights = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=y_train)
    sample_weights = np.array([class_weights[int(t)] for t in y_train])

    configs = [
        ("tabular_only_rf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ("tabular_only_gb", GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=4)),
        ("imagery_only_rf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ("imagery_only_gb", GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=4)),
        ("combined_rf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ("combined_gb", GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=4)),
    ]

    for name, clf in configs:
        print(f"\n  ---- {name} ----")
        if "tabular" in name:
            Xtr, Xvl = X_train_tab, X_val_tab
        elif "imagery" in name:
            Xtr, Xvl = X_train_img, X_val_img
        else:
            Xtr, Xvl = X_train_combined, X_val_combined

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xvl_s = scaler.transform(Xvl)

        if hasattr(clf, "fit") and "sample_weight" in clf.fit.__code__.co_varnames:
            clf.fit(Xtr_s, y_train, sample_weight=sample_weights)
        else:
            clf.fit(Xtr_s, y_train)

        result = _evaluate_classifier(clf, Xvl_s, y_val, class_names, name)
        results.append(result)
        print(f"    accuracy={result['accuracy']:.4f}  macro_f1={result['macro_f1']:.4f}")

    # ── Summary table ──────────────────────────────────────────────────
    print("\n  ---- BASELINE COMPARISON ----")
    print(f"  {'model':<25}{'acc':>7}{'bal_acc':>9}{'macro_f1':>9}")
    print("  " + "-" * 52)
    for r in results:
        print(f"  {r['name']:<25}{r['accuracy']:>7.4f}{r['balanced_accuracy']:>9.4f}{r['macro_f1']:>9.4f}")

    # ── Emit JSON ──────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "corpus": str(corpus_path),
        "frozen_contract": contract,
        "actual_split_sizes": {
            "train": len(train_obs),
            "val": len(val_obs),
            "test": len(test_obs),
        },
        "class_names": class_names,
        "results": results,
    }
    target = output_dir / "baselines_report.json"
    target.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote report -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
