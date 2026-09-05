"""Classifier-collapse diagnostic driver (R5.5) — Kaggle GPU phases.

Runs the R5.5 classifier-collapse investigation on a Kaggle kernel where the
imagery mount and a GPU are available. Builds the frozen corpus exactly like
:mod:`run_pipeline` (DatasetManager -> STAM -> FrozenCorpusLoader -> Preprocessor
fitted on train only), then executes the GPU-backed diagnostic phases:

    Phase 5/9   image separability + model-input normalization stats
    Phase 3     binary coconut-vs-pepper training (loss variants)
    Phase 10    first-N-step training dynamics (softmax collapse probe)
    Phase 11    tiny-set (20+20) overfit capacity probe
    Phase 12-14 sklearn baselines (delegated to :mod:`run_baselines`)

Every phase is optional via ``--skip-*`` and everything writes a single
``diagnostic_r5_5.json`` report. Tabular-only phases (1,2,4,6,7,8,16,17) run
locally via :mod:`diagnose_collapse`.

Run from repo root on Kaggle::

    python training/kaggle/scripts/diagnose_collapse_kaggle.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_repo_root(repo_root: Path) -> None:
    import sys

    root = str(repo_root.resolve())
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
from training.kaggle.frozen_corpus import FrozenCorpusLoader  # noqa: E402
from training.models import ModelFactory  # noqa: E402
from training.models.config import (  # noqa: E402
    ModelConfig,
    load_model_config as load_model_cfg,
)
from training.preprocessing import Preprocessor, load_preprocessing_config  # noqa: E402
from training.preprocessing.dataloader import build_dataloader  # noqa: E402
from training.preprocessing.dataset import CropFusionDataset  # noqa: E402
from training.stam import STAM  # noqa: E402
from training.stam.config import load_stam_config  # noqa: E402
from training.stam.observation_resolver import (  # noqa: E402
    ObservationCorpus,
    ResolvedSample,
)
from training.training.config import (  # noqa: E402
    load_training_config as load_training_cfg,
)
from training.training.checkpoint import TrainingCheckpointManager  # noqa: E402
from training.training.cropfusion_trainer import CropFusionTrainer  # noqa: E402
from training.training.interfaces import Callback  # noqa: E402
from training.training.losses import build_class_weights  # noqa: E402

BINARY_CLASSES = ["coconut", "pepper"]


def _crop_logits(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        return out["crop"] if "crop" in out else out["crop_logits"]
    return getattr(out, "crop_logits", out)


def _softmax_distribution(
    model: torch.nn.Module, inputs: dict[str, Any], device: torch.device
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        out = model({k: v.to(device) for k, v in inputs.items()})
    probs = torch.softmax(_crop_logits(out).float(), dim=-1)
    means = probs.mean(dim=0).tolist()
    model.train()
    return {"mean_softmax_per_class": [round(float(m), 5) for m in means]}


class DynamicsProbe(Callback):
    def __init__(
        self,
        probe_inputs: dict[str, Any],
        probe_targets: dict[str, Any],
        device: torch.device,
        limit: int = 10,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.probe_inputs = {k: v.detach().clone() for k, v in probe_inputs.items()}
        self.probe_targets = {
            k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in probe_targets.items()
        }
        self.device = device
        self.limit = limit
        self.num_classes = num_classes
        self.rows: list[dict[str, Any]] = []

    def on_batch_end(self, step: int, logs: dict[str, Any] | None = None) -> None:
        if step > self.limit:
            return
        logs = logs or {}
        if self.trainer is None:
            return
        model = self.trainer.raw_model
        with torch.no_grad():
            out = model({k: v.to(self.device) for k, v in self.probe_inputs.items()})
        logits = _crop_logits(out).float()
        means = torch.softmax(logits, dim=-1).mean(dim=0)
        gap = (
            float(logits.mean(dim=0)[0] - logits.mean(dim=0)[1])
            if self.num_classes >= 2
            else float("nan")
        )
        probs = torch.softmax(logits, dim=-1)
        row = {
            "step": step,
            "epoch": logs.get("epoch"),
            "train_loss": logs.get("train_loss"),
            "lr": logs.get("lr"),
            "mean_logits": [round(float(v), 4) for v in logits.mean(dim=0).tolist()],
            "mean_softmax": [round(float(v), 5) for v in means.tolist()],
            "coconut_minus_pepper_mean_logit": round(float(gap), 4),
        }
        targets = self.probe_targets.get("crop")
        if torch.is_tensor(targets):
            preds = probs.argmax(dim=-1)
            correct = preds.eq(targets.to(preds.device)).float().mean().item()
            row["probe_batch_accuracy"] = round(float(correct), 5)
        self.rows.append(row)


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
    }


def _per_class_frame_stats(
    loader: torch.utils.data.DataLoader,
    class_names: list[str],
    max_per_class: int = 200,
) -> dict[str, Any]:
    buckets: dict[int, dict[str, list[float]]] = {}
    for batch in loader:
        labels = batch["crop_label"].numpy()
        ndvi = batch["ndvi"]
        evi = batch["evi"]
        mask = batch["temporal_mask"]
        for i in range(ndvi.size(0)):
            cls = int(labels[i])
            if cls not in buckets:
                buckets[cls] = {
                    "ndvi_mean": [], "ndvi_std": [], "ndvi_min": [], "ndvi_max": [],
                    "evi_mean": [], "evi_std": [], "evi_min": [], "evi_max": [],
                    "zero_fill_frac": [],
                }
            if len(buckets[cls]["ndvi_mean"]) >= max_per_class:
                continue
            real = mask[i]
            n_real = int(real.sum().item())
            fraction = n_real / max(1, real.numel())
            for stream, key in ((ndvi[i], "ndvi"), (evi[i], "evi")):
                t = stream[real == 1]
                if t.numel() == 0:
                    continue
                vals = t.float()
                buckets[cls][f"{key}_mean"].append(vals.mean().item())
                buckets[cls][f"{key}_std"].append(vals.std().item())
                buckets[cls][f"{key}_min"].append(vals.min().item())
                buckets[cls][f"{key}_max"].append(vals.max().item())
            buckets[cls]["zero_fill_frac"].append(1.0 - fraction)
    out: dict[str, Any] = {}
    for cls in range(len(class_names)):
        stats = buckets.get(cls)
        if not stats:
            continue
        out[class_names[cls]] = {
            field: _mean_std(values) for field, values in stats.items()
        }
    return out


def _evaluate_binary(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    preds_all: list[int] = []
    targets_all: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v for k, v in batch.items()
                      if k in ("tabular", "ndvi", "evi", "temporal_mask")}
            labels = batch["crop_label"]
            out = model({k: v.to(device) for k, v in inputs.items()})
            preds = torch.softmax(_crop_logits(out).float(), dim=-1).argmax(dim=-1)
            preds_all.extend(preds.tolist())
            targets_all.extend(labels.tolist())
    model.train()
    y_pred = np.asarray(preds_all)
    y_true = np.asarray(targets_all)
    n = len(y_true)
    acc = float((y_pred == y_true).sum()) / max(1, n)
    support = {name: int((y_true == i).sum()) for i, name in enumerate(class_names)}
    pred_count = {name: int((y_pred == i).sum()) for i, name in enumerate(class_names)}
    prior = max(support.values()) / max(1, n)
    rows = []
    for i, name in enumerate(class_names):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            "class": name,
            "support": support[name],
            "predicted": pred_count[name],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    macro_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "majority_prior_accuracy": round(prior, 4),
        "beats_majority_prior": bool(acc > prior + 1e-6),
        "per_class": rows,
        "prediction_distribution": pred_count,
        "val_support": support,
    }


def _model_config_for(
    pre: Preprocessor, model_cfg: Any
) -> ModelConfig:
    base = model_cfg if isinstance(model_cfg, dict) else model_cfg.model_dump()
    tabular = dict(base.get("tabular") or {})
    tabular.pop("numeric_dim", None)
    tabular.pop("categorical_cardinalities", None)
    heads = dict(base.get("heads") or {})
    crop = dict(heads.get("crop") or {})
    crop.pop("num_classes", None)
    heads["crop"] = crop
    image_encoder = dict(base.get("image_encoder") or {})
    image_encoder.pop("input_size", None)
    temporal = dict(base.get("temporal") or {})
    temporal.pop("max_len", None)
    return ModelConfig.from_preprocessor(
        pre,
        tabular=tabular,
        heads=heads,
        image_encoder=image_encoder,
        temporal=temporal,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-diagnose-collapse-kaggle",
        description="R5.5 classifier-collapse GPU diagnostics",
    )
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument(
        "--paths-config",
        default=str(_REPO_ROOT / "training" / "config" / "paths.yaml"),
    )
    parser.add_argument(
        "--dataset-config",
        default=str(_REPO_ROOT / "training" / "config" / "dataset.yaml"),
    )
    parser.add_argument(
        "--training-config",
        default=str(_REPO_ROOT / "training" / "config" / "training.yaml"),
    )
    parser.add_argument(
        "--model-config",
        default=str(_REPO_ROOT / "training" / "config" / "model.yaml"),
    )
    parser.add_argument(
        "--stam-config",
        default=str(_REPO_ROOT / "training" / "config" / "stam.yaml"),
    )
    parser.add_argument(
        "--preprocessing-config",
        default=str(_REPO_ROOT / "training" / "config" / "preprocessing.yaml"),
    )
    parser.add_argument(
        "--validation-config",
        default=str(_REPO_ROOT / "training" / "config" / "validation.yaml"),
    )
    parser.add_argument(
        "--frozen-crop-csv",
        default=str(_REPO_ROOT / "govt_crop_matched_v2" / "crop_supervised_v2.csv"),
    )
    parser.add_argument(
        "--frozen-manifest",
        default=str(
            _REPO_ROOT / "training_manifests" / "crop_supervised_v2.0_manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "training" / "kaggle" / "outputs" / "reports"),
    )
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-image-stats", action="store_true")
    parser.add_argument("--skip-binary", action="store_true")
    parser.add_argument("--skip-dynamics", action="store_true")
    parser.add_argument("--skip-tiny", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--probe-steps", type=int, default=10)
    parser.add_argument("--tiny-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset-per-class", type=int, default=200)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    _add_repo_root(repo_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {"device": str(device)}

    def save_report(label: str) -> Path:
        target = output / "diagnostic_r5_5.json"
        target.write_text(
            json.dumps(report, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[checkpoint:{label}] report -> {target}")
        return target

    checkpoint_manager = TrainingCheckpointManager(
        output / "diagnostic_checkpoints", keep_last=2
    )

    dataset_settings = load_settings(args.dataset_config)
    stam_cfg = load_stam_config(args.stam_config)
    preprocessing_cfg = load_preprocessing_config(args.preprocessing_config)
    training_cfg = load_training_cfg(args.training_config)
    model_cfg = load_model_cfg(args.model_config)

    manager = DatasetManager(dataset_settings)
    try:
        manifests = manager.provider_manifests()
        imagery = manifests.get("kaggle_hub_image", {})
        if not imagery.get("available"):
            print("[FATAL] imagery not available; diagnostic requires the Kaggle mount")
            return 1
        manager.ensure_image()
        manager.generate_image_metadata(force=False)
        stam = STAM(manager, stam_cfg)
        stam.initialize()
        extractor = stam.get_patch

        frozen_loader = FrozenCorpusLoader(
            csv_path=args.frozen_crop_csv,
            manifest_path=args.frozen_manifest,
        )
        frozen_loader.validate()
        declared = list(frozen_loader.manifest.get("supervised_classes") or [])
        excluded = list(frozen_loader.manifest.get("excluded_classes") or [])
        preprocessing_cfg.label.declared_classes = declared
        preprocessing_cfg.label.excluded_classes = excluded

        train_obs, val_obs, test_obs = frozen_loader.build(stam)
        accepted = train_obs + val_obs + test_obs
        report["corpus"] = {
            "total": len(accepted),
            "train": len(train_obs),
            "val": len(val_obs),
            "test": len(test_obs),
            "declared_classes": declared,
            "excluded_classes": excluded,
            "class_counts": {
                "train": {cls: sum(1 for o in train_obs if o.crop == cls)
                          for cls in declared},
                "val": {cls: sum(1 for o in val_obs if o.crop == cls)
                        for cls in declared},
                "test": {cls: sum(1 for o in test_obs if o.crop == cls)
                         for cls in declared},
            },
        }
        corpus_path = output / "frozen_corpus.json"
        ObservationCorpus(
            samples=[
                ResolvedSample(
                    location_id=o.provenance.get("record_id")
                    or str(o.observation_id),
                    name=o.crop or "unknown",
                    lon=o.location.lon,
                    lat=o.location.lat,
                    year=o.temporal.year or 0,
                    season=o.temporal.season or "unknown",
                    status="accepted",
                    quality_score=o.quality.overall_score,
                    observation=o,
                )
                for o in accepted
            ],
            config={"source": "frozen_crop_supervised_v2"},
        ).save(corpus_path)
        print(f"[build] saved corpus -> {corpus_path}")

        pre = Preprocessor(preprocessing_cfg)
        pre.fit(train_obs, extractor=extractor)
        print(f"[build] preprocessor fitted on train ({len(train_obs)} obs)")

        train_loader = build_dataloader(
            CropFusionDataset.build(pre, train_obs, split="train", extractor=extractor),
            config=preprocessing_cfg,
            split="train",
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = build_dataloader(
            CropFusionDataset.build(pre, val_obs, split="val", extractor=extractor),
            config=preprocessing_cfg,
            split="val",
            batch_size=args.batch_size,
            shuffle=False,
        )

        class_names = list(pre.label.crop_encoder.classes_)
        n_classes = len(class_names)
        report["class_schema"] = {"classes": class_names, "value": declared}

        train_counts = torch.zeros(n_classes)
        for o in train_obs:
            idx = class_names.index(o.crop)
            train_counts[idx] += 1
        weights = build_class_weights(training_cfg.loss, n_classes, train_counts)
        report["class_weights"] = {
            "mode": training_cfg.loss.class_weight_mode,
            "per_class": {
                cls: float(weights[i]) if weights is not None else None
                for i, cls in enumerate(class_names)
            },
        }

        if not args.skip_image_stats:
            print("\n=== Phase 5/9: image separability + normalization ===")
            stats_loader = build_dataloader(
                CropFusionDataset.build(
                    pre, train_obs, split="train", extractor=extractor
                ),
                config=preprocessing_cfg,
                split="train",
                batch_size=args.batch_size,
                shuffle=False,
            )
            img_stats = _per_class_frame_stats(
                stats_loader, class_names, max_per_class=args.subset_per_class
            )
            report["image_normalization"] = {
                "scope": "train subset (real frames from fitted preprocessor)",
                "per_class": img_stats,
            }
            print(json.dumps(img_stats, indent=2, default=str))
            save_report("phase_5_9_image_stats")

        if not args.skip_binary:
            print("\n=== Phase 3: binary coconut vs pepper ===")
            bin_train = [o for o in train_obs if o.crop in BINARY_CLASSES]
            bin_val = [o for o in val_obs if o.crop in BINARY_CLASSES]
            bin_counts = {c: sum(1 for o in bin_train if o.crop == c)
                          for c in BINARY_CLASSES}
            val_counts = {c: sum(1 for o in bin_val if o.crop == c)
                          for c in BINARY_CLASSES}
            majority = max(val_counts.values())
            prior_acc = majority / max(1, len(bin_val))
            report["binary"] = {
                "train_counts": bin_counts,
                "val_counts": val_counts,
                "binary_val_majority_acc": round(float(prior_acc), 4),
            }
            print(f"  binary train={len(bin_train)} val={len(bin_val)} "
                  f"prior_acc={prior_acc:.4f}")
            bin_cfg = Preprocessor.from_config(args.preprocessing_config)
            bin_cfg.config.label.declared_classes = BINARY_CLASSES
            bin_cfg.config.label.excluded_classes = []
            pre_bin = bin_cfg
            pre_bin.fit(bin_train, extractor=extractor)
            bin_class_names = list(pre_bin.label.crop_encoder.classes_)
            bin_train_loader = build_dataloader(
                CropFusionDataset.build(
                    pre_bin, bin_train, split="train", extractor=extractor
                ),
                config=pre_bin.config,
                split="train",
                batch_size=args.batch_size,
                shuffle=True,
            )
            bin_val_loader = build_dataloader(
                CropFusionDataset.build(
                    pre_bin, bin_val, split="val", extractor=extractor
                ),
                config=pre_bin.config,
                split="val",
                batch_size=args.batch_size,
                shuffle=False,
            )
            variants = [
                ("default_focal_sqrt_inv", {}),
                ("ce_balanced", {"crop_loss": "cross_entropy",
                                 "class_weight_mode": "balanced"}),
                ("ce_none", {"crop_loss": "cross_entropy",
                             "class_weight_mode": "none"}),
            ]
            variants_out: list[dict[str, Any]] = []
            for variant_name, loss_updates in variants:
                print(f"\n  -- binary variant: {variant_name} --")
                loss_cfg = training_cfg.loss.model_copy(update=loss_updates)
                run_cfg = training_cfg.model_copy(update={
                    "loss": loss_cfg,
                    "train": training_cfg.train.model_copy(update={"epochs": args.epochs}),
                    "general": training_cfg.general.model_copy(update={"log_every": 1}),
                })
                model_config = _model_config_for(pre_bin, model_cfg)
                model = ModelFactory.create(model_config)
                trainer = CropFusionTrainer(
                    model,
                    bin_train_loader,
                    run_cfg,
                    val_loader=bin_val_loader,
                    checkpoint_manager=checkpoint_manager,
                    device=device,
                )
                result = trainer.train()
                res_dict = result.summary()
                eval_block = _evaluate_binary(
                    model, bin_val_loader, device, bin_class_names
                )
                variants_out.append({
                    "variant": variant_name,
                    "loss": {"crop_loss": loss_cfg.crop_loss,
                             "class_weight_mode": loss_cfg.class_weight_mode},
                    "metrics": eval_block,
                    "training_summary": res_dict,
                })
            report["binary"]["variants"] = variants_out
            print(json.dumps(report["binary"], indent=2, default=str))
            save_report("phase_3_binary")

        if not args.skip_dynamics:
            print("\n=== Phase 10: first-N-step training dynamics ===")
            probe_batch = next(iter(val_loader))
            probe_inputs = {k: v for k, v in probe_batch.items()
                            if k in ("tabular", "ndvi", "evi", "temporal_mask")}
            probe_targets = {"crop": probe_batch["crop_label"]}
            probe = DynamicsProbe(
                probe_inputs, probe_targets, device,
                limit=args.probe_steps, num_classes=n_classes,
            )
            dynamics_cfg = training_cfg.model_copy(update={
                "train": training_cfg.train.model_copy(update={"epochs": 1}),
                "general": training_cfg.general.model_copy(update={"log_every": 1}),
            })
            model_config = _model_config_for(pre, model_cfg)
            model = ModelFactory.create(model_config)
            trainer = CropFusionTrainer(
                model, train_loader, dynamics_cfg,
                val_loader=val_loader, callbacks=[probe],
                checkpoint_manager=checkpoint_manager, device=device,
            )
            trainer.train()
            report["dynamics"] = {
                "probe_steps": args.probe_steps,
                "steps": probe.rows,
                "note": "mean softmax over the fixed val probe batch",
            }
            for row in probe.rows:
                print(json.dumps(row, default=str))
            save_report("phase_10_dynamics")

        if not args.skip_tiny:
            print("\n=== Phase 11: tiny-set overfit (20 + 20) ===")
            rng = np.random.RandomState(2026)
            tiny = []
            for cls in BINARY_CLASSES:
                rows = [o for o in train_obs if o.crop == cls]
                idx = rng.choice(len(rows), size=min(20, len(rows)), replace=False)
                tiny.extend(rows[int(i)] for i in idx)
            tiny_cfg = Preprocessor.from_config(args.preprocessing_config)
            tiny_cfg.config.label.declared_classes = BINARY_CLASSES
            tiny_cfg.config.label.excluded_classes = []
            pre_tiny = tiny_cfg
            pre_tiny.fit(tiny, extractor=extractor)
            tiny_ds = CropFusionDataset.build(
                pre_tiny, tiny, split="train", extractor=extractor
            )
            tiny_loader = torch.utils.data.DataLoader(
                tiny_ds, batch_size=len(tiny), shuffle=True,
                num_workers=0, drop_last=False,
            )
            tiny_probe_batch = next(iter(tiny_loader))
            model_config = _model_config_for(pre_tiny, model_cfg)
            model = ModelFactory.create(model_config)
            probe = DynamicsProbe(
                {k: v for k, v in tiny_probe_batch.items()
                 if k in ("tabular", "ndvi", "evi", "temporal_mask")},
                {"crop": tiny_probe_batch["crop_label"]},
                device,
                limit=args.tiny_steps,
                num_classes=2,
            )
            tiny_config_single = training_cfg.model_copy(update={
                "train": training_cfg.train.model_copy(
                    update={"epochs": args.tiny_steps}
                ),
                "general": training_cfg.general.model_copy(update={"log_every": 1}),
            })
            trainer = CropFusionTrainer(
                model, tiny_loader, tiny_config_single,
                callbacks=[probe], checkpoint_manager=checkpoint_manager,
                device=device,
            )
            trainer.train()
            acc_at = [r for r in probe.rows if r.get("probe_batch_accuracy", -1) >= 0.9]
            report["tiny_overfit"] = {
                "n_train": len(tiny),
                "classes": {c: 20 for c in BINARY_CLASSES},
                "steps_to_acc_090": acc_at[0]["step"] if acc_at
                else None,
                "best_probe_accuracy": max(
                    (r.get("probe_batch_accuracy", -1) for r in probe.rows), default=None
                ),
                "steps_traced": args.tiny_steps,
            }
            print(json.dumps(report["tiny_overfit"], indent=2, default=str))
            save_report("phase_11_tiny")

        if not args.skip_baselines:
            print("\n=== Phases 12-14: sklearn baselines ===")
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            import run_baselines
            baseline_rc = run_baselines.main([
                "--corpus", str(corpus_path),
                "--output", str(output / "baselines"),
                "--manifest", args.frozen_manifest,
                "--preprocessing-config", args.preprocessing_config,
                "--dataset-config", args.dataset_config,
                "--stam-config", args.stam_config,
                "--batch-size", str(args.batch_size),
            ])
            report["baselines_return_code"] = baseline_rc

        save_report("final")
        print("\n[diagnose] complete")
    finally:
        manager.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())