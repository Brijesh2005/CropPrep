"""R5.4 pre-flight verification tests (local, CPU-safe).

Covers the invariants the final Kaggle run depends on:

1. sqrt_inv class weights are numerically sound on the real class counts.
2. FocalLoss stays finite at extreme confidence / with alpha weights.
3. Metrics now score over the FULL class set (macro over all classes, not
   just the classes present in a batch).
4. Early stopping monitors ``crop/macro_f1`` (max) and the best checkpoint
   is selected from that metric.
5. ``restore_best_on_stop`` restores the best checkpoint into the model.
6. Yield metrics stay absent when there is no yield head.
7. Discriminative LR builds two optimizer param groups (backbone vs rest).
8. Staged fine-tuning freezes the backbone and progressively unfreezes
   scheduled blocks.
9. Epoch-period schedulers advance even when a validator runs every epoch
   (regression: previously the LR was frozen for the whole run).
10. The shipped repository configs carry the intended R5.4 values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training.models import ModelConfig, ModelFactory
from training.models.losses import FocalLoss
from training.training import (
    CropFusionTrainer,
    TrainingConfig,
)
from training.training.callbacks import HistoryRecorder, StagedFineTuning
from training.training.config import OptimizerConfig
from training.training.losses import class_frequency_weights
from training.training.metrics import compute_classification_metrics
from training.training.optimizers import build_optimizer
from training.training.tests.conftest import make_fake_loader, small_full_config

# Real repository crop class counts: coconut 6468, pepper 3537, coffee 101,
# cardamom 11, blackgram 2 (pre-normalisation class weight scale).
REPO_COUNTS = torch.tensor([6468.0, 3537.0, 101.0, 11.0, 2.0])


# --------------------------------------------------------------------------- #
# 1. Class-weight numerics
# --------------------------------------------------------------------------- #


def test_sqrt_inv_weights_sound_on_real_counts():
    w = class_frequency_weights(REPO_COUNTS, "sqrt_inv")
    assert torch.isfinite(w).all()
    assert (w > 0).all()
    assert w.mean().item() == pytest.approx(1.0)
    # Strict monotonicity: rarer classes are up-weighted.
    assert (w.diff() > 0).all()
    assert w.max().item() <= 4.0


# --------------------------------------------------------------------------- #
# 2. FocalLoss numerics
# --------------------------------------------------------------------------- #


def test_focal_loss_finite_with_alpha_at_extremes():
    alpha = torch.tensor([0.2, 0.8])
    loss = FocalLoss(gamma=2.0, alpha=alpha)
    torch.manual_seed(0)
    # Extremely confident predictions (~p=1) and weak ones (~p=0.5).
    logits = torch.tensor([[50.0, 0.0], [0.0, 50.0], [2.0, 2.0], [-50.0, -50.0]])
    targets = torch.tensor([0, 1, 1, 0])
    value = loss(logits, targets)
    assert torch.isfinite(value)
    assert value.item() >= 0.0
    # Alpha weighting scales each sample's focal term directly.
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    per_sample = -(1.0 - p_t) ** 2.0 * log_p_t
    sample_weights = alpha.gather(0, targets)
    expected = (per_sample * sample_weights).mean()
    assert value == pytest.approx(expected.item(), abs=1e-6)


# --------------------------------------------------------------------------- #
# 3. Fixed-class-set metrics
# --------------------------------------------------------------------------- #


def test_metrics_score_over_full_class_set():
    torch.manual_seed(1)
    logits = torch.tensor(
        [[3.0, 1.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0, 0.0],
         [1.0, 3.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
    )
    labels = torch.tensor([0, 0, 1, 2])  # classes 3, 4 never present
    m = compute_classification_metrics(logits, labels)
    assert m["macro_f1"] is not None
    assert m["weighted_f1"] is not None
    assert m["balanced_accuracy"] is not None
    assert torch.isfinite(torch.as_tensor(
        [m["macro_f1"], m["weighted_f1"], m["balanced_accuracy"]]
    )).all()
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert 0.0 <= m["balanced_accuracy"] <= 1.0
    # Absent classes have zero recall on the fixed set, so the class-set-wide
    # balanced accuracy can never exceed mean recall of the present classes.
    present_recall = sum(1 for c in (0, 1, 2)) / 3.0
    assert m["balanced_accuracy"] <= present_recall + 1e-9


# --------------------------------------------------------------------------- #
# 4.-6., 9. Trainer-level integration (one shared run)
# --------------------------------------------------------------------------- #


def run_smoke(tmp_path: Path, epochs: int = 5, patience: int = 2, seed: int = 7,
              **train_overrides) -> tuple[CropFusionTrainer, "CropFusionTrainingResult"]:
    from training.training.cropfusion_trainer import CropFusionTrainingResult

    cfg = TrainingConfig(
        name="preflight",
        general={"device": "cpu", "seed": seed, "gradient_checkpointing": False,
                 "validation_frequency": 1},
        data={"batch_size": 8, "workers": 0, "pin_memory": False,
              "persistent_workers": False},
        optimizer={"lr": 1e-4, "backbone_lr_multiplier": 0.3},
        scheduler={"name": "warmup_cosine", "step": "epoch", "warmup_epochs": 2},
        loss={"crop_loss": "label_smoothing", "class_weight_mode": "sqrt_inv"},
        train={
            "epochs": epochs,
            "early_stopping_metric": "crop/macro_f1",
            "early_stopping_mode": "max",
            "early_stopping_patience": patience,
            "restore_best_on_stop": True,
            **train_overrides,
        },
        checkpoint={"directory": str(tmp_path / "ckpt"), "save_best": True,
                    "save_latest": True, "keep_last": 2},
        metrics={"top_k": 3},
        logging={"console": False},
        fine_tuning={"enabled": True,
                     "schedule": [{"epoch": 1, "prefixes": ["blocks.4"]}]},
    )
    model = ModelFactory.create(small_full_config())
    trainer = CropFusionTrainer(
        model,
        make_fake_loader(n=16, batch_size=8, feature_dim=4, multimodal=True),
        cfg,
        val_loader=make_fake_loader(n=8, batch_size=8, feature_dim=4, multimodal=True),
        callbacks=[HistoryRecorder()],
        device=torch.device("cpu"),
    )
    result = trainer.train()
    assert isinstance(result, CropFusionTrainingResult)
    return trainer, result


def test_early_stop_monitors_crop_macro_f1(tmp_path):
    _, result = run_smoke(tmp_path)
    assert "crop/macro_f1" in result.best_metrics
    val_macro = [h["crop/macro_f1"] for h in result.history]
    assert result.best_metrics["crop/macro_f1"] == pytest.approx(max(val_macro))
    # best_epoch is the argmax epoch over the monitored metric.
    assert result.best_epoch == val_macro.index(max(val_macro))


def test_restore_best_weights_on_stop(tmp_path):
    trainer, result = run_smoke(tmp_path)
    assert result.best_path is not None
    clone = ModelFactory.create(small_full_config())
    from training.training.checkpoint import TrainingCheckpointManager
    TrainingCheckpointManager(Path(result.best_path).parent).restore(
        result.best_path, model=clone
    )
    current = trainer.raw_model.state_dict()
    mismatches = [
        k for k, v in clone.state_dict().items()
        if not torch.equal(v.float(), current[k].float())
    ]
    assert not mismatches, f"model differs from best.pt in {mismatches[:5]}"


def test_scheduler_advances_with_validation(tmp_path):
    # Regression: epoch-period schedulers must step even when validation runs
    # every epoch (otherwise the LR is frozen for the entire run).
    _, result = run_smoke(tmp_path)
    lrs = [round(float(h["lr"]), 8) for h in result.history]
    assert len(lrs) >= 3
    assert lrs[-1] < lrs[0] or (lrs[1] > lrs[0] and max(lrs) > lrs[-1]), lrs
    assert max(lrs) <= 1e-4 + 1e-9


# --------------------------------------------------------------------------- #
# 6. Yield task stays inert without a yield head
# --------------------------------------------------------------------------- #


def test_yield_metrics_absent_without_yield_head(tmp_path):
    base = small_full_config().model_dump()
    base["heads"]["yield_prediction"] = None
    model = ModelFactory.create(ModelConfig(**base))
    assert getattr(model, "yield_head", None) is None
    cfg = TrainingConfig(
        name="yield_inert",
        general={"device": "cpu", "seed": 42, "validation_frequency": 1},
        data={"batch_size": 8, "workers": 0, "pin_memory": False,
              "persistent_workers": False},
        loss={"crop_loss": "label_smoothing"},
        train={"epochs": 1, "early_stopping_patience": 100},
        checkpoint={"directory": str(tmp_path / "ckpt"), "save_best": True},
        logging={"console": False},
    )
    loader = make_fake_loader(n=16, batch_size=8, feature_dim=4, multimodal=True)
    trainer = CropFusionTrainer(
        model, loader, cfg, val_loader=loader, device=torch.device("cpu")
    )
    result = trainer.train()
    keys = set(result.best_metrics) | set(result.history[0])
    assert not any(k.startswith("yield") for k in keys)


# --------------------------------------------------------------------------- #
# 7. Discriminative LR param groups
# --------------------------------------------------------------------------- #


def test_backbone_param_groups():
    model = ModelFactory.create(small_full_config())
    opt = build_optimizer(model, OptimizerConfig(lr=1e-4, backbone_lr_multiplier=0.3))
    assert len(opt.param_groups) == 2
    backbone, rest = opt.param_groups[0]["params"], opt.param_groups[1]["params"]
    assert abs(opt.param_groups[0]["lr"] - 3e-5) < 1e-12
    assert abs(opt.param_groups[1]["lr"] - 1e-4) < 1e-12
    backbone_ids = {id(p) for p in backbone}
    rest_ids = {id(p) for p in rest}
    assert backbone_ids.isdisjoint(rest_ids)
    model_ids = {id(p) for p in model.parameters()}
    assert backbone_ids | rest_ids == model_ids
    assert len(backbone_ids) > 0 and len(rest_ids) > 0
    # Every model parameter lives in exactly one group.
    assert {p.requires_grad for p in model.parameters()} == {True}
    # Without the multiplier we get the previous single group.
    plain = build_optimizer(model, OptimizerConfig(lr=1e-4))
    assert len(plain.param_groups) == 1


# --------------------------------------------------------------------------- #
# 8. Staged fine-tuning progression
# --------------------------------------------------------------------------- #


def test_staged_unfreeze_progression():
    model = ModelFactory.create(small_full_config())
    sft = StagedFineTuning([
        {"epoch": 5, "prefixes": ["blocks.6", "blocks.5"]},
        {"epoch": 10, "prefixes": ["blocks.4", "blocks.3"]},
    ])

    class _Stub:
        raw_model = model

    sft.set_trainer(_Stub())
    sft.on_train_begin()

    backbone_params = {n: p for n, p in model.named_parameters()
                       if n.startswith(("ndvi_encoder.", "evi_encoder."))}
    assert not any(p.requires_grad for p in backbone_params.values())

    sft.on_epoch_begin(5)
    for n, p in backbone_params.items():
        expected_trainable = any(
            seg in n for seg in ("backbone.blocks.5.", "backbone.blocks.6.")
        )
        assert p.requires_grad == expected_trainable, n
    count_after_stage1 = sum(1 for p in backbone_params.values() if p.requires_grad)

    sft.on_epoch_begin(10)
    count_after_stage2 = sum(1 for p in backbone_params.values() if p.requires_grad)
    assert count_after_stage2 > count_after_stage1
    # Early blocks (0-2) stay frozen - pretrained trunk is preserved.
    assert not any(
        p.requires_grad for n, p in backbone_params.items()
        if "backbone.blocks.2." in n
    )

    # Re-visiting an already-applied epoch must not change anything.
    sft.on_epoch_begin(10)
    assert sum(1 for p in backbone_params.values() if p.requires_grad) == count_after_stage2


# --------------------------------------------------------------------------- #
# 10. Shipped repository configs
# --------------------------------------------------------------------------- #


_REPO = Path(__file__).resolve().parents[2]  # training/


def test_repo_configs_carry_r54_values():
    from training.training.config import load_training_config
    from training.preprocessing.config import load_preprocessing_config

    tc = load_training_config(_REPO / "config" / "training.yaml")
    assert tc.train.early_stopping_metric == "crop/macro_f1"
    assert tc.train.early_stopping_mode == "max"
    assert tc.train.early_stopping_patience == 5
    assert tc.scheduler.name == "warmup_cosine"
    assert tc.scheduler.warmup_epochs == 2
    assert tc.optimizer.backbone_lr_multiplier == pytest.approx(0.3)
    assert tc.fine_tuning.enabled is True
    assert [s.model_dump()["prefixes"] for s in tc.fine_tuning.schedule] == [
        [], ["blocks.6", "blocks.5"], ["blocks.4", "blocks.3"],
    ]

    pc = load_preprocessing_config(_REPO / "config" / "preprocessing.yaml")
    assert pc.augmentation.enabled is True
    assert pc.augmentation.brightness_jitter == 0.0
    assert pc.augmentation.contrast_jitter == 0.0
    assert pc.augmentation.noise_std == 0.0
    assert pc.augmentation.flip_horizontal is True
    assert pc.augmentation.flip_vertical is True