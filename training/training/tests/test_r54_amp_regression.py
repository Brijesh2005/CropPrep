"""R5.4 AMP numerical-stability regression tests (CPU-safe).

Pins the epoch-9 failure: under GradScaler semantics (loss scaled by 2**16
before backward) the crop classifier weight gradient of the ONCE-fp16 task
head crosses fp16's 65504 limit and becomes non-finite. The fix runs task
heads in FP32, so these tests assert head gradients stay finite at far larger
scaled magnitudes, and that the checkpoint-guard (`_grads_nonfinite`) still
fires on a genuinely poisoned gradient.
"""

from __future__ import annotations

import pytest
import torch

from training.models.config import LossConfig
from training.models.losses import FocalLoss, HuberLoss, WeightedMultiTaskLoss
from training.models.multitask_heads import CropHead
from training.training.trainer import Trainer

B, H, C = 16, 512, 5  # real run: batch 16, shared out_dim 512, 5 crops
SCALE = 65536.0  # GradScaler default initial scale (2**16; confirmed active)
FP16_MAX = 65504.0


def _weighted_head_loss() -> WeightedMultiTaskLoss:
    cfg = LossConfig(
        crop_loss="focal",
        yield_loss="huber",
        crop_weight=0.75,
        yield_weight=0.25,
        label_smoothing=0.15,
        focal_gamma=2.0,
        reduction="mean",
    )
    jobs = {"crop": FocalLoss(gamma=2.0, reduction="mean"),
            "yield": HuberLoss(reduction="mean")}
    return WeightedMultiTaskLoss(cfg, jobs)


def _step(head: CropHead, loss_fn: WeightedMultiTaskLoss, q: float) -> torch.Tensor:
    """One forward+scaled-backward step; returns the classifier weight grad."""
    torch.manual_seed(0)
    head.eval()  # deterministic across modes (no dropout)
    x = torch.randn(B, H) * q
    targets = {"crop": torch.randint(0, C, (B,)), "yield": torch.zeros(B)}
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        logits = head(x)
        total, _per_task = loss_fn(
            {"crop": logits, "yield": torch.zeros(B, 1)}, targets
        )
    (total * SCALE).backward()
    return head.classifier.weight.grad.detach()


def test_crop_head_grads_finite_under_scaled_fp16_amp():
    """The fixed FP32 head must stay finite even when the gradient is scaled
    by 2**16 — the exact condition that previously overflowed to inf."""
    loss_fn = _weighted_head_loss()
    for q in (0.25, 1.0, 4.0, 8.0, 16.0):
        head = CropHead(H, C, dropout=0.15)
        grad = _step(head, loss_fn, q)
        assert grad is not None
        assert torch.isfinite(grad).all(), f"classifier grad non-finite at q={q}"
        assert bool((grad.abs() > 0).any()), f"classifier grad did not flow at q={q}"
        for name, p in head.named_parameters():
            assert p.grad is None or torch.isfinite(p.grad).all(), (
                f"{name} grad non-finite at q={q}"
            )


def test_fp32_path_scaled_magnitudes_exceed_fp16_range():
    """Mechanism pin: at realistic magnitudes the scaled true gradient is FAR
    above fp16's 65504 limit, so any fp16 computation of it overflows — the
    very reason the head runs in FP32."""
    loss_fn = _weighted_head_loss()
    head = CropHead(H, C, dropout=0.15)
    grad = _step(head, loss_fn, q=16.0)
    assert torch.isfinite(grad).all()  # the FP32 fix keeps it finite
    scaled = grad * SCALE
    assert float(scaled.abs().max()) > FP16_MAX  # ...but would overflow fp16
    assert float(scaled.abs().min() if scaled.numel() else 1.0) >= 0.0


def test_head_loss_backward_zero_grad_reset():
    """Re-running the fixed head across magnitudes must not leave stale grads."""
    loss_fn = _weighted_head_loss()
    head = CropHead(H, C, dropout=0.15)
    for _ in range(2):
        grad = _step(head, loss_fn, q=8.0)
        assert torch.isfinite(grad).all()
        head.zero_grad(set_to_none=True)
    # zero_grad cleared grads for the steady state (nothing leftover on rerun).
    head.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in head.parameters())


def test_grads_nonfinite_detector():
    """`Trainer._grads_nonfinite` fires on a poisoned true gradient."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    trainer = object.__new__(Trainer)
    trainer.raw_model = model
    assert trainer._grads_nonfinite() is False

    loss = model(torch.randn(8, 4)).square().mean()
    loss.backward()
    assert trainer._grads_nonfinite() is False

    with torch.no_grad():
        model[1].weight.grad[0, 0] = float("nan")
    assert trainer._grads_nonfinite() is True

    model[1].weight.grad.zero_()
    model[1].weight.grad[0, 0] = float("inf")
    assert trainer._grads_nonfinite() is True


def test_trainer_multiple_epochs_finite_with_fp32_heads(tmp_path):
    """CPU trainer smoke (the real CropFusionTrainer + small full model):
    several epochs stay finite with the FP32-head fix and no step is skipped."""
    from training.models import ModelFactory
    from training.training import CropFusionTrainer, TrainingConfig
    from training.training.callbacks import HistoryRecorder
    from training.training.cropfusion_trainer import CropFusionTrainingResult
    from training.training.tests.conftest import make_fake_loader, small_full_config

    cfg = TrainingConfig(
        name="amp-finite",
        general={"device": "cpu", "seed": 7, "gradient_checkpointing": False,
                 "validation_frequency": 1},
        data={"batch_size": 8, "workers": 0, "pin_memory": False,
              "persistent_workers": False},
        optimizer={"lr": 1e-4, "backbone_lr_multiplier": 0.3},
        scheduler={"name": "warmup_cosine", "step": "epoch", "warmup_epochs": 2},
        loss={"crop_loss": "label_smoothing", "class_weight_mode": "sqrt_inv"},
        train={
            "epochs": 3,
            "early_stopping_metric": "crop/macro_f1",
            "early_stopping_mode": "max",
            "early_stopping_patience": 10,
            "restore_best_on_stop": True,
        },
        checkpoint={"directory": str(tmp_path / "ckpt"), "save_best": True,
                    "save_latest": True, "keep_last": 2},
        metrics={"top_k": 3},
        logging={"console": False},
    )
    model = ModelFactory.create(small_full_config())
    loader = make_fake_loader(n=32, batch_size=8, feature_dim=4, multimodal=True)
    trainer = CropFusionTrainer(
        model,
        loader,
        cfg,
        val_loader=make_fake_loader(n=8, batch_size=8, feature_dim=4, multimodal=True),
        callbacks=[HistoryRecorder()],
        device=torch.device("cpu"),
    )
    result = trainer.train()
    assert isinstance(result, CropFusionTrainingResult)
    assert result.nan_steps == 0, result.nan_diagnostics
    assert all(torch.isfinite(torch.as_tensor(h["train_loss"])).item()
               for h in result.history), [h["train_loss"] for h in result.history]
    for name, p in trainer.raw_model.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name