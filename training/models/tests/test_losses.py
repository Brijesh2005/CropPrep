"""Loss interface tests: values, gradients, weighted multi-task."""

from __future__ import annotations

import pytest
import torch

from training.models import (
    CrossEntropyLoss,
    FocalLoss,
    HuberLoss,
    LabelSmoothingLoss,
    MSELoss,
    WeightedMultiTaskLoss,
    ModelConfig,
)


def _logits(shape=(4, 3), seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _targets(n: int, high: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, high, (n,), generator=g)


def test_cross_entropy_scalar():
    loss = CrossEntropyLoss()
    value = loss(_logits(), _targets(4))
    assert value.dim() == 0
    assert value > 0


def test_label_smoothing_equals_ce_when_smoothing_zero():
    smoothing = LabelSmoothingLoss(smoothing=0.0)
    ce = CrossEntropyLoss()
    logits = _logits()
    targets = _targets(4)
    assert torch.allclose(smoothing(logits, targets), ce(logits, targets), atol=1e-6)


def test_focal_reduces_to_ce_when_gamma_zero():
    focal = FocalLoss(gamma=0.0)
    ce = CrossEntropyLoss()
    logits = _logits()
    targets = _targets(4)
    assert torch.allclose(focal(logits, targets), ce(logits, targets), atol=1e-6)


def test_mse_scalar_and_huber():
    pred = torch.randn(4, 1)
    target = torch.randn(4)
    mse = MSELoss()
    huber = HuberLoss(beta=1.0)
    assert mse(pred, target).dim() == 0
    assert huber(pred, target).dim() == 0


def test_huber_agrees_with_smoothl1():
    import torch.nn.functional as F

    huber = HuberLoss(beta=0.5)
    pred = torch.randn(6, 1)
    target = torch.randn(6)
    assert torch.allclose(huber(pred, target), F.smooth_l1_loss(pred.squeeze(1), target, beta=0.5))


def test_classification_losses_gradient_flow():
    for criterion in (CrossEntropyLoss(), LabelSmoothingLoss(), FocalLoss()):
        logits = _logits(seed=1).requires_grad_(True)
        criterion(logits, _targets(4)).backward()
        assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_focal_with_alpha():
    focal = FocalLoss(gamma=2.0, alpha=torch.tensor([0.3, 0.7, 0.4]))
    value = focal(_logits(seed=5), _targets(4))
    assert value.dim() == 0 and value > 0


def test_mse_sum_reduction():
    mse = MSELoss(reduction="sum")
    pred = torch.randn(4, 1)
    target = torch.randn(4)
    summed = mse(pred, target)
    mean = MSELoss()(pred, target)
    assert torch.allclose(summed, mean * 4, atol=1e-5)


def test_label_smoothing_rejects_invalid():
    with pytest.raises(Exception):
        LabelSmoothingLoss(smoothing=1.0)


def test_regression_losses_gradient_flow():
    for criterion in (MSELoss(), HuberLoss()):
        pred = torch.randn(4, 1, requires_grad=True)
        criterion(pred, torch.randn(4)).backward()
        assert pred.grad is not None


def test_weighted_multi_task_fixed():
    config = ModelConfig().loss
    loss = WeightedMultiTaskLoss(config)
    logits = _logits(seed=2)
    targets = _targets(4)
    pred = torch.randn(4, 1)
    total, per_task = loss(
        {"crop": logits, "yield": pred},
        {"crop": targets, "yield": torch.randn(4)},
    )
    assert total.dim() == 0
    assert set(per_task) == {"crop", "yield"}
    # fixed weights: total == crop_weight*ce + yield_weight*huber
    expected = (
        config.crop_weight * per_task["crop"]
        + config.yield_weight * per_task["yield"]
    )
    assert torch.allclose(total, expected)


def test_weighted_multi_task_learnable_has_log_vars():
    config = ModelConfig().loss.model_copy(update={"weighting_mode": "learnable"})
    loss = WeightedMultiTaskLoss(config)
    logits = _logits(seed=3)
    targets = _targets(4)
    total, per_task = loss(
        {"crop": logits, "yield": torch.randn(4, 1)},
        {"crop": targets, "yield": torch.randn(4)},
    )
    assert total.dim() == 0
    assert hasattr(loss, "log_var_crop")
    assert hasattr(loss, "log_var_yield")


def test_weighted_multi_task_missing_task_raises():
    config = ModelConfig().loss
    loss = WeightedMultiTaskLoss(config)
    with pytest.raises(Exception):
        loss({"crop": _logits(seed=4)}, {"crop": _targets(4)})
