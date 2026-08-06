"""Class-frequency weighting tests: statistics, weights, weighted losses."""

from __future__ import annotations

import pytest
import torch

from training.models.losses import (
    CrossEntropyLoss,
    FocalLoss,
    LabelSmoothingLoss,
)
from training.training.config import LossConfig
from training.training.exceptions import LossBuildError
from training.training.losses import (
    WeightedLabelSmoothingLoss,
    build_class_weights,
    build_multi_task_loss,
    build_task_loss,
    class_frequency_weights,
    compute_class_counts,
)


# --------------------------------------------------------------------------- #
# Class statistics
# --------------------------------------------------------------------------- #


def test_compute_class_counts():
    counts = compute_class_counts(torch.tensor([0, 0, 1, 2, 2, 2]))
    assert counts.tolist() == [2.0, 1.0, 3.0]


def test_compute_class_counts_empty():
    counts = compute_class_counts(torch.tensor([]))
    assert counts.numel() == 0


def test_class_frequency_weights_balanced():
    counts = torch.tensor([2.0, 1.0, 5.0])
    w = class_frequency_weights(counts, "balanced")
    # raw: total/(K*count) = [8/6, 8/3, 8/15], then normalised to mean 1
    assert w.tolist() == pytest.approx([15.0 / 17.0, 30.0 / 17.0, 6.0 / 17.0])
    assert w.mean().item() == pytest.approx(1.0)


def test_class_frequency_weights_sqrt_inv():
    counts = torch.tensor([4.0, 2.0, 1.0])
    w = class_frequency_weights(counts, "sqrt_inv")
    assert w[0] < w[1] < w[2]  # rarer classes weigh more
    assert w.mean().item() == pytest.approx(1.0)


def test_class_frequency_weights_effective_num():
    counts = torch.tensor([4.0, 2.0, 1.0])
    w = class_frequency_weights(counts, "effective_num", beta=0.99)
    assert w[0] < w[1] < w[2]
    assert w.mean().item() == pytest.approx(1.0)


def test_class_frequency_weights_zero_counts_stable():
    counts = torch.tensor([0.0, 2.0, 2.0])
    w = class_frequency_weights(counts, "balanced")
    assert torch.isfinite(w).all()
    assert w[0] > w[1]  # missing class gets the highest weight


def test_class_frequency_weights_unknown_mode():
    with pytest.raises(LossBuildError):
        class_frequency_weights(torch.tensor([1.0]), "bogus")


# --------------------------------------------------------------------------- #
# build_class_weights
# --------------------------------------------------------------------------- #


def test_build_class_weights_disabled():
    config = LossConfig(class_weight_mode="none")
    assert build_class_weights(config, 3, torch.tensor([2.0, 2.0, 2.0])) is None


def test_build_class_weights_pads_to_num_classes():
    config = LossConfig(class_weight_mode="balanced")
    counts = torch.tensor([1.0, 1.0])  # only two classes observed
    w = build_class_weights(config, 4, counts)
    assert w is not None
    assert w.numel() == 4
    assert torch.isfinite(w).all()


def test_build_class_weights_empty_counts_returns_none():
    config = LossConfig(class_weight_mode="balanced")
    assert build_class_weights(config, 3, torch.zeros(3)) is None


# --------------------------------------------------------------------------- #
# Weighted losses
# --------------------------------------------------------------------------- #


def test_weighted_label_smoothing_matches_model_loss():
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    model_loss = LabelSmoothingLoss(smoothing=0.1)
    weighted = WeightedLabelSmoothingLoss(smoothing=0.1)
    assert torch.allclose(
        model_loss(logits, targets), weighted(logits, targets), atol=1e-6
    )


def test_weighted_label_smoothing_applies_weights():
    torch.manual_seed(0)
    logits = torch.randn(16, 3)
    targets = torch.randint(0, 3, (16,))
    weights = torch.tensor([1.0, 0.0, 0.0])  # zero out classes 1, 2
    loss = WeightedLabelSmoothingLoss(smoothing=0.0, weight=weights)
    value = loss(logits, targets).item()
    mask = targets == 0
    # mean over ALL samples (weighted): sum(masked ce) / N
    expected = (
        torch.nn.functional.cross_entropy(
            logits[mask], targets[mask], reduction="mean"
        ).item()
        * mask.sum().item()
        / targets.numel()
    )
    assert value == pytest.approx(expected)


def test_build_task_loss_crop_weighted_variants():
    w = torch.tensor([1.0, 2.0, 3.0])
    ce = build_task_loss(
        "crop", LossConfig(crop_loss="cross_entropy"), class_weights=w
    )
    assert isinstance(ce, CrossEntropyLoss)
    ls = build_task_loss(
        "crop", LossConfig(crop_loss="label_smoothing"), class_weights=w
    )
    assert isinstance(ls, WeightedLabelSmoothingLoss)
    focal = build_task_loss("crop", LossConfig(crop_loss="focal"), class_weights=w)
    assert isinstance(focal, FocalLoss)


def test_build_task_loss_unweighted():
    # label_smoothing (default) still resolves to the weighted-capable loss,
    # but without a weight tensor it behaves identically to the plain one.
    ls = build_task_loss("crop", LossConfig(), class_weights=None)
    assert isinstance(ls, WeightedLabelSmoothingLoss)
    assert ls.weight is None


def test_multitask_loss_with_class_weights_forward():
    config = LossConfig(class_weight_mode="balanced")
    weights = torch.tensor([0.8, 1.2, 1.0])
    loss = build_multi_task_loss(config, class_weights={"crop": weights})
    torch.manual_seed(0)
    inputs = {
        "crop": torch.randn(8, 3, requires_grad=True),
        "yield": torch.randn(8, 1, requires_grad=True),
    }
    targets = {"crop": torch.randint(0, 3, (8,)), "yield": torch.randn(8, 1)}
    total, per_task = loss(inputs, targets)
    assert total.ndim == 0
    assert set(per_task) == {"crop", "yield"}
    assert total.requires_grad
    total.backward()
    assert inputs["crop"].grad is not None


def test_multitask_loss_weights_are_buffers():
    config = LossConfig(class_weight_mode="balanced")
    weights = torch.tensor([1.0, 1.0, 1.0])
    loss = build_multi_task_loss(config, class_weights={"crop": weights})
    crop = loss.tasks["crop"]
    assert hasattr(crop, "weight")
    assert crop.weight.tolist() == [1.0, 1.0, 1.0]
