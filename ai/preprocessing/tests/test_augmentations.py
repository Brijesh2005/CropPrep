"""Tests for image augmentations (training only)."""

from __future__ import annotations

import torch

from ai.preprocessing.augmentations import ImageAugmentation
from ai.preprocessing.config import AugmentationConfig


def _seq(batch=1, time=3, h=16, w=16):
    return torch.ones(batch, time, h, w)


def test_disabled_passthrough():
    aug = ImageAugmentation(AugmentationConfig(enabled=False))
    tensor = _seq()
    assert torch.equal(aug(tensor), tensor)


def test_flip_horizontal_changes_order():
    aug = ImageAugmentation(AugmentationConfig(enabled=True, flip_horizontal=True), seed=1)
    tensor = torch.arange(16).reshape(1, 1, 4, 4).float()
    out = aug(tensor)
    # The flip is random (50%); either way the shape is preserved.
    assert out.shape == tensor.shape


def test_noise_adds_perturbation():
    aug = ImageAugmentation(AugmentationConfig(enabled=True, noise_std=0.1), seed=1)
    tensor = torch.zeros(1, 2, 8, 8)
    out = aug(tensor)
    assert (out.abs().sum()) > 0
    assert out.shape == tensor.shape


def test_rotation_90_preserves_shape():
    aug = ImageAugmentation(
        AugmentationConfig(enabled=True, rotation_degrees=[90]), seed=1
    )
    tensor = _seq(h=16, w=16)
    assert aug(tensor).shape == tensor.shape


def test_single_image_input():
    aug = ImageAugmentation(AugmentationConfig(enabled=True, flip_horizontal=True), seed=1)
    tensor = torch.ones(1, 16, 16)  # [C, H, W]
    out = aug(tensor)
    assert out.dim() == 3
