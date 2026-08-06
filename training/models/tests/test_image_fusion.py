"""ImageFusion tests: all four methods produce correct shapes + gradients."""

from __future__ import annotations

import pytest
import torch

from training.models import ImageFusion
from training.models.exceptions import ModelConfigurationError

METHODS = ("concat", "weighted_sum", "learnable", "attention")


@pytest.mark.parametrize("method", METHODS)
def test_shape_each_method(method):
    fusion = ImageFusion(8, 8, method=method, hidden_dim=16)
    ndvi = torch.randn(2, 4, 8)
    evi = torch.randn(2, 4, 8)
    out = fusion(ndvi, evi)
    assert out.shape == (2, 4, 16)


@pytest.mark.parametrize("method", METHODS)
def test_gradient_each_method(method):
    fusion = ImageFusion(8, 8, method=method, hidden_dim=16)
    out = fusion(torch.randn(2, 3, 8), torch.randn(2, 3, 8))
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in fusion.parameters())


def test_mismatched_input_dims():
    fusion = ImageFusion(8, 16, method="learnable", hidden_dim=16)
    out = fusion(torch.randn(2, 4, 8), torch.randn(2, 4, 16))
    assert out.shape == (2, 4, 16)


def test_default_out_dim():
    fusion = ImageFusion(32, 32, method="concat")
    assert fusion.out_dim == 32


def test_unknown_method():
    with pytest.raises(ModelConfigurationError):
        ImageFusion(8, 8, method="magic")
