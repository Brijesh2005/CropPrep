"""SharedMultimodalEncoder tests: shapes, gradients, output width."""

from __future__ import annotations

import pytest
import torch

from training.models import SharedMultimodalEncoder
from training.models.config import SharedEncoderConfig


def _encoder(out_dim=96):
    return SharedMultimodalEncoder(
        input_dim=32,
        config=SharedEncoderConfig(
            d_model=32, depth=2, num_heads=4, ff_dim=128, out_dim=out_dim,
        ),
    )


def test_shape():
    enc = _encoder(out_dim=96)
    out = enc(torch.randn(5, 32))
    assert out.shape == (5, 96)


def test_output_width_configurable():
    for out_dim in (64, 128):
        enc = _encoder(out_dim=out_dim)
        assert enc(torch.randn(2, 32)).shape == (2, out_dim)


def test_gradient_flows():
    enc = _encoder()
    enc(torch.randn(3, 32)).sum().backward()
    assert all(p.grad is not None for p in enc.parameters())


def test_input_dim_mismatch_raises():
    enc = _encoder()
    with pytest.raises(Exception):
        enc(torch.randn(2, 40))
