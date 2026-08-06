"""Image encoder tests: shapes, channel expansion, resize, freezing."""

from __future__ import annotations

import pytest
import torch

from training.models import NdviEncoder, EviEncoder
from training.models.exceptions import ModelConfigurationError, ShapeMismatchError


def _encoder(**kwargs):
    return NdviEncoder("mobilenetv3_small_050", input_size=32, **kwargs)


def test_sequence_shape():
    enc = _encoder()
    x = torch.randn(2, 4, 1, 32, 32)
    out = enc(x)
    assert out.shape == (2, 4, enc.feature_dim)
    assert enc.feature_dim > 0


def test_feature_dim_probed_positive():
    enc = _encoder()
    assert enc.feature_dim > 0


def test_resize_from_small_patch():
    enc = _encoder()
    x = torch.randn(2, 3, 1, 16, 16)  # smaller than input_size 32
    out = enc(x)
    assert out.shape == (2, 3, enc.feature_dim)


def test_channel_expansion_conv():
    enc = _encoder(channel_expansion="conv")
    x = torch.randn(2, 2, 1, 32, 32)
    out = enc(x)
    assert out.shape == (2, 2, enc.feature_dim)
    assert enc.expand_conv is not None


def test_3_channel_direct_input():
    enc = _encoder()
    x = torch.randn(2, 1, 3, 32, 32)
    out = enc(x)
    assert out.shape == (2, 1, enc.feature_dim)


def test_bad_channel_expansion():
    with pytest.raises(ModelConfigurationError):
        _encoder(channel_expansion="nope")


def test_bad_channels():
    enc = _encoder()
    with pytest.raises(ShapeMismatchError):
        enc(torch.randn(2, 4, 2, 32, 32))


def test_freeze_backbone():
    enc = _encoder(freeze_backbone=True)
    assert all(p.requires_grad is False for p in enc.backbone.parameters())
    # expand_conv is not part of the backbone and stays trainable
    assert enc.expand_conv is None or enc.expand_conv.weight.requires_grad


def test_evi_and_ndvi_independent():
    ndvi = NdviEncoder("mobilenetv3_small_050", input_size=32)
    evi = EviEncoder("mobilenetv3_small_050", input_size=32)
    # independent modules -> distinct parameter objects
    assert ndvi.backbone is not evi.backbone


def test_gradient_flows():
    enc = _encoder()
    out = enc(torch.randn(2, 2, 1, 32, 32))
    out.sum().backward()
    assert any(p.grad is not None for p in enc.backbone.parameters())
