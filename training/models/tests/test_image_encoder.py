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


# --------------------------------------------------------------------------- #
# Per-timestep gradient checkpointing (OOM fix for B*T multi-frame batches)
# --------------------------------------------------------------------------- #


def test_checkpointing_forward_shape_parity():
    """Checkpointed vs non-checkpointed forwards produce the same [B, T, D].

    Non-reentrant per-timestep checkpointing must be a pure memory optimisation:
    the output tensor contract and the computed values are unchanged. The
    backbone is put in eval() so batch-norm running parts are deterministic and
    dropout is off, while the encoder stays in train() so the checkpoint path
    (self.training) is exercised.
    """
    enc = _encoder()
    enc.train()
    enc.backbone.eval()
    x = torch.randn(2, 4, 1, 32, 32)

    out_plain = enc(x)
    enc.set_gradient_checkpointing(True)
    out_ckpt = enc(x)

    assert out_plain.shape == (2, 4, enc.feature_dim)
    assert out_ckpt.shape == (2, 4, enc.feature_dim)
    assert torch.allclose(out_plain, out_ckpt, atol=1e-4, rtol=1e-4)


def test_checkpointing_gradients_flow_finite():
    """Per-timestep checkpointing still back-propagates finite gradients."""
    enc = _encoder()
    enc.train()
    enc.backbone.eval()
    enc.set_gradient_checkpointing(True)

    out = enc(torch.randn(2, 4, 1, 32, 32))
    out.sum().backward()

    grads = [p.grad for p in enc.backbone.parameters() if p.grad is not None]
    assert grads, "no backbone gradients flowed"
    assert all(torch.isfinite(g).all().item() for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_checkpointing_disabled_flag_absent_by_default():
    enc = _encoder()
    assert enc._checkpoint_timesteps is False
