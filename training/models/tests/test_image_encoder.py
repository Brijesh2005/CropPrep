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


def test_checkpointed_frames_are_contiguous():
    """Regression: the checkpointed per-timestep frame must be a materialised
    contiguous tensor, never the zero-stride ``expand`` view of the ``repeat``
    path. A broadcast view fed to the cuDNN conv stack under FP16 AMP can make
    the implicit contiguity copy / algorithm selection differ between the
    checkpointed forward and its backward recompute, which trips
    ``torch.utils.checkpoint.CheckpointError`` ("different number of tensors
    saved") on GPU."""
    enc = _encoder(channel_expansion="repeat")
    enc.train()
    enc.backbone.eval()
    enc.set_gradient_checkpointing(True)

    captured: dict[str, bool] = {}
    original_checkpoint = torch.utils.checkpoint.checkpoint

    def spy(*args, **kwargs):
        flat = args[2] if len(args) > 2 else None
        captured["contiguous"] = flat is not None and flat.is_contiguous()
        return original_checkpoint(*args, **kwargs)

    torch.utils.checkpoint.checkpoint = spy
    try:
        enc(torch.randn(2, 4, 1, 32, 32))
    finally:
        torch.utils.checkpoint.checkpoint = original_checkpoint

    assert captured.get("contiguous") is True


def test_checkpointing_train_mode_backward_finite_repeat_expansion():
    """Real config regression (R5.3 CheckpointError): efficientnetv2_s with
    ``repeat`` channel expansion, stochastic depth and the backbone LEFT in
    train mode must survive a forward+backward pass with per-timestep
    checkpointing enabled and yield finite, non-zero gradients."""
    enc = NdviEncoder(
        "efficientnetv2_s",
        input_size=64,
        channel_expansion="repeat",
        drop_path_rate=0.1,
    )
    enc.train()
    enc.set_gradient_checkpointing(True)
    x = torch.randn(2, 4, 1, 64, 64)

    out = enc(x)
    assert out.shape == (2, 4, enc.feature_dim)
    assert torch.isfinite(out).all().item(), "checkpointed forward is not finite"
    out.sum().backward()

    grads = [p.grad for p in enc.backbone.parameters() if p.grad is not None]
    assert grads, "no backbone gradients flowed"
    assert all(torch.isfinite(g).all().item() for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_checkpointing_grads_match_non_checkpointed():
    """Recomputation parity: with BN eval'd (deterministic) and drop out
    disabled, checkpointed backward gradients must match the plain backward —
    i.e. recomputation replays exactly the saved forward computation."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 1, 32, 32)

    def grads(enabled: bool):
        torch.manual_seed(123)
        enc = _encoder()
        enc.train()
        enc.backbone.eval()
        enc.set_gradient_checkpointing(enabled)
        out = enc(x)
        out.sum().backward()
        return [p.grad.clone() for p in enc.backbone.parameters() if p.grad is not None]

    g_plain = grads(enabled=False)
    g_ckpt = grads(enabled=True)
    assert len(g_plain) == len(g_ckpt)
    for a, b in zip(g_plain, g_ckpt):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (
            "checkpointed recomputation drifted from the plain backward"
        )
