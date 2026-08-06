"""AdaptiveGatedFusion tests: shapes, gate ranges, gradients."""

from __future__ import annotations

import torch

from training.models import AdaptiveGatedFusion


def _fusion():
    return AdaptiveGatedFusion(
        image_dim=32, tabular_dim=16, cross_dim=24,
        out_dim=64, hidden_dim=32, dropout=0.0,
    )


def test_output_shape():
    gf = _fusion()
    out = gf(
        torch.randn(4, 32),
        torch.randn(4, 16),
        torch.randn(4, 24),
    )
    assert out["fused"].shape == (4, 64)


def test_gates_in_unit_interval():
    gf = _fusion()
    out = gf(
        torch.randn(6, 32),
        torch.randn(6, 16),
        torch.randn(6, 24),
    )
    for name in ("image_gate", "tabular_gate", "fusion_gate"):
        gate = out[name]
        assert gate.shape == (6, 1)
        assert (gate >= 0).all() and (gate <= 1).all()


def test_gates_are_sample_dependent():
    gf = _fusion()
    gf.eval()
    a = torch.randn(1, 32)
    b = torch.randn(1, 32)
    tab = torch.randn(1, 16)
    cross = torch.randn(1, 24)
    with torch.no_grad():
        o1 = gf(a, tab, cross)
        o2 = gf(b, tab, cross)
    assert not torch.allclose(o1["image_gate"], o2["image_gate"])


def test_gradient_flows():
    gf = _fusion()
    out = gf(
        torch.randn(3, 32),
        torch.randn(3, 16),
        torch.randn(3, 24),
    )
    out["fused"].sum().backward()
    assert all(p.grad is not None for p in gf.parameters())


def test_modality_tokens_present():
    gf = _fusion()
    out = gf(
        torch.randn(2, 32),
        torch.randn(2, 16),
        torch.randn(2, 24),
    )
    assert out["image_token"].shape == (2, 64)
    assert out["tabular_token"].shape == (2, 64)
