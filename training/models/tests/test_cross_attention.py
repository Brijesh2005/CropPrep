"""CrossAttention tests: shapes, residual, attention-return mode."""

from __future__ import annotations

import pytest
import torch

from training.models import CrossAttention


def test_shape():
    ca = CrossAttention(query_dim=32, key_dim=16, num_heads=4, out_dim=24)
    out = ca(torch.randn(5, 32), torch.randn(5, 16))
    assert out.shape == (5, 24)


def test_gradient_flows():
    ca = CrossAttention(query_dim=32, key_dim=16, num_heads=4, out_dim=24)
    out = ca(torch.randn(4, 32), torch.randn(4, 16))
    out.sum().backward()
    assert all(p.grad is not None for p in ca.parameters())


def test_return_attention():
    ca = CrossAttention(
        query_dim=32, key_dim=16, num_heads=4, out_dim=24, return_attention=True
    )
    out, weights = ca(torch.randn(2, 32), torch.randn(2, 16))
    assert out.shape == (2, 24)
    assert weights.shape[0] == 2
    assert torch.isfinite(weights).all()


def test_output_changes_with_inputs():
    ca = CrossAttention(query_dim=16, key_dim=16, num_heads=4, out_dim=16)
    img = torch.randn(2, 16)
    tab1 = torch.randn(2, 16)
    tab2 = torch.randn(2, 16)
    ca.eval()
    with torch.no_grad():
        o1 = ca(img, tab1)
        o2 = ca(img, tab2)
    assert not torch.allclose(o1, o2)
