"""TemporalTransformer tests: variable length, masks, shapes, gradients."""

from __future__ import annotations

import pytest
import torch

from training.models import TemporalTransformer
from training.models.config import TemporalModelConfig


@pytest.fixture(scope="session")
def temporal_config() -> TemporalModelConfig:
    return TemporalModelConfig(
        d_model=32, depth=2, num_heads=4, ff_dim=128, dropout=0.0,
        embedding_dim=24, position_encoding="learned", max_len=8,
    )


def test_shape_with_mask(temporal_config):
    tt = TemporalTransformer(temporal_config, input_dim=16)
    seq = torch.randn(3, 5, 16)
    mask = torch.cat([torch.ones(3, 3), torch.zeros(3, 2)], dim=1)
    out = tt(seq, mask=mask)
    assert out.shape == (3, 24)


def test_shape_without_mask(temporal_config):
    tt = TemporalTransformer(temporal_config, input_dim=16)
    out = tt(torch.randn(3, 5, 16))
    assert out.shape == (3, 24)


def test_sequence_length_variation(temporal_config):
    tt = TemporalTransformer(temporal_config, input_dim=16)
    out1 = tt(torch.randn(2, 3, 16), mask=torch.ones(2, 3))
    out2 = tt(torch.randn(2, 6, 16), mask=torch.ones(2, 6))
    assert out1.shape == (2, 24)
    assert out2.shape == (2, 24)


def test_exceeds_max_len_raises(temporal_config):
    tt = TemporalTransformer(temporal_config, input_dim=16)
    with pytest.raises(Exception):
        tt(torch.randn(2, 9, 16))


def test_all_padding_mask_still_finite(temporal_config):
    tt = TemporalTransformer(temporal_config, input_dim=16)
    seq = torch.randn(2, 4, 16)
    mask = torch.zeros(2, 4)  # everything padding
    out = tt(seq, mask=mask)
    assert torch.isfinite(out).all()


def test_gradient_flows(temporal_config):
    tt = TemporalTransformer(temporal_config, input_dim=16)
    out = tt(torch.randn(3, 4, 16), mask=torch.ones(3, 4))
    out.sum().backward()
    assert any(p.grad is not None for p in tt.parameters())


def test_sinusoidal_positional_encoding(temporal_config):
    cfg = temporal_config.model_copy(update={"position_encoding": "sinusoidal"})
    tt = TemporalTransformer(cfg, input_dim=16)
    out = tt(torch.randn(2, 4, 16), mask=torch.ones(2, 4))
    assert out.shape == (2, 24)


def test_no_positional_encoding(temporal_config):
    cfg = temporal_config.model_copy(update={"position_encoding": "none"})
    tt = TemporalTransformer(cfg, input_dim=16)
    out = tt(torch.randn(2, 4, 16))
    assert out.shape == (2, 24)
