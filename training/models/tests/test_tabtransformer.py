"""TabTransformer tests: shapes, OOV handling, gradients."""

from __future__ import annotations

import pytest
import torch

from training.models import TabTransformer, ModelFactory
from training.models.config import TabularModelConfig


@pytest.fixture(scope="session")
def tab_config() -> TabularModelConfig:
    return TabularModelConfig(
        numeric_dim=3, categorical_cardinalities=[4, 2],
        embedding_dim=32, depth=2, num_heads=4, ff_dim=64, dropout=0.0,
    )


def test_forward_shape(tab_config):
    tab = TabTransformer(tab_config)
    x = torch.randn(5, 5)
    out = tab(x)
    assert out.shape == (5, 32)


def test_feature_count_matches_width(tab_config):
    tab = TabTransformer(tab_config)
    assert tab.feature_count == 1 + 2  # continuous token + 2 categoricals


def test_oov_index_mapped_to_padding():
    cfg = TabularModelConfig(
        numeric_dim=0, categorical_cardinalities=[3],
        embedding_dim=16, depth=1, num_heads=2, ff_dim=32, dropout=0.0,
    )
    tab = TabTransformer(cfg)
    # code -1 (unseen) should be zero-vector due to padding_idx=0
    x = torch.tensor([[0.0], [-1.0]])
    with torch.no_grad():
        out = tab(x)
    assert torch.isfinite(out).all()
    # embedding table has cardinality+1 rows
    assert tab.cat_embeddings[0].embedding.num_embeddings == 4


def test_continuous_only():
    cfg = TabularModelConfig(
        numeric_dim=4, categorical_cardinalities=[],
        embedding_dim=16, depth=1, num_heads=2, ff_dim=32, dropout=0.0,
    )
    tab = TabTransformer(cfg)
    out = tab(torch.randn(3, 4))
    assert out.shape == (3, 16)


def test_categorical_only():
    cfg = TabularModelConfig(
        numeric_dim=0, categorical_cardinalities=[5],
        embedding_dim=16, depth=1, num_heads=2, ff_dim=32, dropout=0.0,
    )
    tab = TabTransformer(cfg)
    out = tab(torch.randint(0, 5, (3, 1)).float())
    assert out.shape == (3, 16)


def test_wrong_width_raises(tab_config):
    tab = TabTransformer(tab_config)
    with pytest.raises(Exception):
        tab(torch.randn(3, 4))


def test_gradient_flows(tab_config):
    tab = TabTransformer(tab_config)
    out = tab(torch.randn(4, 5))
    out.sum().backward()
    assert all(p.grad is not None for p in tab.parameters() if p.requires_grad)


def test_within_full_model(batch, model):
    out = model.tab_encoder(batch["tabular"])
    assert out.shape == (batch["tabular"].size(0), model.tab_dim)
