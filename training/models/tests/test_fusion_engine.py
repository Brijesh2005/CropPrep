"""CrossModalFusionEngine tests: construction, forward, gates, ablations."""

from __future__ import annotations

import pytest
import torch

from training.models import (
    AdaptiveGatedFusion,
    CrossAttention,
    CrossModalFusionEngine,
    ModelFactory,
    SharedMultimodalEncoder,
)
from training.models.exceptions import ModelConfigurationError
from training.models.fusion_engine import FusionOutput


def _model_embeddings(model, batch):
    """Recompute the tabular / image embeddings the way the model's forward does."""
    tab = model.tab_encoder(batch["tabular"])
    fused_seq = model.image_fusion(
        model.ndvi_encoder(batch["ndvi"]), model.evi_encoder(batch["evi"])
    )
    img = model.temporal_transformer(fused_seq, mask=batch["temporal_mask"])
    return tab, img


def test_engine_built_for_multimodal_model(model):
    engine = model.fusion_engine
    assert isinstance(engine, CrossModalFusionEngine)
    assert isinstance(engine.cross_attention, CrossAttention)
    assert isinstance(engine.gated_fusion, AdaptiveGatedFusion)
    assert isinstance(engine.shared_encoder, SharedMultimodalEncoder)
    assert engine.output_dim == model.shared_encoder.output_dim == 128


def test_engine_forward_contract(model, batch):
    tab, img = _model_embeddings(model, batch)
    out = model.fusion_engine(img, tab)
    assert isinstance(out, FusionOutput)
    assert tuple(out.shared_embedding.shape) == (4, 128)
    assert set(out.gates) == {"image_gate", "tabular_gate", "fusion_gate"}
    assert out.cross_output is not None
    assert out.image_token is not None
    assert out.tabular_token is not None
    assert out.temporal_token is None


def test_engine_forward_equals_model_shared(model, batch):
    model.eval()
    with torch.no_grad():
        model_out = model(batch)
        tab, img = _model_embeddings(model, batch)
        engine_out = model.fusion_engine(img, tab)
    assert torch.allclose(model_out.shared_representation, engine_out.shared_embedding)
    assert torch.allclose(
        model_out.gates["image_gate"], engine_out.gates["image_gate"]
    )


def test_engine_return_attention(model, batch):
    tab, img = _model_embeddings(model, batch)
    out = model.fusion_engine(img, tab, return_attention=True)
    assert "cross_attention" in out.gates
    weights = out.gates["cross_attention"]
    assert weights.dim() == 3  # [B, q=1, k=1], head-averaged
    assert (weights >= 0).all() and (weights <= 1).all()


def test_engine_temporal_stream(config):
    cfg = config.model_copy(deep=True)
    cfg.fusion.use_temporal_stream = True
    model = ModelFactory.create(cfg)
    batch = model.sample_batch(batch_size=2, seq_len=4)
    out = model(batch)
    assert "temporal_gate" in out.gates
    assert out.temporal_embedding is not None
    assert tuple(out.temporal_embedding.shape) == (2, model.image_dim)
    # gates still bounded in [0, 1]
    for key in ("image_gate", "tabular_gate", "temporal_gate", "fusion_gate"):
        gate = out.gates[key]
        assert (gate >= 0).all() and (gate <= 1).all()


def test_engine_as_dict(model, batch):
    tab, img = _model_embeddings(model, batch)
    d = model.fusion_engine(img, tab).as_dict()
    assert set(d) == {
        "shared_embedding",
        "fused",
        "cross_output",
        "image_token",
        "tabular_token",
        "temporal_token",
        "gates",
    }


def test_engine_requires_both_modalities(config):
    with pytest.raises(ModelConfigurationError):
        CrossModalFusionEngine(config, image_dim=0, tabular_dim=16)


def test_engine_no_cross_no_gated(config):
    cfg = config.model_copy(deep=True)
    cfg.cross_attention.enabled = False
    cfg.gated_fusion.enabled = False
    model = ModelFactory.create(cfg)
    engine = model.fusion_engine
    assert engine.cross_attention is None
    assert engine.gated_fusion is None
    batch = model.sample_batch(batch_size=2, seq_len=4)
    out = model(batch)
    assert out.gates == {}
    assert tuple(out.shared_representation.shape) == (2, 128)
    assert engine.output_dim == 128


def test_engine_no_gated_concat(config):
    cfg = config.model_copy(deep=True)
    cfg.gated_fusion.enabled = False  # cross-attention stays on
    model = ModelFactory.create(cfg)
    batch = model.sample_batch(batch_size=2, seq_len=4)
    out = model(batch)
    assert out.gates == {}
    assert out.shared_representation.shape == (2, 128)


def test_engine_residual_fusion_ablation(config):
    cfg = config.model_copy(deep=True)
    cfg.fusion.residual_fusion = False
    model = ModelFactory.create(cfg)
    assert model.fusion_engine.residual_fusion is False
    batch = model.sample_batch(batch_size=2, seq_len=4)
    out = model(batch)
    assert tuple(out.crop_logits.shape) == (2, 3)


def test_engine_gradient_flows(model, batch):
    model.train()
    tab, img = _model_embeddings(model, batch)
    out = model.fusion_engine(img, tab)
    out.shared_embedding.sum().backward()
    missing = [
        name
        for name, p in model.fusion_engine.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert missing == []


def test_engine_output_dim_matches_shared_encoder(config):
    cfg = config.model_copy(deep=True)
    cfg.shared_encoder.out_dim = 96
    model = ModelFactory.create(cfg)
    assert model.fusion_engine.output_dim == 96 == model.shared_encoder.output_dim
