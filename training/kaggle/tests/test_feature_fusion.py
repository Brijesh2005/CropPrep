"""Unit tests for the feature-level fusion campaign.

Fast model/pattern tests run immediately; the frozen-data tests share one
pipeline load (module fixture) and assert the load-bearing honest-evaluation
constraints (no leakage, no split overlap, embedding cache round-trip).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from training.models.feature_fusion import (
    FeatureFusion, FeatureFusionClassifier, ImageFeatureEncoder,
    TabularFeatureEncoder, TemporalAttentionPool,
)

B = 8
T = 4
F = 12
TAB_D = 10


@pytest.fixture(scope="module")
def pipeline():
    from training.kaggle.scripts.feature_fusion_utils import load_pipeline
    pop, feat, img, idx, y = load_pipeline(with_location=False)
    return pop, feat, img, idx, y


@pytest.fixture(scope="module")
def fused_model():
    return FeatureFusionClassifier(tabular_dim=TAB_D, image_feature_dim=F,
                                   num_classes=1)


def _seq():
    return torch.randn(B, T, F)


def _msk():
    return torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0],
                         [1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0],
                         [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)


def test_tabular_encoder_shape():
    m = TabularFeatureEncoder(TAB_D)
    x = torch.randn(B, TAB_D)
    assert m(x).shape == (B, 256)


def test_image_encoder_shape():
    m = ImageFeatureEncoder(F)
    assert m(_seq()).shape == (B, T, 512)


def test_temporal_attention_masks_padded():
    m = TemporalAttentionPool(F)
    seq = _seq()
    msk = _msk()
    pool_a = m(seq, msk)
    seq2 = seq + (1.0 - msk.unsqueeze(-1)) * 999.0
    pool_b = m(seq2, msk)
    assert torch.allclose(pool_a, pool_b, atol=1e-8)
    fully_masked = torch.zeros(2, T)
    out = m(seq[:2], fully_masked)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out))


def test_fusion_output_shape(fused_model):
    tab = torch.randn(B, TAB_D)
    out = fused_model(tab, _seq(), _msk())
    assert out["fused_embedding"].shape == (B, 256)
    assert out["tabular_embedding"].shape == (B, 256)
    assert out["image_embedding"].shape == (B, 512)
    assert out["logits"].shape == (B, 1)


def test_padded_obs_do_not_affect_output(fused_model):
    fused_model.eval()
    tab = torch.randn(B, TAB_D)
    msk = _msk()
    s1 = _seq()
    a = fused_model(tab, s1, msk)
    seq = s1 + (1.0 - msk.unsqueeze(-1)) * 1234.5
    b = fused_model(tab, seq, msk)
    assert torch.allclose(a["image_embedding"], b["image_embedding"], atol=1e-6)
    assert torch.allclose(a["logits"], b["logits"], atol=1e-6)


def test_no_train_test_field_overlap(pipeline):
    from training.kaggle.scripts.feature_fusion_utils import audit_split_overlap
    pop, *_ = pipeline
    aud = audit_split_overlap(pop)
    assert aud["issues"] == []
    assert aud["split_sizes"]["train"] == 1540
    assert aud["split_sizes"]["val"] == 1465
    assert aud["split_sizes"]["test"] == 980


def test_leakage_detection_catches_forbidden_columns():
    from training.kaggle.scripts.feature_fusion_utils import audit_feature_leakage
    issues = audit_feature_leakage(["ndvi", "soil_ph", "is_cropland"])["issues"]
    assert issues == []
    bad = audit_feature_leakage(
        ["soil_ph"], ["Yield_Proxy_NPP", "dominance_margin",
                      "coconut_fraction", "source_crop_name", "ndvi"])["issues"]
    assert len(bad) == 4
    ok = audit_feature_leakage(["is_cropland"], ["ndvi"])["issues"]
    assert ok == []


def test_binary_classifier_output(fused_model):
    fused_model.eval()
    tab = torch.randn(B, TAB_D)
    with torch.no_grad():
        out = fused_model(tab, _seq(), _msk())
    logits = out["logits"]
    assert logits.shape == (B, 1)
    p = torch.sigmoid(logits)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_embedding_cache_reload_consistency(tmp_path):
    from training.kaggle.scripts.feature_fusion_utils import (
        load_image_cache, save_image_cache,
    )
    rng = np.random.RandomState(0)
    img = {"X": rng.randn(6, T, F), "mask": rng.rand(6, T), "cols": ["a"] * F}
    import pandas as pd
    pop = pd.DataFrame({
        "field_id": [f"F{i}" for i in range(6)],
        "split": ["train", "val", "test"] * 2,
        "survey_year": [2020] * 6,
    })
    out = tmp_path / "cache.npz"
    meta = tmp_path / "meta.json"
    img["audit"] = {"timesteps": [2018, 2019, 2020, 2021]}
    save_image_cache(pop, img, out, meta)
    c = load_image_cache(out)
    assert np.allclose(c["X"], img["X"])
    assert np.allclose(c["mask"], img["mask"])
    assert list(c["field_id"]) == list(pop["field_id"])


def test_gated_fusion_matches_concat_shapes(fused_model):
    m = FeatureFusionClassifier(tabular_dim=TAB_D, image_feature_dim=F,
                                fusion_type="gated", num_classes=1)
    out = m(torch.randn(B, TAB_D), _seq(), _msk())
    assert out["fused_embedding"].shape == (B, 256)
    g = out["reasoning"]
    assert g["tabular_gate"].shape == (B, 256)
    assert g["image_gate"].shape == (B, 512)


def test_pipeline_audit_passes(pipeline):
    from training.kaggle.scripts.feature_fusion_utils import audit_all
    pop, feat, img, *_ = pipeline
    aud = audit_all(pop, {"cols": feat["cols"]})
    assert aud["pass"]
    assert img["X"].shape[1] == 4
    assert img["X"].shape[2] == 12