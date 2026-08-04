"""Temporal + cross-modal attention tests."""

from __future__ import annotations

import numpy as np
import pytest

from ai.explainability import CrossModalExplainer, TemporalAttentionExplainer
from ai.explainability.config import ExplainabilityConfig
from ai.explainability.exceptions import AttentionError


def test_temporal_importance(full_model, full_sample):
    explainer = TemporalAttentionExplainer(full_model, ExplainabilityConfig().temporal_attention)
    result = explainer.explain(full_sample)
    assert result["importance"].shape == (4,)
    assert result["ranking"] is not None
    # The padded timestep (mask=0) has zero importance.
    assert result["importance"][-1] < 1e-6
    # Rollout matrix is square.
    assert result["rollout"].shape[0] == result["rollout"].shape[1] == 5  # CLS + 4


def test_rollout_defaults_to_identity_when_single_layer(full_model, full_sample):
    config = ExplainabilityConfig(temporal_attention={"rollout": True, "include_residual": True})
    explainer = TemporalAttentionExplainer(full_model, config.temporal_attention)
    result = explainer.explain(full_sample)
    assert np.isfinite(result["rollout"]).all()


def test_tabular_only_raises(tabular_model, sample):
    with pytest.raises(AttentionError):
        TemporalAttentionExplainer(tabular_model)


def test_cross_modal_components(full_model, full_sample):
    explainer = CrossModalExplainer(full_model, ExplainabilityConfig().cross_modal)
    result = explainer.explain(full_sample, feature_names=["a", "b", "c", "d", "e"])
    assert "cross_attention_score" in result
    assert "gates" in result
    assert result["gates"]  # image_gate / tabular_gate present
    assert result["feature_importance"] is not None
    assert result["observation_importance"] is not None
    heatmap = result["cross_modal_heatmap"]
    assert heatmap is not None
    assert heatmap.ndim == 2  # [T, F]
