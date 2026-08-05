"""SHAP tests: kernel correctness, gradient method, global importance, export."""

from __future__ import annotations

import numpy as np

from training.explainability import SHAPExplainer
from training.explainability.config import ExplainabilityConfig


def test_kernel_shap_local(tabular_model, samples, explainability_config):
    explainer = SHAPExplainer(tabular_model, explainability_config.shap)
    result = explainer.explain(samples[0], samples[1:], kind="crop")
    assert result.values.shape == (samples[0]["tabular"].shape[0],)
    assert np.isfinite(result.values).all()
    # f(x) ≈ base + sum(phi)
    batch = {k: v.unsqueeze(0) for k, v in samples[0].items() if hasattr(v, "unsqueeze")}
    logits = tabular_model(batch).crop_logits
    target_class = int(logits[0].argmax().item())
    f_x = float(logits[0, target_class].item())
    assert abs(f_x - (result.base_value + result.values.sum())) < 1e-2


def test_kernel_shap_yield(tabular_model, samples, explainability_config):
    explainer = SHAPExplainer(tabular_model, explainability_config.shap)
    result = explainer.explain(samples[0], samples[1:], kind="yield")
    assert result.values.shape == (samples[0]["tabular"].shape[0],)
    assert np.isfinite(result.values).all()


def test_gradient_shap(tabular_model, samples):
    config = ExplainabilityConfig(shap={"method": "gradient"})
    explainer = SHAPExplainer(tabular_model, config.shap)
    result = explainer.explain(samples[0], samples[1:], kind="crop")
    assert result.values.shape == (samples[0]["tabular"].shape[0],)


def test_global_importance(tabular_model, samples, explainability_config):
    explainer = SHAPExplainer(tabular_model, explainability_config.shap)
    result = explainer.global_importance(
        samples, samples, kind="crop", feature_names_=[f"f{i}" for i in range(5)]
    )
    assert result.global_importance is not None
    assert len(result.global_importance) == 5
    assert all(v >= 0 for v in result.global_importance.values())


def test_csv_json_export(tabular_model, samples, explainability_config, tmp_path):
    explainer = SHAPExplainer(tabular_model, explainability_config.shap)
    result = explainer.explain(samples[0], samples[1:], kind="crop")
    csv_path = explainer.to_csv(result, tmp_path / "shap.csv")
    json_path = explainer.to_json(result, tmp_path / "shap.json")
    assert csv_path.exists()
    assert json_path.exists()
    import json

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "base_value" in data
    assert "values" in data
