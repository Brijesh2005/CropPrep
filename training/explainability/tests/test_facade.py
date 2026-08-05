"""End-to-end facade tests over the real STAM -> preprocessing chain."""

from __future__ import annotations

import pytest

from training.explainability import Explainer
from training.explainability.config import ExplainabilityConfig
from training.models import ModelFactory


@pytest.fixture
def explainer(preprocessor, observations, derived_model_config, stam_chain, tmp_path):
    config = ExplainabilityConfig(
        shap={"background_size": 2, "max_samples": 16},
        uncertainty={"mc_dropout_samples": 0},
        visualization={"directory": str(tmp_path / "figs")},
        export={"directory": str(tmp_path / "exports")},
    )
    return Explainer(
        ModelFactory.create(derived_model_config),
        preprocessor,
        config,
        observations=observations,
        extractor=stam_chain.get_patch,
    )


def test_explain_full_pipeline(explainer, observations):
    explanation = explainer.explain(observations[0])
    assert explanation.crop
    assert explanation.crop_probs
    assert explanation.yield_prediction is not None
    assert explanation.feature_importance
    assert explanation.gates
    assert explanation.reasoning


def test_explain_crop_and_yield(explainer, observations):
    crop_exp = explainer.explain_crop(observations[0])
    yield_exp = explainer.explain_yield(observations[0])
    assert crop_exp.crop
    assert yield_exp.crop_probs  # crop probs present for the report


def test_generate_report_farmer(explainer, observations):
    report = explainer.generate_report(observations[0], mode="farmer")
    assert report["recommended_crop"]
    assert "why" in report
    assert "limitations" in report


def test_visualize(explainer, observations):
    explanation = explainer.explain(observations[0])
    figures = explainer.visualize(explanation)
    assert figures  # at least feature_importance produced
    assert all(path.exists() for path in figures.values())


def test_export(explainer, observations):
    explanation = explainer.explain(observations[0])
    written = explainer.export(explanation, formats=["html", "json", "csv"])
    assert written["html"].exists()
    assert written["json"].exists()
    assert written["csv"].exists()
