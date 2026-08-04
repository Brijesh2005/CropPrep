"""Report-generator tests: farmer/research reports, historical comparison."""

from __future__ import annotations

import numpy as np

from ai.explainability import Explanation, ReportGenerator
from ai.explainability.config import ReportConfig


def _explanation() -> Explanation:
    return Explanation(
        observation_id="obs_1",
        crop="Paddy",
        crop_probs={"Paddy": 0.8, "Wheat": 0.2},
        yield_prediction=6.12,
        confidence={"crop_conf": 0.8, "crop_entropy": 0.5},
        feature_importance={"rainfall_mm": 0.4, "soil_moisture": -0.2},
        shap_values=np.asarray([0.4, -0.2]),
        shap_base_value=0.0,
        temporal_importance={"2020-06-01": 0.5, "2020-07-01": 0.3},
        temporal_ranking=["2020-06-01", "2020-07-01"],
        counterfactuals=[{"label": "rainfall +10%", "crop_changed": False}],
        gates={"image_gate": 0.5, "tabular_gate": 0.5},
        reasoning=["Recommended crop: Paddy"],
    )


def test_farmer_report():
    generator = ReportGenerator(ReportConfig())
    report = generator.farmer_report(_explanation())
    assert report["recommended_crop"] == "Paddy"
    assert report["confidence_percent"] == 80.0
    assert report["expected_yield_t_per_ha"] == 6.12
    assert report["top_factors"]  # rainfall_mm first
    assert report["important_dates"] == ["2020-06-01", "2020-07-01"]
    assert report["limitations"]


def test_research_report():
    generator = ReportGenerator(ReportConfig())
    report = generator.research_report(_explanation())
    assert report["prediction"]["crop"] == "Paddy"
    assert "feature_importance" in report
    assert "gates" in report
    assert "counterfactuals" in report


def test_historical_comparison():
    class Obs:
        def __init__(self, crop, yield_value):
            self.crop = crop
            self.yield_value = yield_value

    observations = [
        Obs("Paddy", 5.0),
        Obs("Paddy", 7.0),
        Obs("Wheat", 3.0),
    ]
    generator = ReportGenerator(ReportConfig(), observations=observations)
    explanation = _explanation()
    historical = generator.historical_comparison(explanation, predicted_yield=6.0)
    assert historical is not None
    assert historical["crop"] == "Paddy"
    assert historical["historical_mean_yield"] == 6.0  # mean(5,7)
    assert historical["prediction_vs_historical"] == 0.0


def test_reasoning_direction():
    generator = ReportGenerator(ReportConfig())
    explanation = _explanation()
    reasoning = generator.build_reasoning(explanation)
    assert any("rainfall" in r.lower() for r in reasoning)


def test_to_dict_json_safe():
    data = _explanation().to_dict()
    assert isinstance(data["shap_base_value"], float) or data["shap_base_value"] is None
