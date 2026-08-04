"""Unified explanation report.

:class:`Explanation` is the single result object of the explainability
framework — recommended crop, expected yield, confidence, feature importance,
image heatmaps, temporal importance, cross-modal contributions, counterfactuals
and reasoning.

:class:`ReportGenerator` turns an :class:`Explanation` into farmer-friendly
plain-language reasoning and a detailed research report, plus an automatic
historical comparison against the training observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .config import ReportConfig

#: Feature-name -> plain-language statement template.
_FEATURE_PHRASING = {
    "rainfall": "The rainfall pattern was {direction} influential for this prediction",
    "rainfall_mm": "The rainfall pattern was {direction} influential for this prediction",
    "precipitation": "Precipitation was {direction} influential for this prediction",
    "soil_moisture": "Soil moisture matched {direction} conditions",
    "soil": "The soil characteristics were {direction} influential",
    "temperature": "Temperature was {direction} influential",
    "ndvi": "Seasonal vegetation health was {direction} influential",
    "evi": "Vegetation greenness was {direction} influential",
    "humidity": "Humidity was {direction} influential",
}

#: Image-region phrasing for the farmer report.
_IMAGE_PHRASING = {
    "ndvi": "Seasonal vegetation remained healthy",
    "evi": "Vegetation greenness was strong",
}


@dataclass
class Explanation:
    """The complete explanation for one prediction."""

    observation_id: str = ""
    crop: str = ""
    crop_probs: dict[str, float] = field(default_factory=dict)
    yield_prediction: float | None = None
    confidence: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    shap_values: np.ndarray | None = None
    shap_base_value: float | None = None
    image_heatmaps: dict[str, Any] = field(default_factory=dict)
    image_overlays: dict[str, Any] = field(default_factory=dict)
    temporal_importance: dict[str, float] = field(default_factory=dict)
    temporal_ranking: list[str] = field(default_factory=list)
    cross_modal: dict[str, Any] = field(default_factory=dict)
    counterfactuals: list[dict[str, Any]] = field(default_factory=list)
    gates: dict[str, float] = field(default_factory=dict)
    integrated_gradients: dict[str, Any] = field(default_factory=dict)
    reasoning: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    historical: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "crop": self.crop,
            "crop_probs": self.crop_probs,
            "yield_prediction": self.yield_prediction,
            "confidence": self.confidence,
            "feature_importance": self.feature_importance,
            "shap_base_value": self.shap_base_value,
            "image_heatmaps": {k: _numpy_dict(v) for k, v in self.image_heatmaps.items()},
            "temporal_importance": self.temporal_importance,
            "temporal_ranking": self.temporal_ranking,
            "cross_modal": _numpy_dict(self.cross_modal),
            "counterfactuals": self.counterfactuals,
            "gates": self.gates,
            "integrated_gradients": _numpy_dict(self.integrated_gradients),
            "reasoning": self.reasoning,
            "limitations": self.limitations,
            "historical": self.historical,
        }

    @property
    def top_features(self) -> list[tuple[str, float]]:
        """Top features by |importance|, descending."""
        return sorted(
            self.feature_importance.items(), key=lambda kv: abs(kv[1]), reverse=True
        )


class ReportGenerator:
    """Builds farmer / research reports from an :class:`Explanation`."""

    def __init__(
        self,
        config: ReportConfig | None = None,
        observations: Sequence[Any] | None = None,
        crop_classes: Sequence[str] | None = None,
    ) -> None:
        self.config = config or ReportConfig()
        self.observations = list(observations or [])
        self.crop_classes = list(crop_classes or [])

    # ------------------------------------------------------------------ #
    # Reasoning
    # ------------------------------------------------------------------ #

    def build_reasoning(self, explanation: Explanation) -> list[str]:
        """Plain-language reasoning statements for a farmer / researcher."""
        statements: list[str] = []
        if explanation.crop:
            statements.append(f"Recommended crop: {explanation.crop}")

        # Feature-driven statements.
        for name, value in explanation.feature_importance.items():
            direction = "positively" if value > 0 else "negatively"
            template = None
            for key, phrase in _FEATURE_PHRASING.items():
                if key.lower() in name.lower():
                    template = phrase
                    break
            if template:
                statements.append(template.format(direction=direction))
            elif abs(value) > 0.05 * max(
                [abs(v) for v in explanation.feature_importance.values()] or [1.0]
            ):
                statements.append(
                    f"The field factor '{name}' was {direction} influential "
                    f"(SHAP {value:+.3f})"
                )

        # Image-driven statements.
        for index in ("ndvi", "evi"):
            result = explanation.image_heatmaps.get(index)
            if result and result.get("heatmaps") is not None:
                phrase = _IMAGE_PHRASING.get(index, "The imagery supported the prediction")
                statements.append(phrase)

        # Temporal statements.
        if explanation.temporal_ranking:
            statements.append(
                f"The most influential observation was {explanation.temporal_ranking[0]}"
            )

        # Historical comparison.
        if explanation.historical:
            hist = explanation.historical
            if hist.get("prediction_vs_historical") is not None:
                delta = hist["prediction_vs_historical"]
                if delta > 0:
                    statements.append(
                        "The predicted yield is higher than the historical average "
                        "for this crop in this region"
                    )
                else:
                    statements.append(
                        "The predicted yield is below the historical average for "
                        "this crop in this region"
                    )
        return statements

    def build_limitations(self, explanation: Explanation) -> list[str]:
        """Honest limitations of the explanation."""
        limitations = [
            "SHAP values reflect the model's learned associations, not necessarily "
            "causal relationships",
        ]
        if not explanation.image_heatmaps:
            limitations.append("Image (NDVI/EVI) evidence is unavailable for this sample")
        if not explanation.counterfactuals:
            limitations.append("Counterfactual ('what-if') evidence is unavailable")
        if not explanation.temporal_importance:
            limitations.append("Temporal importance is unavailable (no image sequence)")
        if explanation.confidence.get("crop_conf", 0) < 0.6:
            limitations.append(
                "The model confidence is low; treat the recommendation with caution"
            )
        return limitations

    # ------------------------------------------------------------------ #
    # Historical comparison
    # ------------------------------------------------------------------ #

    def historical_comparison(
        self, explanation: Explanation, predicted_yield: float | None = None
    ) -> dict[str, Any] | None:
        """Compare the prediction against historical observations.

        Uses observations matching the recommended crop; falls back to all
        observations when no crop matches.
        """
        if not self.observations:
            return None
        crop = explanation.crop
        matched = [
            obs for obs in self.observations
            if getattr(obs, "crop", None) == crop and getattr(obs, "yield_value", None) is not None
        ]
        if not matched:
            matched = [
                obs for obs in self.observations
                if getattr(obs, "yield_value", None) is not None
            ]
        if not matched:
            return None
        yields = np.asarray([float(obs.yield_value) for obs in matched], dtype="float64")
        historical_mean = float(yields.mean())
        result: dict[str, Any] = {
            "samples": int(len(matched)),
            "historical_mean_yield": historical_mean,
            "historical_std": float(yields.std()),
            "crop": crop,
        }
        predicted = predicted_yield if predicted_yield is not None else explanation.yield_prediction
        if predicted is not None:
            result["predicted_yield"] = float(predicted)
            result["prediction_vs_historical"] = float(predicted - historical_mean)
        return result

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #

    def farmer_report(self, explanation: Explanation) -> dict[str, Any]:
        """A concise, plain-language report for farmers / non-specialists."""
        reasoning = explanation.reasoning or self.build_reasoning(explanation)
        return {
            "recommended_crop": explanation.crop,
            "expected_yield_t_per_ha": explanation.yield_prediction,
            "confidence_percent": round(
                float(explanation.confidence.get("crop_conf", 0) or 0) * 100, 1
            ),
            "why": reasoning,
            "top_factors": [
                name for name, _ in explanation.top_features[: self.config.top_k_features]
            ],
            "important_dates": explanation.temporal_ranking[:5],
            "historical": explanation.historical,
            "limitations": explanation.limitations or self.build_limitations(explanation),
        }

    def research_report(self, explanation: Explanation) -> dict[str, Any]:
        """A detailed report for researchers / developers."""
        return {
            "observation_id": explanation.observation_id,
            "prediction": {
                "crop": explanation.crop,
                "crop_probs": explanation.crop_probs,
                "yield_prediction": explanation.yield_prediction,
            },
            "confidence": explanation.confidence,
            "feature_importance": explanation.feature_importance,
            "shap_base_value": explanation.shap_base_value,
            "integrated_gradients": explanation.integrated_gradients,
            "gates": explanation.gates,
            "temporal_importance": explanation.temporal_importance,
            "temporal_ranking": explanation.temporal_ranking,
            "image_heatmaps": {k: _numpy_dict(v) for k, v in explanation.image_heatmaps.items()},
            "cross_modal": _numpy_dict(explanation.cross_modal),
            "counterfactuals": explanation.counterfactuals,
            "historical": explanation.historical,
            "reasoning": explanation.reasoning,
            "limitations": explanation.limitations,
        }


def _numpy_dict(value: Any) -> Any:
    """Recursively convert numpy arrays / tensors to JSON-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _numpy_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numpy_dict(v) for v in value]
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    return value
