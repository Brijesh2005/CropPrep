"""Explainability module service — integrates the Phase 7 MXAI framework."""

from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger, PerformanceTimer
from app.modules.explainability.schemas import ExplanationResponse

logger = get_logger("explainability")


class ExplainabilityService:
    """Wraps :class:`ai.explainability.Explainer` for the API layer."""

    def __init__(self, ai_explainer: Any, settings: Settings, stam: Any) -> None:
        self._ai = ai_explainer
        self._settings = settings
        self._stam = stam

    # ------------------------------------------------------------------ #
    # Full explanation
    # ------------------------------------------------------------------ #

    def explain_observation(self, observation: Any) -> ExplanationResponse:
        with PerformanceTimer("explain.observation"):
            explanation = self._ai.explain(observation)
        return ExplanationResponse(
            observation_id=explanation.observation_id,
            crop=explanation.crop,
            crop_probs=explanation.crop_probs,
            yield_prediction=explanation.yield_prediction,
            confidence=explanation.confidence,
            top_features=[[name, float(value)] for name, value in explanation.top_features],
            important_dates=explanation.temporal_ranking[:5],
            modality_gates=explanation.gates,
            reasoning=explanation.reasoning[:8],
            limitations=explanation.limitations[:8],
            raw=explanation.to_dict(),
        )

    def explain_location(self, lon: float, lat: float, *, year: int | None, season: str | None) -> ExplanationResponse:
        observation = self._stam.build_observation(
            lon, lat, year=year or 2020, season=season or "Kharif"
        )
        return self.explain_observation(observation)

    # ------------------------------------------------------------------ #
    # Summary for a stored prediction
    # ------------------------------------------------------------------ #

    def summarize(self, *, sample: Any = None, observation: Any = None) -> dict[str, Any]:
        """A compact explanation summary for storing with a prediction."""
        try:
            explanation = self._ai.explain(observation)
            return {
                "crop": explanation.crop,
                "top_features": [[name, float(value)] for name, value in explanation.top_features[:10]],
                "important_dates": explanation.temporal_ranking[:5],
                "modality_gates": explanation.gates,
                "confidence": {
                    k: v for k, v in explanation.confidence.items()
                    if isinstance(v, (int, float))
                },
                "reasoning": explanation.reasoning[:5],
                "limitations": explanation.limitations[:5],
            }
        except Exception as exc:
            logger.warning("explanation summary unavailable ({})", exc)
            return {"message": "explanation unavailable", "detail": str(exc)}
