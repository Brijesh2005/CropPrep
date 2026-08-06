"""Predictions module service — orchestrates inference + persistence.

REPLACES app/modules/predictions/service.py. The old version pulled
``observation`` / ``raw_sample`` off the result dict (STAM concepts). This
version reads the flat result dict produced by the release-package
``InferenceEngine`` (see app/modules/inference/service.py in this batch):
``location``, ``top3``, ``model_version``, ``dataset_version``,
``feature_names``/``feature_values``.

Explanation generation here is a lightweight placeholder (top-N scaled
feature magnitudes) rather than the full Explainability module — wiring the
real explainability pipeline against release-package artifacts is a
follow-up chunk, out of scope for "backend core: model loader + inference
engine + /predict, /model, /health".
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import PerformanceTimer, get_logger
from app.models.prediction import ExplanationRecord, Prediction
from app.modules.inference.service import InferenceEngine
from app.modules.predictions.repository import ExplanationRepository, PredictionRepository
from app.modules.predictions.schemas import (
    CropCandidate,
    MapPredictionRequest,
    PredictionRequest,
    PredictionResponse,
)

logger = get_logger("predictions")


class PredictionService:
    """Builds, persists and (lightly) explains predictions."""

    def __init__(self, engine: InferenceEngine, session: AsyncSession) -> None:
        self._engine = engine
        self._repository = PredictionRepository(session)
        self._explanation_repository = ExplanationRepository(session)

    async def predict(self, req: PredictionRequest, user_id: int | None = None) -> PredictionResponse:
        with PerformanceTimer("predict.endpoint"):
            result = await self._engine.predict(req.lon, req.lat)
            prediction = await self._persist(result, user_id, source="point")
            response = self._to_response(prediction, result)

            if req.include_explanation and not result.get("fallback"):
                response.feature_importance = self._explain(result)
                response.explanation_summary = {"top_features": response.feature_importance}
                await self._store_explanation(prediction, response.explanation_summary)

            return response

    async def predict_map(
        self, req: MapPredictionRequest, user_id: int | None = None
    ) -> list[PredictionResponse]:
        return [await self.predict(point, user_id) for point in req.points]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _persist(self, result: dict[str, Any], user_id: int | None, *, source: str) -> Prediction:
        location = result.get("location", {})
        record = Prediction(
            user_id=user_id,
            location_lon=location.get("lon", 0.0),
            location_lat=location.get("lat", 0.0),
            location_name=location.get("village"),
            district=location.get("district"),
            taluk=location.get("taluk"),
            season=location.get("season"),
            year=location.get("year"),
            crop=result["recommended_crop"],
            crop_probs=result.get("crop_probs", {}),
            top3=result.get("top3", []),
            yield_prediction=result.get("expected_yield"),
            confidence=float(result.get("confidence", 0.0)),
            model_version=result.get("model_version", ""),
            dataset_version=result.get("dataset_version", ""),
            inference_time_ms=float(result.get("inference_time_ms", 0.0)),
            source=source,
            fallback=bool(result.get("fallback", False)),
        )
        await self._repository.add(record)
        await self._repository.commit()
        await self._repository.refresh(record)
        return record

    async def _store_explanation(self, prediction: Prediction, summary: dict[str, Any]) -> None:
        record = ExplanationRecord(prediction_id=prediction.id, summary=summary)
        await self._explanation_repository.add(record)
        await self._explanation_repository.commit()

    @staticmethod
    def _explain(result: dict[str, Any], top_n: int = 5) -> dict[str, float]:
        """Rank features by |scaled value| as a cheap, model-agnostic proxy for
        importance. Placeholder pending the real Explainability module wiring."""
        values: dict[str, float] = result.get("feature_values", {})
        ranked = sorted(values.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return {name: value for name, value in ranked[:top_n]}

    @staticmethod
    def _to_response(prediction: Prediction, result: dict[str, Any]) -> PredictionResponse:
        return PredictionResponse(
            prediction_id=prediction.id,
            village=prediction.location_name or "",
            district=prediction.district or "",
            taluk=prediction.taluk,
            coordinates={"lon": prediction.location_lon, "lat": prediction.location_lat},
            season=prediction.season or "",
            year=prediction.year,
            recommended_crop=prediction.crop,
            expected_yield=prediction.yield_prediction,
            confidence=prediction.confidence,
            crop_probs=prediction.crop_probs,
            top3=[CropCandidate(**c) for c in (prediction.top3 or [])],
            model_version=prediction.model_version,
            dataset_version=prediction.dataset_version,
            inference_time_ms=prediction.inference_time_ms,
            fallback=prediction.fallback,
        )


__all__ = ["PredictionService"]
