"""Predictions module service — orchestrates inference + persistence + explanation."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger, PerformanceTimer
from app.models.prediction import ExplanationRecord, Prediction
from app.modules.explainability.service import ExplainabilityService
from app.modules.inference.service import InferenceEngine
from app.modules.predictions.repository import ExplanationRepository, PredictionRepository
from app.modules.predictions.schemas import MapPredictionRequest, PredictionRequest, PredictionResponse

logger = get_logger("predictions")


class PredictionService:
    """Builds, persists and explains predictions."""

    def __init__(
        self,
        engine: InferenceEngine,
        explainer: ExplainabilityService,
        session: AsyncSession,
    ) -> None:
        self._engine = engine
        self._explainer = explainer
        self._repository = PredictionRepository(session)
        self._explanation_repository = ExplanationRepository(session)

    async def predict(
        self, req: PredictionRequest, user_id: int | None = None
    ) -> PredictionResponse:
        with PerformanceTimer("predict.endpoint"):
            result = await self._engine.predict(
                req.lon, req.lat, year=req.year, season=req.season
            )
            prediction = await self._persist(req, result, user_id, source="point")
            response = self._to_response(prediction, result)

            if req.include_explanation:
                response.explanation_summary = await self._explain_and_store(
                    prediction, result, user_id
                )
            return response

    async def predict_map(
        self, req: MapPredictionRequest, user_id: int | None = None
    ) -> list[PredictionResponse]:
        responses = []
        for point in req.points:
            responses.append(await self.predict(point, user_id))
        return responses

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _persist(
        self,
        req: PredictionRequest,
        result: dict[str, Any],
        user_id: int | None,
        *,
        source: str,
    ) -> Prediction:
        observation = result.get("observation")
        location_name = ""
        if observation is not None:
            admin = getattr(getattr(observation, "location", None), "admin", None)
            location_name = str(getattr(admin, "village", "") or "")
        record = Prediction(
            user_id=user_id,
            location_lon=req.lon,
            location_lat=req.lat,
            location_name=location_name or None,
            crop=result["recommended_crop"],
            crop_probs=result.get("crop_probs", {}),
            yield_prediction=result.get("expected_yield"),
            confidence=float(result.get("confidence", 0.0)),
            model_version=result.get("model_version", ""),
            inference_time_ms=float(result.get("inference_time_ms", 0.0)),
            source=source,
        )
        await self._repository.add(record)
        await self._repository.commit()
        await self._repository.refresh(record)
        return record

    async def _explain_and_store(
        self, prediction: Prediction, result: dict[str, Any], user_id: int | None
    ) -> dict[str, Any]:
        observation = result.get("observation")
        sample = result.get("raw_sample")
        if observation is None and sample is None:
            return {"message": "no explainable input available"}
        summary = self._explainer.summarize(sample=sample, observation=observation)
        record = ExplanationRecord(prediction_id=prediction.id, summary=summary)
        await self._explanation_repository.add(record)
        await self._explanation_repository.commit()
        return summary

    @staticmethod
    def _to_response(prediction: Prediction, result: dict[str, Any]) -> PredictionResponse:
        return PredictionResponse(
            prediction_id=prediction.id,
            location={"lon": prediction.location_lon, "lat": prediction.location_lat,
                      "name": prediction.location_name},
            coordinates={"lon": prediction.location_lon, "lat": prediction.location_lat},
            recommended_crop=prediction.crop,
            expected_yield=prediction.yield_prediction,
            confidence=prediction.confidence,
            crop_probs=prediction.crop_probs,
            model_version=prediction.model_version,
            inference_time_ms=prediction.inference_time_ms,
            fallback=bool(result.get("fallback", False)),
        )
