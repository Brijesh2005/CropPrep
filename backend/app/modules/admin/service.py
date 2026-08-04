"""Admin module service — dashboard, statistics, retraining."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.prediction import Prediction
from app.models.user import User
from app.modules.admin.schemas import Dashboard, Statistics
from app.modules.inference.service import InferenceEngine

logger = get_logger("admin")


class AdminService:
    """Administrative views over the system state."""

    def __init__(self, session: AsyncSession, engine: InferenceEngine, registry: Any, dataset_service: Any) -> None:
        self._session = session
        self._engine = engine
        self._registry = registry
        self._dataset = dataset_service

    async def dashboard(self) -> Dashboard:
        prediction_count = await self._count(Prediction)
        users_count = await self._count(User)
        return Dashboard(
            model_ready=self._registry.is_ready(),
            model_version=self._registry.version,
            device=str(self._registry.device),
            prediction_count=prediction_count,
            users_count=users_count,
            dataset_ready=self._dataset is not None,
            queue_size=self._engine.status().get("queue_size", 0),
        )

    async def statistics(self) -> Statistics:
        result = await self._session.execute(
            select(
                func.count(Prediction.id),
                func.avg(Prediction.confidence),
                func.avg(Prediction.inference_time_ms),
            )
        )
        total, avg_conf, avg_time = result.one()
        crop_result = await self._session.execute(
            select(Prediction.crop, func.count(Prediction.id))
            .group_by(Prediction.crop)
            .order_by(func.count(Prediction.id).desc())
        )
        crop_distribution = {crop: int(count) for crop, count in crop_result.all()}
        return Statistics(
            total_predictions=int(total or 0),
            crop_distribution=crop_distribution,
            avg_confidence=round(float(avg_conf or 0.0), 4),
            avg_inference_time_ms=round(float(avg_time or 0.0), 3),
        )

    async def retrain(self) -> dict[str, Any]:
        """Trigger retraining as a background task (framework placeholder hook)."""
        logger.info("retraining requested")
        # A real deployment would enqueue a Phase 6 training job here.
        return {"message": "retraining job enqueued", "started": True}

    async def _count(self, model: Any) -> int:
        result = await self._session.execute(select(func.count(model.id)))
        return int(result.scalar_one() or 0)
