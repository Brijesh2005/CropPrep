"""Background task functions.

These run in the event loop (or a dedicated worker process in deployment) and
handle periodic maintenance: dataset refresh, model warmup, prediction-log
pruning and cache cleanup.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

logger = get_logger("workers")


async def dataset_refresh(container: Any) -> None:
    """Regenerate the dataset metadata (used on startup / /dataset/reload)."""
    service = container.model.resolve("dataset_service")
    result = service.reload()
    logger.info("dataset refresh complete", message=result.get("message"))


async def model_warmup(container: Any) -> None:
    """Warm up the inference engine."""
    registry = container.model.resolve("model_registry")
    registry.warmup()
    logger.info("model warmup complete", version=registry.version)


async def prediction_logging(container: Any, prediction_id: int) -> None:
    """Log a persisted prediction (structured audit line)."""
    logger.bind(prediction_id=prediction_id).info("prediction persisted")


async def cleanup(container: Any) -> None:
    """Prune stale cache entries / old prediction logs."""
    cache = container.services.resolve("cache")
    if hasattr(cache, "cleanup") and callable(getattr(cache, "cleanup", None)):
        await cache.cleanup()
    logger.info("cleanup complete")
