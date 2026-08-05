"""Inference engine — STAM -> preprocessing -> model -> prediction.

Implements model loading, warmup, an async inference queue, prediction caching,
model versioning and a heuristic fallback. The pipeline never bypasses STAM or
preprocessing.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

import torch

from training.explainability.utils import single_sample_batch
from training.models.cropfusion import CropFusionOutput

from app.core.config import InferenceSettings
from app.core.exceptions import InferenceError, PredictionError
from app.core.logging import get_logger, PerformanceTimer
from app.modules.inference.schemas import PredictionResponse
from app.services.cache import Cache
from app.services.model_registry import ModelRegistry

logger = get_logger("inference")


class InferenceEngine:
    """Runs the CropFusion inference pipeline for a location."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        stam: Any,
        preprocessor: Any,
        config: InferenceSettings,
        cache: Cache | None = None,
    ) -> None:
        self.registry = registry
        self.stam = stam
        self.preprocessor = preprocessor
        self.config = config
        self.cache = cache
        self._semaphore: asyncio.Semaphore | None = None
        self._queue: asyncio.Queue | None = None
        self._workers: list[asyncio.Task] = []
        self._running = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the bounded inference queue workers (called at startup)."""
        if self._running:
            return
        self._running = True
        self._queue = asyncio.Queue(maxsize=self.config.queue_size)
        self._semaphore = asyncio.Semaphore(self.config.max_workers)
        for i in range(self.config.max_workers):
            self._workers.append(asyncio.create_task(self._worker(i)))

    async def stop(self) -> None:
        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def _worker(self, index: int) -> None:
        while self._running:
            fn, future = await self._queue.get()
            try:
                result = await fn()
                if not future.done():
                    future.set_result(result)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
                logger.warning("inference worker error ({}): {}", index, exc)
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ #
    # Public predict API
    # ------------------------------------------------------------------ #

    async def predict(
        self,
        lon: float,
        lat: float,
        *,
        year: int | None = None,
        season: str | None = None,
    ) -> dict[str, Any]:
        """Predict for a location (cached, queued, fallback-aware)."""
        if self.cache is not None and self.config.enable_cache:
            cached = await self.cache.get(self._cache_key(lon, lat, year, season))
            if cached is not None:
                return cached

        result = await self._run_in_queue(lon, lat, year=year, season=season)

        if self.cache is not None and self.config.enable_cache and not result.get("fallback"):
            await self.cache.set(
                self._cache_key(lon, lat, year, season), result,
                ttl=self.config.cache_ttl_seconds,
            )
        return result

    async def _run_in_queue(
        self, lon: float, lat: float, *, year: int | None, season: str | None
    ) -> dict[str, Any]:
        if not self.registry.is_ready():
            if self.config.enable_fallback:
                return self.registry.fallback_prediction(lon, lat)
            raise InferenceError("model is not ready")

        semaphore = self._semaphore
        if semaphore is None:
            semaphore = asyncio.Semaphore(1)

        async def _task() -> dict[str, Any]:
            async with semaphore:
                return await asyncio.to_thread(
                    self._run_sync, lon, lat, year=year, season=season
                )

        if self._queue is not None:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            await self._queue.put((_task, future))
            return await future
        return await _task()

    # ------------------------------------------------------------------ #
    # Synchronous pipeline (never bypasses STAM / preprocessing)
    # ------------------------------------------------------------------ #

    def _run_sync(
        self, lon: float, lat: float, *, year: int | None, season: str | None
    ) -> dict[str, Any]:
        timer = PerformanceTimer("inference.run", lon=lon, lat=lat)
        try:
            observation = self.stam.build_observation(
                lon, lat, year=year or 2020, season=season or "Kharif"
            )
            sample = self.preprocessor.transform(
                observation, extractor=self.stam.get_patch
            )
            batch = single_sample_batch(sample, self.registry.device)
            model = self.registry.model
            model.eval()
            with torch.no_grad():
                out: CropFusionOutput = model(batch)

            crop_logits = out.crop_logits
            yield_pred = out.yield_pred
            probs = torch.softmax(crop_logits.float(), dim=-1)[0]
            crop_class = int(probs.argmax().item())
            crop_name = self._crop_name(crop_class)
            crop_probs = {
                self._crop_name(i): float(p) for i, p in enumerate(probs.tolist())
            }
            predicted_yield = (
                float(yield_pred[0, 0].item()) if yield_pred is not None else None
            )
            if predicted_yield is not None:
                predicted_yield = self._inverse_scale_yield(predicted_yield)
            elapsed_ms = timer.stop() * 1000
            return {
                "recommended_crop": crop_name,
                "crop_probs": crop_probs,
                "expected_yield": predicted_yield,
                "confidence": float(probs[crop_class].item()),
                "model_version": self.registry.version,
                "inference_time_ms": round(elapsed_ms, 3),
                "fallback": False,
                "raw_sample": sample,
                "observation": observation,
            }
        except InferenceError:
            raise
        except Exception as exc:
            raise PredictionError("inference pipeline failed", detail=str(exc)) from exc

    def _inverse_scale_yield(self, value: float) -> float:
        scaler = getattr(getattr(self.preprocessor, "label", None), "yield_scaler", None)
        if scaler is not None and hasattr(scaler, "inverse_transform"):
            import numpy as np

            arr = np.asarray([[float(value)]], dtype="float64")
            return float(np.asarray(scaler.inverse_transform(arr))[0, 0])
        return value

    def _crop_name(self, index: int) -> str:
        classes = getattr(self.preprocessor.label, "crop_encoder", None)
        names = getattr(classes, "classes_", None)
        if names is not None and index < len(names):
            return str(names[index])
        return f"crop_{index}"

    @staticmethod
    def _cache_key(lon: float, lat: float, year: int | None, season: str | None) -> str:
        return f"pred:{lon:.5f}:{lat:.5f}:{year}:{season}"

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.registry.is_ready(),
            "model_version": self.registry.version,
            "queue_size": self._queue.qsize() if self._queue else 0,
            "cache_enabled": self.cache is not None and self.config.enable_cache,
            "device": str(self.registry.device),
        }

    def warmup(self) -> None:
        if self.config.warmup:
            self.registry.warmup()
