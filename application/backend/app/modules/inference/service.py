"""Inference engine — Location -> Result, using only the exported release package.

REPLACES ``app/modules/inference/service.py``. The old version called into
STAM / the training preprocessor / a live checkpoint. This version:

    Location -> LocationResolver -> Historical Context Lookup -> FeatureBuilder
    -> ModelInput -> CropFusion Model -> Crop Recommendation + Yield + Confidence

and never imports the Dataset Manager, STAM, or Kaggle/GeoTIFF code. The
public interface (``start``, ``stop``, ``predict``, ``status``, ``warmup``)
is unchanged from the previous engine, so ``app_container.py``'s wiring keeps
working with only the constructor arguments updated (see the app_container.py
file in this same batch).

Note: farmer-mode requests carry lon/lat only — no year/season — per the R6
spec; season and year are resolved by ``LocationResolver`` from the request
date, not accepted from the client.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import torch

from app.core.exceptions import InferenceError, PredictionError
from app.core.logging import PerformanceTimer, get_logger
from app.modules.inference.feature_builder import FeatureBuilder
from app.modules.inference.location_resolver import LocationNotServedError, LocationResolver
from app.services.release_model_registry import ModelRegistry
from app.services.cache import Cache

logger = get_logger("inference")

#: How many top crop candidates to return alongside the recommendation.
TOP_K = 3


class InferenceEngine:
    """Runs the CropFusion inference pipeline for a location, backed by the
    release package only."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        config: Any,
        cache: Cache | None = None,
    ) -> None:
        self.registry = registry
        self.config = config
        self.cache = cache
        self._resolver: LocationResolver | None = None
        self._feature_builder: FeatureBuilder | None = None
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
        if self.registry.is_ready() and self.registry.package is not None:
            self._resolver = LocationResolver(self.registry.package)
            self._feature_builder = FeatureBuilder(self.registry.package)
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

    async def predict(self, lon: float, lat: float) -> dict[str, Any]:
        """Predict for a location (cached, queued, fallback-aware). Location only —
        season/year/context are resolved internally."""
        cache_key = self._cache_key(lon, lat)
        if self.cache is not None and self.config.enable_cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached

        result = await self._run_in_queue(lon, lat)

        if self.cache is not None and self.config.enable_cache and not result.get("fallback"):
            await self.cache.set(cache_key, result, ttl=self.config.cache_ttl_seconds)
        return result

    async def _run_in_queue(self, lon: float, lat: float) -> dict[str, Any]:
        if not self.registry.is_ready() or self._resolver is None or self._feature_builder is None:
            if self.config.enable_fallback:
                return self.registry.fallback_prediction(lon, lat)
            raise InferenceError("model is not ready")

        semaphore = self._semaphore or asyncio.Semaphore(1)

        async def _task() -> dict[str, Any]:
            async with semaphore:
                return await asyncio.to_thread(self._run_sync, lon, lat)

        if self._queue is not None:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            await self._queue.put((_task, future))
            return await future
        return await _task()

    # ------------------------------------------------------------------ #
    # Synchronous pipeline
    # ------------------------------------------------------------------ #

    def _run_sync(self, lon: float, lat: float) -> dict[str, Any]:
        timer_start = time.perf_counter()
        try:
            location = self._resolver.resolve(lon, lat)
            model_input = self._feature_builder.build(location)

            model = self.registry.model
            with torch.no_grad():
                output = model(model_input.tensor)

            crop_logits, yield_pred = self._unpack_output(output)
            probs = torch.softmax(crop_logits.float(), dim=-1)[0]
            top_indices = torch.topk(probs, k=min(TOP_K, probs.shape[0])).indices.tolist()

            classes = getattr(self.registry.package.label_encoder, "classes_", None)
            crop_probs = {self._crop_name(classes, i): float(probs[i]) for i in range(probs.shape[0])}
            top3 = [
                {"crop": self._crop_name(classes, i), "probability": float(probs[i])}
                for i in top_indices
            ]
            recommended_crop = top3[0]["crop"]
            confidence = top3[0]["probability"]

            expected_yield = float(yield_pred.item()) if yield_pred is not None else None

            inference_time_ms = (time.perf_counter() - timer_start) * 1000

            return {
                "recommended_crop": recommended_crop,
                "crop_probs": crop_probs,
                "top3": top3,
                "expected_yield": expected_yield,
                "confidence": confidence,
                "model_version": self.registry.version,
                "dataset_version": self.registry.dataset_version,
                "inference_time_ms": round(inference_time_ms, 3),
                "fallback": False,
                "location": {
                    "village": location.village,
                    "district": location.district,
                    "taluk": location.taluk,
                    "lon": location.lon,
                    "lat": location.lat,
                    "season": location.season,
                    "year": location.year,
                },
                "feature_names": model_input.feature_names,
                "feature_values": model_input.raw_values,
            }
        except LocationNotServedError as exc:
            raise PredictionError(
                "location is outside the exported package's coverage area", detail=str(exc)
            ) from exc
        except InferenceError:
            raise
        except Exception as exc:
            raise PredictionError("inference pipeline failed", detail=str(exc)) from exc

    @staticmethod
    def _unpack_output(output: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Supports either a (crop_logits, yield_pred) tuple, a namedtuple/dataclass
        with those attributes, or a dict — whatever shape the exporter produced."""
        if isinstance(output, tuple):
            crop_logits = output[0]
            yield_pred = output[1] if len(output) > 1 else None
            return crop_logits, yield_pred
        if isinstance(output, dict):
            return output["crop_logits"], output.get("yield_pred")
        crop_logits = getattr(output, "crop_logits")
        yield_pred = getattr(output, "yield_pred", None)
        return crop_logits, yield_pred

    @staticmethod
    def _crop_name(classes: Any, index: int) -> str:
        if classes is not None and index < len(classes):
            return str(classes[index])
        return f"crop_{index}"

    @staticmethod
    def _cache_key(lon: float, lat: float) -> str:
        return f"pred:{lon:.5f}:{lat:.5f}"

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.registry.is_ready(),
            "model_version": self.registry.version,
            "dataset_version": self.registry.dataset_version,
            "queue_size": self._queue.qsize() if self._queue else 0,
            "cache_enabled": self.cache is not None and self.config.enable_cache,
            "device": self.registry.device,
        }

    def warmup(self) -> None:
        if self.config.warmup:
            self.registry.warmup()


__all__ = ["InferenceEngine"]
