"""Inference engine tests (with a stubbed model registry)."""

from __future__ import annotations

import asyncio

from app.core.config import InferenceSettings, load_settings
from app.modules.inference.service import InferenceEngine


class _StubRegistry:
    device = "cpu"
    version = "stub-1.0"

    def __init__(self) -> None:
        self.ready_flag = True

    def is_ready(self) -> bool:
        return self.ready_flag

    def version_info(self) -> dict:
        return {"version": self.version, "ready": self.ready_flag}

    def warmup(self) -> None:
        pass

    def fallback_prediction(self, lon: float, lat: float) -> dict:
        return {"recommended_crop": "Fallback", "crop_probs": {}, "expected_yield": 0.0,
                "confidence": 0.0, "model_version": "fallback", "fallback": True}


class _StubSTAM:
    def build_observation(self, lon, lat, *, year, season):
        class Obs:
            def __init__(self):
                self.sequence = type("S", (), {"pairs": []})()
                self.location = type("L", (), {"admin": None})()
        return Obs()

    def get_patch(self, *args, **kwargs):
        raise NotImplementedError


class _StubPreprocessor:
    label = type("L", (), {"yield_scaler": None, "crop_encoder": type(
        "E", (), {"classes_": ["Paddy", "Wheat"]})()})()

    def transform(self, observation, *, extractor=None):
        import torch

        return {
            "tabular": torch.randn(5),
            "ndvi": torch.zeros(4, 1, 32, 32),
            "evi": torch.zeros(4, 1, 32, 32),
            "temporal_mask": torch.ones(4),
        }


class _StubModel:
    def sample_batch(self, **kwargs):
        import torch

        return {"tabular": torch.randn(2, 5)}

    def eval(self):
        pass

    def to(self, device):
        return self

    def forward(self, batch):
        import torch

        return type("Out", (), {
            "crop_logits": torch.tensor([[0.8, 0.2, 0.1]]),
            "yield_pred": torch.tensor([[0.5]]),
        })()

    def __call__(self, batch):
        return self.forward(batch)


def _build_engine(**kwargs):
    registry = _StubRegistry()
    engine = InferenceEngine(
        registry,
        stam=_StubSTAM(),
        preprocessor=_StubPreprocessor(),
        config=InferenceSettings(**kwargs),
    )
    registry.model = _StubModel()
    return engine


def test_predict_point():
    engine = _build_engine()
    result = asyncio.run(engine.predict(74.8, 13.1))
    assert result["recommended_crop"] == "Paddy"
    assert result["confidence"] > 0
    assert result["model_version"] == "stub-1.0"
    assert result["fallback"] is False


def test_predict_caching():
    engine = _build_engine(enable_cache=True, cache_ttl_seconds=100)
    first = asyncio.run(engine.predict(74.8, 13.1))
    second = asyncio.run(engine.predict(74.8, 13.1))
    assert first["recommended_crop"] == second["recommended_crop"]


def test_fallback_when_not_ready():
    engine = _build_engine(enable_fallback=True)
    engine.registry.ready_flag = False
    result = asyncio.run(engine.predict(74.8, 13.1))
    assert result["fallback"] is True


def test_queued_predict():
    engine = _build_engine()
    async def _run():
        engine.start()
        result = await engine.predict(74.8, 13.1)
        await engine.stop()
        return result

    result = asyncio.run(_run())
    assert result["recommended_crop"] == "Paddy"


def test_status():
    engine = _build_engine()
    status = engine.status()
    assert status["ready"] is True
    assert status["model_version"] == "stub-1.0"
