"""Optimization wrapper tests — runtimes, batch, ONNX parity, benchmark."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ai.models import ModelConfig, ModelFactory
from quality.optimization import (
    BatchInferenceEngine,
    OptimizationBenchmark,
    OptimizedRuntime,
    run_autocast,
)


@pytest.fixture(scope="module")
def model():
    config = ModelConfig(
        tabular={"numeric_dim": 3, "categorical_cardinalities": [4, 2]},
        image_encoder={"backbone": "mobilenetv3_small_050", "input_size": 32},
        temporal={"d_model": 64, "depth": 2, "num_heads": 4, "ff_dim": 256,
                  "embedding_dim": 64, "max_len": 8},
        cross_attention={"num_heads": 4, "out_dim": 64},
        gated_fusion={"out_dim": 64, "hidden_dim": 64},
        shared_encoder={"d_model": 64, "depth": 2, "num_heads": 4, "ff_dim": 256,
                        "out_dim": 128},
        heads={"crop": {"num_classes": 3}, "yield_prediction": {}},
    )
    return ModelFactory.create(config).eval()


@pytest.fixture(scope="module")
def batch(model):
    return model.sample_batch(batch_size=2, seq_len=4)


def test_eager_runtime_matches_model(model, batch):
    runtime = OptimizedRuntime(model, mode="eager")
    with torch.no_grad():
        expected = model(batch).crop_logits
    probs = runtime.predict_proba(batch)
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs, torch.softmax(expected, dim=-1).numpy(), atol=1e-5)


def test_invalid_mode_rejected(model):
    with pytest.raises(ValueError):
        OptimizedRuntime(model, mode="quantize")


def test_autocast_runtime_preserves_prediction(model, batch):
    runtime = OptimizedRuntime(model, mode="autocast")
    eager = OptimizedRuntime(model, mode="eager").predict_proba(batch)
    fast = runtime.predict_proba(batch)
    assert fast.shape == eager.shape
    np.testing.assert_allclose(fast.argmax(axis=1), eager.argmax(axis=1), atol=0)


def test_onnx_runtime_parity(model, batch, tmp_path):
    engine_runtime = OptimizedRuntime(
        model,
        mode="onnx",
        onnx_path=str(tmp_path / "model.onnx"),
        sample_batch=batch,
    )
    outputs = engine_runtime.predict(batch)
    assert "crop_logits" in outputs
    assert outputs["crop_logits"].shape == (2, 3)
    errors = engine_runtime._onnx.assert_parity(batch, rtol=1e-2, atol=1e-2)
    assert set(errors) <= {"crop_logits", "yield_pred", "shared_representation"}
    assert max(errors.values()) < 1e-1


def test_batch_inference_consistent_with_single(model):
    engine = BatchInferenceEngine(model)
    single = OptimizedRuntime(model, mode="eager")
    samples = [model.sample_batch(batch_size=1) for _ in range(3)]
    batched = engine.predict(samples)
    assert len(batched) == 3
    for i, sample in enumerate(samples):
        expected = single.predict_proba(sample).argmax(axis=1)
        actual = np.asarray(batched[i]["crop_logits"]).argmax(axis=1)
        assert actual.tolist() == expected.tolist()


def test_benchmark_writes_report(model, tmp_path):
    bench = OptimizationBenchmark(model, batch_size=2, iterations=2, warmup=1,
                                  modes=("eager", "onnx"))
    paths = bench.write_report(tmp_path)
    assert paths["json"].exists() and paths["html"].exists()
    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert {v["name"] for v in report["variants"]} == {"eager", "onnx"}
    assert report["speedup"]["eager"] == pytest.approx(1.0, abs=0.01)
    assert report["variants"][0]["latency"]["mean_ms"] >= 0


def test_run_autocast_cpu_returns_usable_context():
    ctx = run_autocast(torch.device("cpu"))
    assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")
    with ctx:
        pass


def test_run_autocast_cuda_device():
    ctx = run_autocast(torch.device("cuda"))
    assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")


def test_compiled_runtime_preserves_prediction(model, batch):
    runtime = OptimizedRuntime(model, mode="compiled")
    eager = OptimizedRuntime(model, mode="eager").predict_proba(batch)
    fast = runtime.predict_proba(batch)
    assert fast.shape == eager.shape
    np.testing.assert_allclose(fast.argmax(axis=1), eager.argmax(axis=1), atol=0)


def test_runtime_inspectors(model):
    runtime = OptimizedRuntime(model, mode="eager")
    assert runtime.num_parameters() > 0
    assert runtime.device_name() == str(next(model.parameters()).device)


def test_predict_proba_raises_without_crop_head():
    class NoCropHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3, 1)

        def forward(self, batch):
            return {"yield_pred": self.linear(batch["tabular"])}

    runtime = OptimizedRuntime(NoCropHead(), mode="eager")
    with pytest.raises(ValueError):
        runtime.predict_proba({"tabular": torch.zeros(2, 3)})


def test_batch_inference_chunks_oversized_requests(model):
    engine = BatchInferenceEngine(model, max_batch=2)
    samples = [model.sample_batch(batch_size=1) for _ in range(5)]
    results = engine.predict(samples)
    assert len(results) == 5
    for result in results:
        assert np.asarray(result["crop_logits"]).shape[0] == 1
