"""Benchmark tests: inference latency, throughput and model size."""

from __future__ import annotations

from training.training import Benchmark


def test_benchmark_inference(tabular_model):
    benchmark = Benchmark(
        tabular_model,
        batch_size=4,
        iterations=5,
        warmup_iterations=2,
    )
    sample_batch = tabular_model.sample_batch(batch_size=4)
    latency, memory, throughput = benchmark.evaluator.benchmark(
        sample_batch, iterations=5, warmup_iterations=2
    )
    assert latency["mean_ms"] >= 0.0
    assert latency["p50_ms"] >= 0.0
    assert throughput["samples_per_second"] > 0.0
    assert memory["parameters"] > 0
    assert memory["model_size_mb"] > 0.0


def test_benchmark_run_report(tabular_model, fake_loader):
    benchmark = Benchmark(tabular_model, batch_size=4, iterations=3, warmup_iterations=1)
    report = benchmark.run(
        train_loader=fake_loader,
        measure_training=True,
        measure_inference=True,
        sample_batch=tabular_model.sample_batch(batch_size=4),
    )
    assert "inference" in report.to_dict()
    assert report.to_dict()["inference"]["mean_ms"] >= 0.0
    assert report.to_dict()["training"]["samples_per_second"] > 0.0
