"""Tests for the MLOps toolkit (registry, gates, experiments, reports)."""

from __future__ import annotations

import json

import pytest

from mlops.config import MLOpsSettings, load_settings
from mlops.experiments import ExperimentTracker
from mlops.gates import (
    GateResult,
    all_passed,
    fairness_gate,
    metric_gate,
    regression_gate,
    write_gate_report,
)
from mlops.registry import ModelRegistry, RegistryError
from mlops.reports import write_benchmark_report, write_release_report


@pytest.fixture
def settings(tmp_path) -> MLOpsSettings:
    return MLOpsSettings(
        registry_dir=tmp_path / "registry",
        reports_dir=tmp_path / "reports",
        experiments_dir=tmp_path / "experiments",
        backups_dir=tmp_path / "backups",
    )


@pytest.fixture
def registry(settings) -> ModelRegistry:
    return ModelRegistry(settings)


@pytest.fixture
def fake_checkpoint(tmp_path):
    path = tmp_path / "yieldnet.pt"
    path.write_bytes(b"fake-weights")
    return path


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_register_and_get(registry, fake_checkpoint):
    record = registry.register("yieldnet", "1.0.0", checkpoint_path=fake_checkpoint,
                               metrics={"accuracy": 0.85})
    assert record.status == "draft"
    assert (record.dir / "manifest.json").exists()
    assert (record.dir / "yieldnet.pt").exists()
    assert registry.get("yieldnet", "1.0.0").manifest.metrics["accuracy"] == 0.85


def test_register_duplicate_raises(registry, fake_checkpoint):
    registry.register("yieldnet", "1.0.0", checkpoint_path=fake_checkpoint)
    with pytest.raises(RegistryError):
        registry.register("yieldnet", "1.0.0", checkpoint_path=fake_checkpoint)


def test_promote_demotes_incumbent_and_archives(registry, fake_checkpoint):
    registry.register("yieldnet", "1.0.0", checkpoint_path=fake_checkpoint,
                      metrics={"accuracy": 0.85})
    registry.register("yieldnet", "1.1.0", checkpoint_path=fake_checkpoint,
                      metrics={"accuracy": 0.87})
    registry.promote("yieldnet", "1.0.0", target="production")
    registry.promote("yieldnet", "1.1.0", target="production")

    active = registry.active("yieldnet")
    assert active.version == "1.1.0"
    statuses = {r.version: r.status for r in registry.list("yieldnet")}
    assert statuses == {"1.0.0": "staging", "1.1.0": "production"}


def test_promote_records_gates(registry, fake_checkpoint):
    registry.register("yieldnet", "1.0.0", checkpoint_path=fake_checkpoint)
    gates = [GateResult(gate="metrics", passed=True, message="ok")]
    record = registry.promote("yieldnet", "1.0.0", target="production", gates=gates)
    assert record.manifest.gates[0]["gate"] == "metrics"
    assert record.manifest.gates[0]["passed"] is True


def test_rollback(registry, fake_checkpoint):
    registry.register("yieldnet", "1.0.0", checkpoint_path=fake_checkpoint)
    registry.register("yieldnet", "1.1.0", checkpoint_path=fake_checkpoint)
    registry.promote("yieldnet", "1.0.0", target="production")
    registry.promote("yieldnet", "1.1.0", target="production")
    registry.rollback("yieldnet", "1.0.0")
    assert registry.active("yieldnet").version == "1.0.0"


def test_archive_rejects_production(registry, fake_checkpoint):
    registry.register("yieldnet", "1.0.0", checkpoint_path=fake_checkpoint)
    registry.promote("yieldnet", "1.0.0", target="production")
    with pytest.raises(RegistryError):
        registry.archive("yieldnet", "1.0.0")


def test_list_filters(registry, fake_checkpoint):
    registry.register("yieldnet", "1.0.0", checkpoint_path=fake_checkpoint)
    registry.register("soilnet", "1.0.0", checkpoint_path=fake_checkpoint)
    assert len(registry.list(name="yieldnet")) == 1
    assert len(registry.list(status="draft")) == 2


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def test_metric_gate_pass_and_fail(settings):
    ok = metric_gate({"accuracy": 0.9}, settings)
    assert ok.passed
    bad = metric_gate({"accuracy": 0.4}, settings)
    assert not bad.passed


def test_metric_gate_regression_against_incumbent(settings):
    result = metric_gate(
        {"accuracy": 0.86}, settings, incumbent_metrics={"accuracy": 0.88}
    )
    assert not result.passed


def test_regression_gate(settings):
    ok = regression_gate(10.8, 10.0, settings)
    assert ok.passed
    bad = regression_gate(12.0, 10.0, settings)
    assert not bad.passed
    assert bad.details["regression_pct"] == 20.0


def test_fairness_gate_skips_without_quality():
    result = fairness_gate([1, 0], [1, 0], {"region": ["a", "b"]})
    assert isinstance(result, GateResult)


def test_all_passed_and_report(settings):
    gates = [GateResult(gate="a", passed=True), GateResult(gate="b", passed=False)]
    assert not all_passed(gates)
    path = write_gate_report(gates, settings.reports_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["overall"] is False
    assert len(payload["gates"]) == 2


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #


def test_experiment_tracker_roundtrip(settings):
    tracker = ExperimentTracker(settings)
    tracker.log(model_name="yieldnet", config={"lr": 1e-3}, metrics={"accuracy": 0.87})
    tracker.log(model_name="yieldnet", config={"lr": 1e-4}, metrics={"accuracy": 0.88})
    runs = tracker.runs("yieldnet")
    assert len(runs) == 2
    assert tracker.best("yieldnet")["metrics"]["accuracy"] == 0.88
    assert tracker.export(settings.reports_dir / "runs.json").exists()


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def test_benchmark_and_release_report(registry, settings, fake_checkpoint):
    registry.register("yieldnet", "1.0.0", checkpoint_path=fake_checkpoint,
                      metrics={"accuracy": 0.85})
    registry.promote("yieldnet", "1.0.0", target="production")
    gates = [GateResult(gate="metrics", passed=True, message="ok")]
    bench = {"variants": [{"mode": "onnx", "mean_latency_ms": 4.2, "p95_latency_ms": 6.1,
                           "speedup": 3.1, "throughput_qps": 120.5}]}
    json_path = write_benchmark_report(bench, settings, model_name="yieldnet", version="1.0.0")
    assert json_path.exists()
    md_path = write_release_report(settings, model_name="yieldnet", version="1.0.0",
                                   target="production", gates=gates, registry=registry)
    assert md_path.exists()
    assert "yieldnet@1.0.0" in md_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_load_settings_env(monkeypatch):
    monkeypatch.setenv("MLOPS_MIN_ACCURACY", "0.5")
    settings = load_settings()
    assert settings.min_accuracy == 0.5
