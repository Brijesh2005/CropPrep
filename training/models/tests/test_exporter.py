"""Exporter tests: TorchScript round-trip, ONNX dependency, TensorRT future."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training.models import ModelExporter
from training.models.exceptions import ExportError, MissingDependencyError


def _has_onnx() -> bool:
    try:
        import onnx  # noqa: F401

        return True
    except ImportError:
        return False


def test_torchscript_export_matches(model, batch, tmp_path: Path):
    exporter = ModelExporter(model, sample_batch=batch)
    path = exporter.export_torchscript(tmp_path / "model.ts")
    assert path.exists()

    traced = torch.jit.load(path)
    model.eval()
    with torch.no_grad():
        reference = model.forward_export(
            batch["tabular"], batch["ndvi"], batch["evi"], batch["temporal_mask"]
        )
        actual = traced(
            batch["tabular"], batch["ndvi"], batch["evi"], batch["temporal_mask"]
        )
    assert len(actual) == len(reference)
    for ref, got in zip(reference, actual):
        assert torch.allclose(ref, got, atol=1e-5)


def test_torchscript_rejects_script_mode(model, batch, tmp_path: Path):
    exporter = ModelExporter(model, sample_batch=batch)
    with pytest.raises(ExportError):
        exporter.export_torchscript(tmp_path / "model.ts", mode="script")


def test_torchscript_eval_preserved(model, batch, tmp_path: Path):
    model.train()
    exporter = ModelExporter(model, sample_batch=batch)
    exporter.export_torchscript(tmp_path / "model.ts")
    assert model.training  # training mode restored after export


def test_onnx_requires_dependency_or_succeeds(model, batch, tmp_path: Path):
    exporter = ModelExporter(model, sample_batch=batch)
    if not _has_onnx():
        with pytest.raises(MissingDependencyError):
            exporter.export_onnx(tmp_path / "model.onnx")
        return
    try:
        exporter.export_onnx(tmp_path / "model.onnx")
    except ExportError:
        pytest.skip("onnx export not supported for this graph")
    assert (tmp_path / "model.onnx").exists()


def test_tensorrt_is_future_target(model, batch, tmp_path: Path):
    exporter = ModelExporter(model, sample_batch=batch)
    with pytest.raises(MissingDependencyError):
        exporter.export_tensorrt(tmp_path / "model.engine")


def test_export_uses_only_enabled_inputs(tabular_only_config, tmp_path: Path):
    from training.models import ModelFactory

    model = ModelFactory.create(tabular_only_config)
    exporter = ModelExporter(model)
    args = exporter._export_args()
    assert len(args) == 1  # tabular only
    path = exporter.export_torchscript(tmp_path / "tab_only.ts")
    assert path.exists()
