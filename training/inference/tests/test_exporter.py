"""Model exporter tests (Phase R5)."""

from __future__ import annotations

import json

import pytest
import torch

from training.inference.exceptions import ExportError
from training.inference.exporter import (
    BUNDLE_FORMAT_FILES,
    ModelExporter,
    load_pytorch_model,
)


def test_bundle_format_files():
    assert BUNDLE_FORMAT_FILES == {
        "pytorch": "cropfusion.pt",
        "torchscript": "cropfusion.torchscript.pt",
        "onnx": "cropfusion.onnx",
    }


def test_export_pytorch_roundtrip(model, tmp_path):
    exporter = ModelExporter(model)
    path = exporter.export_pytorch(tmp_path / "cropfusion.pt")
    assert path.exists()

    restored, metadata = load_pytorch_model(path)
    assert metadata["model_version"] == "1.0.0"
    assert metadata["parameter_count"] == sum(
        p.numel() for p in model.parameters()
    )
    for (a, b) in zip(
        model.state_dict().values(), restored.state_dict().values()
    ):
        assert torch.equal(a, b)


def test_export_pytorch_payload_structure(model, tmp_path):
    path = ModelExporter(model).export_pytorch(tmp_path / "cropfusion.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format"] == "cropfusion-pytorch"
    assert payload["format_version"] == 1
    assert set(payload) == {"format", "format_version", "config", "state_dict", "metadata"}


def test_load_rejects_non_export(tmp_path):
    path = tmp_path / "junk.pt"
    torch.save({"not": "an export"}, path)
    with pytest.raises(ExportError):
        load_pytorch_model(path)


def test_export_bundle_sidecars(model, tmp_path):
    from training.inference.config import InferenceConfig

    config = InferenceConfig(
        exporter={"formats": ["pytorch"]},
    )
    paths = ModelExporter(model).export_bundle(
        tmp_path,
        config=config,
        metrics={"crop": {"accuracy": 0.9}},
    )
    assert "pytorch" in paths
    assert "model_config" in paths
    assert "metrics" in paths
    assert "metadata" in paths
    assert "checksums" in paths

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["formats"] == ["pytorch"]
    assert metadata["model_version"] == "1.0.0"
    checksums = json.loads(paths["checksums"].read_text(encoding="utf-8"))
    assert "cropfusion.pt" in checksums


def test_export_bundle_onnx(model, tmp_path):
    from training.inference.config import InferenceConfig

    config = InferenceConfig(exporter={"formats": ["pytorch", "onnx"]})
    paths = ModelExporter(model).export_bundle(tmp_path, config=config)
    assert paths["onnx"].exists()
    assert paths["onnx"].stat().st_size > 0
