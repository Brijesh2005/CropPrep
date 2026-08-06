"""Architecture registry, version management, metadata and summary-shape tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from training.models import CheckpointManager, ModelConfig, ModelFactory
from training.models.cropfusion import CropFusionModel
from training.models.exceptions import ModelConfigurationError


def _tiny_cfg(**overrides) -> ModelConfig:
    defaults: dict = {
        "tabular": {"numeric_dim": 2, "categorical_cardinalities": [3]},
        "image_encoder": {"backbone": None},
        "shared_encoder": {"d_model": 32, "depth": 1, "num_heads": 4, "ff_dim": 64,
                           "out_dim": 64},
        "heads": {"crop": {"num_classes": 2}, "yield_prediction": {}},
    }
    for key, value in overrides.items():
        defaults[key] = value
    return ModelConfig(**defaults)


# --------------------------------------------------------------------------- #
# Architecture registry / version management
# --------------------------------------------------------------------------- #


def test_registry_contains_builtin():
    assert "cropfusion_v1" in ModelFactory.architecture_names()
    assert ModelFactory.resolve_architecture(_tiny_cfg()) is CropFusionModel


def test_create_uses_builtin_by_name():
    cfg = _tiny_cfg(name="roundtrip")  # unregistered display name
    model = ModelFactory.create(cfg)
    assert isinstance(model, CropFusionModel)


def test_register_and_create_custom_architecture():
    class DummyArchitecture(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.config = config
            self.linear = nn.Linear(2, 2)

    ModelFactory.register_architecture("dummy_v1", DummyArchitecture)
    try:
        assert "dummy_v1" in ModelFactory.architecture_names()
        model = ModelFactory.create(_tiny_cfg(name="dummy_v1"))
        assert isinstance(model, DummyArchitecture)
        explicit = ModelFactory.create(_tiny_cfg(), architecture="dummy_v1")
        assert isinstance(explicit, DummyArchitecture)
    finally:
        ModelFactory._ARCHITECTURES.pop("dummy_v1", None)


def test_register_rejects_non_module():
    with pytest.raises(ModelConfigurationError):
        ModelFactory.register_architecture("not_a_module", object)
    assert "not_a_module" not in ModelFactory.architecture_names()


def test_create_with_unknown_architecture_raises():
    with pytest.raises(ModelConfigurationError):
        ModelFactory.create(_tiny_cfg(), architecture="does_not_exist")


def test_architecture_version_default():
    assert _tiny_cfg().architecture_version == "1.0.0"
    assert _tiny_cfg(architecture_version="2.0.0").architecture_version == "2.0.0"


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def test_metadata_contents(model):
    meta = model.metadata
    assert meta["name"] == "cropfusion_v1"
    assert meta["architecture_version"] == "1.0.0"
    assert meta["output_names"] == ["crop", "yield"]
    assert meta["embedding_dims"]["image"] == model.image_dim
    assert meta["shared_dim"] == 128
    assert meta["pytorch_version"].startswith("2.")
    assert meta["python_version"]


def test_checkpoint_stores_metadata_and_architecture(model, tmp_path: Path):
    path = CheckpointManager(tmp_path).save(model, epoch=1, metrics={"loss": 0.5})
    state = CheckpointManager.load(path)
    assert state["architecture"] == "cropfusion_v1"
    assert state["architecture_version"] == "1.0.0"
    assert state["metadata"]["shared_dim"] == model.shared_encoder.output_dim
    assert isinstance(state["metadata"]["pytorch_version"], str)
    assert "model_state_dict" in state


def test_checkpoint_roundtrip_preserves_weights(model, config, tmp_path: Path):
    path = CheckpointManager(tmp_path).save(model)
    restored = ModelFactory.from_checkpoint(path)
    batch = model.sample_batch(batch_size=2, seq_len=4)
    model.eval()
    restored.eval()
    with torch.no_grad():
        a = model(batch).crop_logits
        b = restored(batch).crop_logits
    assert torch.allclose(a, b)


def test_from_checkpoint_rebuilds_registered_architecture(model, tmp_path: Path):
    path = CheckpointManager(tmp_path).save(model)
    restored = ModelFactory.from_checkpoint(path)
    assert type(restored) is type(model)


# --------------------------------------------------------------------------- #
# Summary shapes / architecture report
# --------------------------------------------------------------------------- #


def test_summary_includes_shapes_and_report(model, batch):
    summary = model.summary(sample_batch=batch)
    assert summary["metadata"]["name"] == "cropfusion_v1"
    assert summary["input_shapes"]["tabular"] == [4, 5]
    assert summary["output_shapes"]["crop_logits"] == [4, 3]
    assert summary["output_shapes"]["shared_representation"] == [4, 128]
    report = summary["architecture_report"]
    assert report
    first = report[0]
    assert first["name"]
    assert first["type"]
    assert first["params"] >= 0
    assert first["output_shapes"] is not None


def test_summary_without_sample_batch(model):
    summary = model.summary()
    assert summary["architecture_report"] is None
    assert summary["input_shapes"] is None
    assert summary["output_shapes"] is None


def test_architecture_report_lists_every_named_module(model, batch):
    from training.models.utils import architecture_report

    report = architecture_report(model, forward_fn=lambda: model(batch))
    names = {row["name"] for row in report}
    assert "fusion_engine.shared_encoder" in names
    assert "heads" in names
    assert "tabular_encoder" not in names
