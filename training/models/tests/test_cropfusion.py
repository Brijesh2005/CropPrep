"""Full-model tests: forward shapes, gradients, gates, summary, validation."""

from __future__ import annotations

import pytest
import torch

from training.models import CropFusionModel, ModelFactory
from training.models.exceptions import ModelInputError, ShapeMismatchError


def test_forward_shapes(model, batch):
    model.train()
    out = model(batch)
    assert out.crop_logits.shape == (4, 3)
    assert out.yield_pred.shape == (4, 1)
    assert out.shared_representation.shape == (4, 128)
    assert out.tabular_embedding is not None
    assert out.image_embedding is not None


def test_gradient_flows_through_all_params(model, batch):
    model.train()
    out = model(batch)
    (out.crop_logits.sum() + out.yield_pred.sum()).backward()
    no_grad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert no_grad == []


def test_gates_in_unit_interval(model, batch):
    out = model(batch)
    assert set(out.gates) == {"image_gate", "tabular_gate", "fusion_gate"}
    for gate in out.gates.values():
        assert (gate >= 0).all() and (gate <= 1).all()


def test_eval_and_train_shapes_match(model, batch):
    model.eval()
    with torch.no_grad():
        eval_out = model(batch)
    model.train()
    train_out = model(batch)
    assert eval_out.crop_logits.shape == train_out.crop_logits.shape


def test_export_forward_tuple(model, batch):
    out = model.forward_export(
        batch["tabular"], batch["ndvi"], batch["evi"], batch["temporal_mask"]
    )
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert out[0].shape == (4, 3)
    assert out[1].shape == (4, 1)


def test_validation_rejects_bad_tabular_width(model, batch):
    bad = dict(batch)
    bad["tabular"] = torch.randn(4, 99)
    with pytest.raises(ShapeMismatchError):
        model(bad)


def test_validation_rejects_missing_mask(model, batch):
    bad = {k: v for k, v in batch.items() if k != "temporal_mask"}
    with pytest.raises(ModelInputError):
        model(bad)


def test_validation_rejects_non_tensor(model, batch):
    bad = dict(batch)
    bad["ndvi"] = [[[0.0]]]
    with pytest.raises(ModelInputError):
        model(bad)


def test_validation_can_be_disabled(model, batch):
    model.config.validate_inputs = False
    try:
        out = model(batch)  # a valid batch still runs end-to-end
        assert out.crop_logits is not None
    finally:
        model.config.validate_inputs = True


def test_model_save_config(model, tmp_path):
    path = model.save_config(tmp_path / "model.yaml")
    assert path.exists()


def test_summary_contains_counts(model, batch):
    summary = model.summary(sample_batch=batch)
    assert summary["parameter_count"] > 0
    assert summary["parameter_summary"]["trainable"] > 0
    assert len(summary["layer_summary"]) > 0
    assert "parameters_mb" in summary["memory_estimate"]
    assert "activation_mb" in summary["memory_estimate"]


def test_tabular_only_model(tabular_only_config):
    model = ModelFactory.create(tabular_only_config)
    batch = model.sample_batch(batch_size=3)
    out = model(batch)
    assert out.crop_logits.shape == (3, 4)
    assert out.yield_pred.shape == (3, 1)
    assert out.image_embedding is None
    assert out.gates == {}
    assert "tabular" in batch and "ndvi" not in batch


def test_image_only_model(image_only_config):
    model = ModelFactory.create(image_only_config)
    batch = model.sample_batch(batch_size=3)
    out = model(batch)
    assert out.crop_logits.shape == (3, 3)
    assert out.tabular_embedding is None
    assert out.gates == {}


def test_add_extra_head(model, batch):
    from training.models import CropHead

    shared_dim = model.shared_encoder.output_dim
    model.add_head("health", CropHead(shared_dim, num_classes=2))
    out = model(batch)
    assert "health" in model.output_names
    assert out.as_dict()["crop_logits"] is not None
    model.remove_head("health")
    assert "health" not in model.output_names
