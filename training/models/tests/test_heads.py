"""Multi-task head tests: shapes, registry, extensibility."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from training.models import CropHead, MultiTaskHeads, YieldHead
from training.models.exceptions import ModelConfigurationError


def test_crop_head_shape():
    head = CropHead(in_dim=64, num_classes=5)
    out = head(torch.randn(4, 64))
    assert out.shape == (4, 5)


def test_crop_head_requires_classes():
    with pytest.raises(ModelConfigurationError):
        CropHead(in_dim=64, num_classes=0)


def test_yield_head_shape():
    head = YieldHead(in_dim=64, output_clamp_min=0.0)
    out = head(torch.randn(4, 64))
    assert out.shape == (4, 1)
    assert (out >= 0).all()


def test_yield_head_unclamped():
    head = YieldHead(in_dim=64, output_clamp_min=None)
    out = head(torch.randn(4, 64))
    assert out.shape == (4, 1)


def test_registry_built_heads(model, batch):
    outputs = model.heads(torch.randn(batch["tabular"].size(0), 128))
    assert "crop" in outputs
    assert "yield" in outputs
    assert outputs["crop"].shape[1] == 3
    assert outputs["yield"].shape[1] == 1


def test_add_and_remove_head():
    heads = MultiTaskHeads()
    heads.add_head("crop", CropHead(64, 3))
    heads.add_head("yield", YieldHead(64))
    assert heads.names == ["crop", "yield"]
    assert "crop" in heads
    assert heads.output_dims == {"crop": 3, "yield": 1}
    heads.remove_head("crop")
    assert heads.names == ["yield"]


def test_duplicate_head_rejected():
    heads = MultiTaskHeads({"crop": CropHead(64, 3)})
    with pytest.raises(ModelConfigurationError):
        heads.add_head("crop", CropHead(64, 3))


def test_named_head_outputs():
    heads = MultiTaskHeads({"crop": CropHead(64, 3), "yield": YieldHead(64)})
    aliased = heads.named_head_outputs({"crop": torch.zeros(2, 3), "yield": torch.zeros(2, 1)})
    assert "crop_logits" in aliased
    assert "yield_pred" in aliased


def test_add_head_rejects_non_module():
    heads = MultiTaskHeads()
    with pytest.raises(ModelConfigurationError):
        heads.add_head("bad", "not a module")


def test_gradient_flows():
    head = CropHead(in_dim=64, num_classes=4)
    head(torch.randn(3, 64)).sum().backward()
    assert all(p.grad is not None for p in head.parameters())
