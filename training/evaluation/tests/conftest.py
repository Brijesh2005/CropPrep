"""Shared fixtures for the evaluation test-suite (Phase R5)."""

from __future__ import annotations

import pytest
import torch

from training.models import ModelConfig, ModelFactory


def small_full_config() -> ModelConfig:
    """A fast full multimodal model config (tiny timm backbone)."""
    return ModelConfig(
        tabular={"numeric_dim": 3, "categorical_cardinalities": [2]},
        image_encoder={"backbone": "mobilenetv3_small_050", "input_size": 32},
        temporal={"d_model": 32, "depth": 1, "num_heads": 4, "ff_dim": 128,
                  "embedding_dim": 32, "max_len": 6},
        cross_attention={"num_heads": 4, "out_dim": 32},
        gated_fusion={"out_dim": 32, "hidden_dim": 32},
        shared_encoder={"d_model": 32, "depth": 1, "num_heads": 4, "ff_dim": 128,
                        "out_dim": 48},
        heads={"crop": {"num_classes": 3}, "yield_prediction": {}},
    )


def make_fake_loader(
    n: int = 16,
    batch_size: int = 8,
    feature_dim: int = 4,
    num_classes: int = 3,
    multimodal: bool = True,
    input_size: int = 32,
):
    """A loader yielding Phase-4-style batches (incl. image + labels)."""

    class _FakeLoader:
        def __init__(self) -> None:
            self.batches = []
            for _ in range(n // batch_size):
                batch = {
                    "tabular": torch.randn(batch_size, feature_dim),
                    "crop_label": torch.randint(0, num_classes, (batch_size,)),
                    "yield_label": torch.randn(batch_size, 1),
                }
                if multimodal:
                    seq_len = 2
                    batch["ndvi"] = torch.randn(
                        batch_size, seq_len, 1, input_size, input_size
                    )
                    batch["evi"] = torch.randn(
                        batch_size, seq_len, 1, input_size, input_size
                    )
                    batch["temporal_mask"] = torch.ones(
                        batch_size, seq_len, dtype=torch.bool
                    )
                self.batches.append(batch)
            self.n = n

        def __len__(self) -> int:
            return len(self.batches)

        def __iter__(self):
            return iter(self.batches)

    return _FakeLoader()


@pytest.fixture(scope="module")
def full_model() -> ModelFactory:
    model = ModelFactory.create(small_full_config())
    model.eval()
    return model


@pytest.fixture(scope="module")
def fake_loader():
    return make_fake_loader(n=16, batch_size=8)
