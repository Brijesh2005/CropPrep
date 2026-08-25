"""CropFusionTrainer tests: curriculum + class weights + reports end-to-end."""

from __future__ import annotations

import copy

import pytest
import torch

from training.models import ModelFactory
from training.training import (
    CropFusionTrainer,
    CropFusionTrainingResult,
    TrainingConfig,
)
from training.training.tests.conftest import make_fake_loader, small_full_config


def _make_config(tmp_path, **overrides) -> TrainingConfig:
    data = {
        "general": {
            "device": "cpu",
            "seed": 42,
            "reports": True,
            "output_dir": str(tmp_path / "out"),
        },
        "train": {"epochs": 5, "early_stopping_patience": 3},
        "loss": {"class_weight_mode": "balanced"},
        "curriculum": {"enabled": True},
        "checkpoint": {"directory": str(tmp_path / "ckpt")},
        "logging": {"console": False},
    }
    data.update(overrides)
    return TrainingConfig(**data)


def test_cropfusion_trainer_full_flow(tabular_model, tmp_path):
    torch.manual_seed(0)
    config = _make_config(tmp_path, train={"epochs": 5, "early_stopping_patience": 100})
    train_loader = make_fake_loader(n=16, batch_size=8)
    trainer = CropFusionTrainer(
        tabular_model, train_loader, config,
        val_loader=make_fake_loader(n=16, batch_size=8),
    )
    result = trainer.train()

    assert isinstance(result, CropFusionTrainingResult)
    assert len(result.history) == 5
    # Tabular-only model: image / temporal / fusion stages are skipped.
    assert [h["stage"] for h in result.history] == [
        "tabular", "tabular", "tabular", "finetune", "finetune",
    ]
    assert len(result.stages) == 5
    assert trainer.class_weights is not None
    assert trainer.class_weights.numel() == 3
    assert abs(trainer.class_weights.mean().item() - 1.0) < 1e-3
    assert set(result.reports) == {
        "training", "validation", "metrics", "checkpoint", "learning_curve",
    }
    assert (tmp_path / "out" / "reports" / "training_report.md").exists()
    assert result.best_path is not None
    assert result.summary()["stages"] == 5


def test_cropfusion_trainer_curriculum_disabled(tabular_model, tmp_path):
    config = _make_config(tmp_path, curriculum={"enabled": False})
    trainer = CropFusionTrainer(
        tabular_model, make_fake_loader(n=16, batch_size=8), config
    )
    result = trainer.train()
    assert result.stages == []
    assert "stage" not in result.history[0]


def test_cropfusion_trainer_class_weights_disabled(tabular_model, tmp_path):
    config = _make_config(
        tmp_path, loss={"class_weight_mode": "none"}, curriculum={"enabled": False}
    )
    trainer = CropFusionTrainer(
        tabular_model, make_fake_loader(n=16, batch_size=8), config
    )
    assert trainer.class_weights is None
    assert trainer.class_frequency is None
    result = trainer.train()
    assert result.epochs == 5


def test_cropfusion_trainer_reports_disabled(tabular_model, tmp_path):
    config = _make_config(
        tmp_path,
        general={
            "device": "cpu", "seed": 42, "reports": False,
            "output_dir": str(tmp_path / "out"),
        },
    )
    trainer = CropFusionTrainer(
        tabular_model, make_fake_loader(n=16, batch_size=8), config
    )
    result = trainer.train()
    assert result.reports == {}
    assert not (tmp_path / "out" / "reports").exists()


def test_cropfusion_trainer_resume_from_stage(tmp_path):
    """start_stage skips earlier stages (resume-from-any-stage semantics)."""
    config = _make_config(
        tmp_path, curriculum={"enabled": True, "start_stage": 3}, train={"epochs": 3}
    )
    model = ModelFactory.create(small_full_config())
    trainer = CropFusionTrainer(
        model,
        make_fake_loader(n=8, batch_size=8, feature_dim=4, multimodal=True),
        config,
    )
    result = trainer.train()
    stages = [h["stage"] for h in result.history]
    assert stages == ["temporal", "fusion", "finetune"]


def test_cropfusion_trainer_compile_eager(tabular_model, tmp_path):
    """torch.compile wiring (eager backend = cheap no-op-ish wrapper)."""
    config = _make_config(
        tmp_path,
        general={
            "device": "cpu", "seed": 42, "reports": False,
            "output_dir": str(tmp_path / "out"),
            "compile": True, "compile_backend": "eager",
        },
        train={"epochs": 1},
        curriculum={"enabled": False},
    )
    trainer = CropFusionTrainer(
        tabular_model, make_fake_loader(n=8, batch_size=8), config
    )
    result = trainer.train()
    assert result.epochs == 1
    assert result.steps > 0


def test_cropfusion_trainer_is_a_trainer(tabular_model, tmp_path):
    config = _make_config(tmp_path, curriculum={"enabled": False})
    trainer = CropFusionTrainer(
        tabular_model, make_fake_loader(n=8, batch_size=8), config
    )
    assert hasattr(trainer, "optimizer")
    assert hasattr(trainer, "checkpoint_manager")
    assert hasattr(trainer, "loss_module")


def test_cropfusion_trainer_manual_loss_module(tmp_path):
    """A user-supplied loss module is respected (no class weights injected)."""
    from training.training.losses import build_multi_task_loss

    config = _make_config(
        tmp_path,
        loss={"class_weight_mode": "balanced"},
        curriculum={"enabled": False},
        train={"epochs": 1},
    )
    model = ModelFactory.create(small_full_config())
    manual = build_multi_task_loss(config.loss)
    trainer = CropFusionTrainer(
        model,
        make_fake_loader(n=8, batch_size=8, feature_dim=4, multimodal=True),
        config,
        loss_module=manual,
    )
    assert trainer.loss_module is manual
    result = trainer.train()
    assert result.epochs == 1


def test_collect_class_counts_fewer_labels_than_num_classes():
    """_collect_class_counts must not crash when a batch has fewer distinct
    labels than num_classes (the bincount-result was smaller than counts)."""
    from training.training.cropfusion_trainer import _collect_class_counts

    # Simulate a loader with 2 batches: first has labels [0,1], second has [0,1,2]
    # but num_classes=5 (counts tensor is size 5).
    class _FakeLoader:
        def __init__(self):
            self._batches = [
                {"crop_label": torch.tensor([0, 0, 1, 1])},
                {"crop_label": torch.tensor([0, 1, 2])},
            ]
            self._idx = 0

        def __iter__(self):
            self._idx = 0
            return self

        def __next__(self):
            if self._idx >= len(self._batches):
                raise StopIteration
            batch = self._batches[self._idx]
            self._idx += 1
            return batch

    counts = _collect_class_counts(_FakeLoader(), num_classes=5)
    assert counts.shape == (5,)
    assert counts[0].item() == 3  # label 0 appears 3 times
    assert counts[1].item() == 3  # label 1 appears 3 times
    assert counts[2].item() == 1  # label 2 appears 1 time
    assert counts[3].item() == 0
    assert counts[4].item() == 0
