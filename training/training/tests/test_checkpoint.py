"""Checkpoint tests: save / restore / resume with full training state."""

from __future__ import annotations

import copy

import torch

from training.training import (
    TrainingCheckpointManager,
    build_optimizer,
    build_scheduler,
    capture_rng_state,
    restore_rng_state,
)
from training.training.config import OptimizerConfig, SchedulerConfig


def _build(tmp_path, tabular_model):
    manager = TrainingCheckpointManager(tmp_path / "ckpt", keep_last=2)
    optimizer = build_optimizer(tabular_model, OptimizerConfig(lr=1e-3))
    scheduler_handle = build_scheduler(
        optimizer, SchedulerConfig(name="cosine"),
        steps_per_epoch=4, total_epochs=2,
    )
    return manager, optimizer, scheduler_handle


def test_save_restore_model_weights(tmp_path, tabular_model):
    manager, optimizer, scheduler_handle = _build(tmp_path, tabular_model)

    manager.save_latest(
        tabular_model, optimizer=optimizer, scheduler=scheduler_handle.scheduler,
        epoch=2, step=40, metrics={"val_loss": 0.5},
        rng_state=capture_rng_state(),
    )

    from training.models import ModelConfig, ModelFactory

    restored = ModelFactory.create(
        ModelConfig(
            tabular={"numeric_dim": 4, "categorical_cardinalities": [3]},
            image_encoder={"backbone": None},
            heads={"crop": {"num_classes": 3}, "yield_prediction": {}},
        )
    )
    state = manager.restore(
        manager.latest_path, model=restored, optimizer=optimizer,
        scheduler=scheduler_handle.scheduler,
    )
    assert state.epoch == 2
    assert state.step == 40
    assert state.metrics["val_loss"] == 0.5
    assert all(
        torch.equal(a, b)
        for a, b in zip(tabular_model.state_dict().values(), restored.state_dict().values())
    )


def test_resume_latest_returns_none_when_absent(tmp_path, tabular_model):
    manager = TrainingCheckpointManager(tmp_path / "ckpt")
    assert manager.resume_latest(model=tabular_model) is None


def test_save_best_and_latest_paths(tmp_path, tabular_model):
    manager, optimizer, _ = _build(tmp_path, tabular_model)
    manager.save_best(tabular_model, optimizer=optimizer, epoch=1, metrics={})
    manager.save_latest(tabular_model, optimizer=optimizer, epoch=2, metrics={})
    assert manager.best_path.exists()
    assert manager.latest_path.exists()
    assert manager.best_path != manager.latest_path


def test_scheduler_state_restored(tmp_path, tabular_model):
    manager, optimizer, handle = _build(tmp_path, tabular_model)
    for _ in range(3):
        handle.step()
    manager.save_latest(tabular_model, optimizer=optimizer, scheduler=handle.scheduler,
                        epoch=1, step=10, metrics={})

    optimizer2 = build_optimizer(copy.deepcopy(tabular_model), OptimizerConfig(lr=1e-3))
    handle2 = build_scheduler(optimizer2, SchedulerConfig(name="cosine"),
                              steps_per_epoch=4, total_epochs=2)
    manager.restore(manager.latest_path, scheduler=handle2.scheduler)
    assert handle.scheduler.state_dict()["last_epoch"] == handle2.scheduler.state_dict()["last_epoch"]


def test_rng_state_round_trip():
    import random

    import numpy as np

    state = capture_rng_state()
    torch.randn(3)
    random.random()
    np.random.randn(2)
    restore_rng_state(state)
    # After restore, draws reproduce the original sequence.
    a1 = torch.randn(1)
    torch.randn(1)
    restore_rng_state(state)
    a2 = torch.randn(1)
    assert torch.equal(a1, a2)
