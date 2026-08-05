"""Trainer tests: end-to-end training, resume, NaN detection, checkpoints."""

from __future__ import annotations

import copy
import os

import pytest
import torch

from training.training import Trainer, TrainingConfig
from training.training.callbacks import EarlyStopping, HistoryRecorder
from training.training.config import LossConfig

from training.training.tests.conftest import make_fake_loader


def test_trainer_trains_and_returns_result(tabular_model, fake_loader, train_config):
    recorder = HistoryRecorder()
    trainer = Trainer(
        tabular_model,
        fake_loader,
        train_config,
        val_loader=make_fake_loader(n=8, batch_size=8),
        callbacks=[recorder],
    )
    result = trainer.train()
    assert len(result.history) == 2  # 2 epochs
    assert "train_loss" in result.history[-1]
    assert result.steps > 0
    assert result.best_path is not None  # best checkpoint saved
    assert trainer.checkpoint_manager.best_path.exists()


def test_trainer_val_metrics(tabular_model, fake_loader, train_config):
    val_loader = make_fake_loader(n=8, batch_size=8)
    trainer = Trainer(tabular_model, fake_loader, train_config, val_loader=val_loader)
    result = trainer.train()
    assert "val_loss" in result.history[-1]
    assert "crop/accuracy" in result.history[-1]


def test_gradient_accumulation_matches_single_large_batch():
    """2 micro-batches with accum=2 == one batch of double size (no dropout)."""
    from training.models import ModelConfig, ModelFactory

    from training.training import Trainer

    model_config = ModelConfig(
        tabular={"numeric_dim": 4, "categorical_cardinalities": [3]},
        image_encoder={"backbone": None},
        heads={"crop": {"num_classes": 3}, "yield_prediction": {}},
    )
    base_model = ModelFactory.create(model_config)
    # Disable every Dropout so forward passes are batch-size deterministic
    # (including the scalar attention-dropout inside nn.MultiheadAttention).
    for module in base_model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.0
            module.attn_dropout_p = 0.0

    torch.manual_seed(0)
    tabular = torch.randn(16, 5)
    crop = torch.randint(0, 3, (16,))
    yield_label = torch.randn(16, 1)
    single_loader = _explicit_loader([{"tabular": tabular, "crop_label": crop,
                                       "yield_label": yield_label}])
    micro_loader = _explicit_loader(
        [
            {"tabular": tabular[:8], "crop_label": crop[:8],
             "yield_label": yield_label[:8]},
            {"tabular": tabular[8:], "crop_label": crop[8:],
             "yield_label": yield_label[8:]},
        ]
    )

    def run(accum: int, loader) -> float:
        model = copy.deepcopy(base_model)
        config = TrainingConfig(
            name=f"accum{accum}",
            general={"device": "cpu", "seed": 42,
                     "gradient_accumulation_steps": accum},
            train={"epochs": 1, "early_stopping_patience": 3},
            logging={"console": False},
            checkpoint={"save_best": False, "save_latest": False},
        )
        return Trainer(model, loader, config).train().history[-1]["train_loss"]

    loss_accum = run(2, micro_loader)   # 2 batches, 1 optimizer step
    loss_single = run(1, single_loader)  # 1 batch, 1 optimizer step
    assert loss_accum == pytest.approx(loss_single, rel=1e-3)


def _explicit_loader(batches):
    class _Fake:
        def __len__(self) -> int:
            return len(batches)

        def __iter__(self):
            return iter(batches)

    return _Fake()


def test_gradient_clipping_passes_max_norm(monkeypatch, tabular_model):
    import torch.nn.utils as nn_utils

    from training.training import Trainer

    captured: dict[str, float] = {}

    def fake_clip_norm(params, max_norm, **kwargs):
        captured["max_norm"] = float(max_norm)
        return 0.0

    monkeypatch.setattr(nn_utils, "clip_grad_norm_", fake_clip_norm)
    model = copy.deepcopy(tabular_model)
    config = TrainingConfig(
        name="clip",
        general={"device": "cpu", "seed": 42, "gradient_clip": 0.25,
                 "gradient_clip_type": "norm"},
        train={"epochs": 1, "early_stopping_patience": 3},
        logging={"console": False},
        checkpoint={"save_best": False, "save_latest": False},
    )
    trainer = Trainer(model, make_fake_loader(n=16, batch_size=8), config)
    trainer.train()
    assert captured["max_norm"] == 0.25


def test_nan_policy_skip_does_not_crash(tabular_model):
    from training.training import Trainer

    model = copy.deepcopy(tabular_model)

    class NanLoader:
        def __len__(self) -> int:
            return 2

        def __iter__(self):
            nan_batch = {"tabular": torch.full((8, 5), float("nan")),
                         "crop_label": torch.randint(0, 3, (8,)),
                         "yield_label": torch.randn(8, 1)}
            good = {"tabular": torch.randn(8, 5),
                    "crop_label": torch.randint(0, 3, (8,)),
                    "yield_label": torch.randn(8, 1)}
            return iter([nan_batch, good])

    config = TrainingConfig(
        name="nan",
        general={"device": "cpu", "seed": 42, "nan_detection": True,
                 "nan_policy": "skip"},
        train={"epochs": 1, "early_stopping_patience": 3},
        logging={"console": False},
        checkpoint={"save_best": False, "save_latest": False},
    )
    trainer = Trainer(model, NanLoader(), config)
    result = trainer.train()
    assert result.steps >= 0


def test_resume_continues_from_checkpoint(tabular_model, train_config):
    from training.training import Trainer

    train_config.train.epochs = 2
    train_config.checkpoint.save_latest = True
    trainer1 = Trainer(tabular_model, make_fake_loader(), train_config)
    r1 = trainer1.train()
    assert r1.steps == 4  # 2 batches x 2 epochs

    # Resume for 4 total epochs.
    resumed_config = copy.deepcopy(train_config)
    resumed_config.train.epochs = 4
    resumed_config.checkpoint.resume = True
    from training.models import ModelFactory

    trainer2 = Trainer(
        ModelFactory.create(tabular_model.config), make_fake_loader(), resumed_config
    )
    r2 = trainer2.train()
    assert len(r2.history) == 2  # resumes after epoch index 1, runs 2..3
    assert r2.history[0]["epoch"] == 3


def test_early_stopping_via_callback(tabular_model, train_config):
    from training.training import Trainer

    train_config.train.early_stopping_patience = 1
    early = EarlyStopping(monitor="val_loss", mode="min", patience=1)
    trainer = Trainer(
        tabular_model,
        make_fake_loader(),
        train_config,
        val_loader=make_fake_loader(n=8, batch_size=8),
        callbacks=[early],
    )
    trainer.train()
    assert early.best_epoch is not None
