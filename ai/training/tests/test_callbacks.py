"""Callback tests: early stopping and checkpoint saving."""

from __future__ import annotations

import os

from ai.training import EarlyStopping, HistoryRecorder, LearningRateLogger
from ai.training.config import TrainingConfig


def test_early_stopping_triggers_after_patience():
    callback = EarlyStopping(monitor="val_loss", mode="min", patience=2, min_delta=0.0)
    losses = [1.0, 0.8, 0.81, 0.82, 0.83, 0.84]
    for epoch, loss in enumerate(losses):
        callback.on_epoch_end(epoch, {"val_loss": loss})
    assert callback.should_stop
    assert callback.stopped_epoch == 5


def test_early_stopping_never_triggers_on_improvement():
    callback = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    for epoch in range(10):
        callback.on_epoch_end(epoch, {"val_loss": 1.0 - epoch * 0.01})
    assert not callback.should_stop


def test_early_stopping_max_mode():
    callback = EarlyStopping(monitor="crop/accuracy", mode="max", patience=1)
    for epoch, value in enumerate([0.5, 0.5, 0.5, 0.6, 0.6]):
        callback.on_epoch_end(epoch, {"crop/accuracy": value})
    # Improves at epoch 3, then flat for 1 -> stops.
    assert callback.stopped_epoch == 4


def test_history_recorder():
    recorder = HistoryRecorder()
    recorder.on_epoch_end(0, {"train_loss": 1.0})
    recorder.on_epoch_end(1, {"train_loss": 0.5})
    assert len(recorder.history) == 2
    assert recorder.history[1]["train_loss"] == 0.5


def test_learning_rate_logger(train_config, tabular_model):
    import torch

    from ai.training import Trainer

    loader = _loader()
    trainer = Trainer(tabular_model, loader, train_config)
    logger = LearningRateLogger()
    logger.set_trainer(trainer)
    logs: dict[str, float] = {}
    logger.on_batch_end(1, logs)
    assert logs["lr"] == train_config.optimizer.lr


def _loader():
    import torch

    class _Fake:
        def __len__(self) -> int:
            return 2

        def __iter__(self):
            return iter(
                [
                    {"tabular": torch.randn(8, 5),
                     "crop_label": torch.randint(0, 3, (8,)),
                     "yield_label": torch.randn(8, 1)}
                    for _ in range(2)
                ]
            )

    return _Fake()
