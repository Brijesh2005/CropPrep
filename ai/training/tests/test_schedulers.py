"""Scheduler tests: each scheduler builds and decays / ramps LR correctly."""

from __future__ import annotations

import torch
import torch.nn as nn

from ai.training import build_scheduler
from ai.training.config import OptimizerConfig, SchedulerConfig
from ai.training.optimizers import build_optimizer


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _fresh_optimizer():
    return build_optimizer(_Net(), OptimizerConfig(name="adamw", lr=1e-3))


def _run(handle, steps: int, metric: float | None = None) -> list[float]:
    lrs = []
    for _ in range(steps):
        lrs.append(handle.get_last_lr()[0])
        handle.step(metric)
    return lrs


def test_cosine_decays():
    handle = build_scheduler(
        _fresh_optimizer(), SchedulerConfig(name="cosine"),
        steps_per_epoch=10, total_epochs=5,
    )
    lrs = _run(handle, 5)
    assert lrs[0] > lrs[-1]  # decays over epochs


def test_polynomial_reaches_end_lr():
    handle = build_scheduler(
        _fresh_optimizer(), SchedulerConfig(name="polynomial", end_lr=0.0001),
        steps_per_epoch=10, total_epochs=5,
    )
    lrs = _run(handle, 6)  # one extra step so the final LR is captured
    assert lrs[1] < lrs[0]
    assert abs(lrs[-1] - 0.0001) < 1e-5


def test_warmup_cosine_ramps_then_decays():
    handle = build_scheduler(
        _fresh_optimizer(),
        SchedulerConfig(name="warmup_cosine", step="step", warmup_steps=4),
        steps_per_epoch=10, total_epochs=3,
    )
    lrs = _run(handle, 8)
    assert lrs[1] > lrs[0]  # warmup ramps up
    assert max(lrs) <= lrs[4] + 1e-9  # peak at end of warmup


def test_onecycle_steps_per_step():
    handle = build_scheduler(
        _fresh_optimizer(), SchedulerConfig(name="onecycle", step="step"),
        steps_per_epoch=10, total_epochs=3,
    )
    assert handle.step_period == "step"
    lrs = _run(handle, 10)
    assert lrs[-1] > lrs[0]  # ramping phase


def test_reduce_on_plateau_requires_metric():
    handle = build_scheduler(
        _fresh_optimizer(), SchedulerConfig(name="reduce_on_plateau"),
        steps_per_epoch=10, total_epochs=5,
    )
    assert handle.requires_metric
    lrs = _run(handle, 5, metric=0.5)
    assert lrs[0] == lrs[-1]  # no improvement -> constant


def test_none_returns_none():
    handle = build_scheduler(
        _fresh_optimizer(), SchedulerConfig(name="none"),
        steps_per_epoch=10, total_epochs=5,
    )
    assert handle is None
