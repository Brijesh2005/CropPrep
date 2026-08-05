"""Learning-rate scheduler factory for the training engine.

Supports cosine annealing, OneCycle, ReduceLROnPlateau, polynomial decay and
linear warmup composed with cosine / polynomial schedules.

``build_scheduler`` returns a :class:`~ai.training.interfaces.SchedulerHandle`
that also encodes *when* the scheduler should be stepped (``epoch`` vs
``step``) so the trainer never guesses.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import lr_scheduler

from .config import SchedulerConfig
from .exceptions import SchedulerBuildError
from .interfaces import SchedulerHandle

_WARMUP_SCHEDULERS = ("warmup_cosine", "warmup_polynomial")


def _effective_warmup(config: SchedulerConfig, step_period: str, total_iters: int) -> int:
    """Warmup length in the scheduler's stepping unit."""
    if config.warmup_ratio > 0:
        return max(1, int(total_iters * config.warmup_ratio))
    return config.warmup_epochs if step_period == "epoch" else config.warmup_steps


def _initial_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _base_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    name: str,
    *,
    start_iters: int,
    total_iters: int,
) -> Any:
    """Build the non-warmup scheduler for ``name`` (no warmup applied)."""
    if name == "cosine" or name == "warmup_cosine":
        t_max = config.t_max or max(1, total_iters - start_iters)
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=config.eta_min
        )

    if name == "polynomial" or name == "warmup_polynomial":
        return _polynomial_lr(optimizer, config, total_iters)

    if name == "onecycle":
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=_initial_lr(optimizer),
            total_steps=total_iters,
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor,
        )

    if name == "reduce_on_plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
        )

    raise SchedulerBuildError(f"unknown scheduler {name!r}")


def _polynomial_lr(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    total_iters: int,
) -> Any:
    """Polynomial decay reaching ``config.end_lr`` at ``total_iters``."""
    base_lr = _initial_lr(optimizer)
    end_factor = (config.end_lr / base_lr) if base_lr > 0 else 0.0
    power = config.power

    def lr_lambda(step: int) -> float:
        progress = min(step, max(total_iters, 1)) / max(total_iters, 1)
        return end_factor + (1.0 - end_factor) * (1.0 - progress) ** power

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig | None = None,
    *,
    steps_per_epoch: int,
    total_epochs: int,
) -> SchedulerHandle | None:
    """Build a scheduler (and stepping policy) from a :class:`SchedulerConfig`.

    Args:
        optimizer: The optimizer to schedule.
        config: Validated :class:`SchedulerConfig`.
        steps_per_epoch: Number of optimizer steps per epoch.
        total_epochs: Total epochs of the run.

    Returns:
        A :class:`SchedulerHandle` describing how / when to step, or ``None``
        when the scheduler name is ``"none"``.

    Raises:
        SchedulerBuildError: On an unknown scheduler name or an invalid
            warmup / step combination.
    """
    config = config or SchedulerConfig()
    name = (config.name or "none").strip().lower()
    if name == "none":
        return None

    step_period = config.step
    if name == "onecycle" and step_period != "step":
        step_period = "step"
    if name == "reduce_on_plateau":
        step_period = "epoch"

    total_iters = (
        steps_per_epoch * total_epochs if step_period == "step" else total_epochs
    )
    warmup = _effective_warmup(config, step_period, total_iters)

    if name in _WARMUP_SCHEDULERS and warmup > 0:
        if warmup >= total_iters:
            raise SchedulerBuildError(
                f"warmup ({warmup}) must be smaller than the schedule length "
                f"({total_iters})"
            )
        base_name = name.replace("warmup_", "")
        base = _base_scheduler(
            optimizer, config, base_name, start_iters=warmup, total_iters=total_iters
        )
        warmup_lr = lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
        )
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr, base], milestones=[warmup]
        )
    else:
        scheduler = _base_scheduler(
            optimizer, config, name, start_iters=0, total_iters=total_iters
        )

    handle = SchedulerHandle(
        scheduler=scheduler,
        step_period=step_period,
        requires_metric=name == "reduce_on_plateau",
        monitor_metric="val_loss",
    )
    handle._last_lr = scheduler.get_last_lr()
    return handle


def get_lr(optimizer: torch.optim.Optimizer) -> list[float]:
    """Current learning rates from an optimizer (for logging)."""
    return [float(group["lr"]) for group in optimizer.param_groups]
