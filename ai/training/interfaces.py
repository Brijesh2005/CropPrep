"""Ports / interfaces for the training package.

Keeps the training engine dependant on abstractions (ports) rather than
concrete implementations so optimizers, schedulers, loss weighters and
callbacks can be swapped without touching the engine (dependency inversion).

* :class:`Callback` — lifecycle hooks the :class:`~ai.training.trainer.Trainer`
  fires before / after batches, epochs and validation passes.
* :class:`SchedulerHandle` — a scheduler plus the stepping strategy
  (``epoch`` vs ``step``) so the trainer knows when to call ``step()``.
* :class:`Weighter` — computes per-task loss weights for the multi-task loss
  (fixed / uncertainty / GradNorm).
* :class:`FoldGenerator` — yields ``(train, val)`` observation subsets for a
  model-validation strategy (hold-out / K-fold / spatial / temporal).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #


class Callback(ABC):
    """Lifecycle hooks fired by the :class:`~ai.training.trainer.Trainer`.

    Subclasses override the hooks they care about; the default implementations
    are no-ops. Callbacks are notified on the primary rank only (in distributed
    training) unless a callback opts into all ranks via :attr:`all_ranks`.
    """

    #: Set to ``True`` to receive hooks on every rank (e.g. saving per-rank
    #: state). Defaults to ``False`` (primary rank only).
    all_ranks: bool = False

    def __init__(self) -> None:
        self.trainer: Any | None = None

    def set_trainer(self, trainer: Any) -> None:
        """Bind the callback to its owning trainer (called once on setup)."""
        self.trainer = trainer

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Called once before the first epoch."""

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Called once after the final epoch (or early stop)."""

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Called at the start of every epoch."""

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Called at the end of every epoch (after validation, if enabled)."""

    def on_batch_begin(
        self, step: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called before every optimizer step."""

    def on_batch_end(self, step: int, logs: dict[str, Any] | None = None) -> None:
        """Called after every optimizer step (gradient updates applied)."""

    def on_validation_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Called before a validation pass."""

    def on_validation_end(self, logs: dict[str, Any] | None = None) -> None:
        """Called after a validation pass with the aggregated metrics."""

    def on_checkpoint_save(
        self, path: Any, kind: str, logs: dict[str, Any] | None = None
    ) -> None:
        """Called after a checkpoint is written (``kind``: best/latest/periodic)."""

    def on_exception(self, exc: BaseException) -> None:
        """Called if the training loop raises an exception."""


# --------------------------------------------------------------------------- #
# Schedulers
# --------------------------------------------------------------------------- #


@dataclass
class SchedulerHandle:
    """A scheduler plus its stepping policy.

    Attributes:
        scheduler: The underlying ``torch.optim.lr_scheduler`` object.
        step_period: ``"epoch"`` — step once per epoch; ``"step"`` — step once
            per optimizer step.
        requires_metric: Whether ``step()`` needs the validation metric
            (``ReduceLROnPlateau``).
        monitor_metric: The metric key to feed ``step()`` (when
            ``requires_metric``).
    """

    scheduler: Any
    step_period: str = "epoch"
    requires_metric: bool = False
    monitor_metric: str = "val_loss"
    _last_lr: list[float] = field(default_factory=list)

    def step(self, metric: float | None = None) -> None:
        """Advance the scheduler (feeding ``metric`` when required)."""
        if self.requires_metric:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
        self._last_lr = self.scheduler.get_last_lr()

    def get_last_lr(self) -> list[float]:
        """Current learning rates (best-effort for logging)."""
        if self._last_lr:
            return self._last_lr
        try:
            return self.scheduler.get_last_lr()
        except Exception:
            return self.scheduler.get_lr()


# --------------------------------------------------------------------------- #
# Loss weighting
# --------------------------------------------------------------------------- #


class Weighter(ABC):
    """Computes per-task loss weights for the multi-task loss."""

    name: str = "abstract"

    @abstractmethod
    def compute(
        self,
        per_task_losses: dict[str, Any],
        *,
        inputs: dict[str, Any],
        targets: dict[str, Any],
        model: Any,
    ) -> dict[str, Any]:
        """Return a mapping of task name -> weight (float or tensor)."""


# --------------------------------------------------------------------------- #
# Model validation (fold generation)
# --------------------------------------------------------------------------- #


class FoldGenerator(ABC):
    """Yields (train, val) observation subsets for cross-validation."""

    strategy: str = "abstract"

    @abstractmethod
    def folds(self, observations: Sequence[Any]) -> Iterator[tuple[list[Any], list[Any]]]:
        """Yield ``(train_observations, val_observations)`` per fold."""
