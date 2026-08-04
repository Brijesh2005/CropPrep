"""Training checkpoint manager.

Extends the Phase 5 :class:`ai.models.checkpoint.CheckpointManager` with the
state a *training* run needs to resume exactly:

* optimizer / scheduler state (from the Phase 5 manager),
* AMP GradScaler state,
* torch / numpy / python random state,
* GradNorm controller state,
* the resolved training configuration and git hash.

Conventions: ``best.pt`` (best metric), ``latest.pt`` (most recent) and
``checkpoint_epochNNNN.pt`` (periodic / resume).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ai.models import CheckpointManager
from ai.models.exceptions import CheckpointError

from .exceptions import TrainingError
from .utils import get_git_hash

_BEST_NAME = "best.pt"
_LATEST_NAME = "latest.pt"


@dataclass
class TrainingResumeState:
    """What a training resume restores."""

    path: Path
    epoch: int | None
    step: int | None
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    model_config: dict[str, Any] | None


class TrainingCheckpointManager:
    """High-level checkpoint persistence for a training run.

    Args:
        directory: Checkpoint directory (created on demand).
        keep_last: Periodic checkpoints to retain (``None`` = keep all).
    """

    def __init__(self, directory: str | Path, keep_last: int | None = 3) -> None:
        self.directory = Path(directory)
        self.keep_last = keep_last
        self._core: CheckpointManager | None = None

    def _get_core(self) -> CheckpointManager:
        """Create the underlying Phase 5 manager lazily (dir on first save)."""
        if self._core is None:
            self._core = CheckpointManager(self.directory, keep_last=self.keep_last)
        return self._core

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    @property
    def best_path(self) -> Path:
        return self.directory / _BEST_NAME

    @property
    def latest_path(self) -> Path:
        return self.directory / _LATEST_NAME

    def latest_checkpoint(self) -> Path | None:
        """Most recent periodic checkpoint, or ``latest.pt`` if present."""
        if self.latest_path.exists():
            return self.latest_path
        periodic = sorted(
            self.directory.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        return periodic[-1] if periodic else None

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #

    def save(
        self,
        model: nn.Module,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        *,
        epoch: int | None = None,
        step: int | None = None,
        metrics: dict[str, Any] | None = None,
        scaler: Any | None = None,
        rng_state: Mapping[str, Any] | None = None,
        gradnorm: Any | None = None,
        training_config: Any | None = None,
        name: str | None = None,
    ) -> Path:
        """Persist a checkpoint and return its path.

        ``name`` defaults to ``best.pt`` / ``latest.pt`` / a periodic name
        depending on the caller; when ``None`` the Phase 5 epoch naming is
        used.
        """
        extra: dict[str, Any] = {
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "rng_state": rng_state if rng_state is not None else capture_rng_state(),
            "gradnorm_state": (
                gradnorm.state_dict() if gradnorm is not None else None
            ),
            "training_config": (
                # JSON mode keeps Path/datetime as primitives so the checkpoint
                # stays loadable with torch.load(weights_only=True).
                training_config.model_dump(mode="json")
                if training_config is not None
                else None
            ),
            "git_hash": get_git_hash(),
            "saved_by": "ai.training",
        }
        return self._get_core().save(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            metrics=metrics or {},
            extra=extra,
            name=name,
        )

    def save_best(self, model: nn.Module, **kwargs: Any) -> Path:
        """Save the best-metric checkpoint (``best.pt``)."""
        return self.save(model, name=_BEST_NAME, **kwargs)

    def save_latest(self, model: nn.Module, **kwargs: Any) -> Path:
        """Save the most-recent checkpoint (``latest.pt``)."""
        return self.save(model, name=_LATEST_NAME, **kwargs)

    def save_periodic(
        self,
        model: nn.Module,
        *,
        epoch: int,
        **kwargs: Any,
    ) -> Path:
        """Save a periodic checkpoint (``checkpoint_epochNNNN.pt``)."""
        return self.save(model, epoch=epoch, name=f"checkpoint_epoch{int(epoch):04d}.pt", **kwargs)

    # ------------------------------------------------------------------ #
    # Load / resume
    # ------------------------------------------------------------------ #

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        return CheckpointManager.load(path)

    def restore(
        self,
        path: str | Path,
        model: nn.Module | None = None,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        gradnorm: Any | None = None,
        *,
        restore_rng: bool = True,
    ) -> TrainingResumeState:
        """Restore training state in place from ``path``.

        Args:
            path: Checkpoint file.
            model / optimizer / scheduler / scaler / gradnorm: Optional objects
                whose state is loaded in-place.
            restore_rng: Restore the captured random state.

        Returns:
            :class:`TrainingResumeState` describing the restored run.
        """
        state = self._get_core().resume(
            path, model=model, optimizer=optimizer, scheduler=scheduler
        )

        extra = state.extra or {}
        scaler_state = extra.get("scaler_state")
        if scaler is not None and scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
            except (ValueError, RuntimeError, KeyError) as exc:
                raise CheckpointError(
                    f"failed to restore AMP scaler state: {exc}", detail=str(path)
                ) from exc

        gradnorm_state = extra.get("gradnorm_state")
        if gradnorm is not None and gradnorm_state is not None:
            gradnorm.load_state_dict(gradnorm_state)

        if restore_rng:
            rng_state = extra.get("rng_state")
            if rng_state:
                restore_rng_state(rng_state)

        return TrainingResumeState(
            path=state.path,
            epoch=state.epoch,
            step=state.step,
            metrics=state.metrics,
            metadata=extra,
            model_config=state.model_config,
        )

    def resume_latest(
        self,
        model: nn.Module | None = None,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        gradnorm: Any | None = None,
        *,
        restore_rng: bool = True,
    ) -> TrainingResumeState | None:
        """Resume from the most recent checkpoint, or ``None`` if absent."""
        path = self.latest_checkpoint()
        if path is None:
            return None
        return self.restore(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            gradnorm=gradnorm,
            restore_rng=restore_rng,
        )


# --------------------------------------------------------------------------- #
# Random-state helpers
# --------------------------------------------------------------------------- #


def capture_rng_state() -> dict[str, Any]:
    """Capture torch / numpy / python (and CUDA) random state.

    The numpy generator state is serialised into primitives so the resulting
    dict is safe to persist and reload with ``torch.load(weights_only=True)``.
    """
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": _serialize_numpy_state(np.random.get_state()),
        "python": random.getstate(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _serialize_numpy_state(state: tuple[Any, ...]) -> tuple[Any, ...]:
    """Convert ``np.random.get_state()`` into a primitive-only representation.

    ``keys`` is a NumPy array, which ``torch.load(weights_only=True)``
    rejects; the array is expanded to a plain list plus its dtype string.
    """
    name, keys, pos, has_gauss, cached_gaussian = state
    return (name, keys.tolist(), pos, has_gauss, cached_gaussian, str(keys.dtype))


def _deserialize_numpy_state(state: tuple[Any, ...]) -> tuple[Any, ...]:
    """Reverse :func:`_serialize_numpy_state` (raw state passes through)."""
    if len(state) == 6:
        name, keys_list, pos, has_gauss, cached_gaussian, dtype = state
        return (name, np.asarray(keys_list, dtype=dtype), pos, has_gauss, cached_gaussian)
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state dict produced by :func:`capture_rng_state`."""
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(_deserialize_numpy_state(state["numpy"]))
    if "python" in state:
        random.setstate(state["python"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            pass


def timestamp_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()
