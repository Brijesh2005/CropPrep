"""Checkpoint manager — save / load / resume / partial loading.

Stores a single ``.pt`` file per checkpoint containing the model state dict,
the serialized model config, training metadata (epoch / step / metrics) and
optional optimizer / scheduler state (for Phase 6 resume).

All loading is delegated to the caller-supplied model instance so the manager
stays a pure persistence layer. Partial loading filters state-dict keys by
regex include / exclude patterns (e.g. loading a pretrained encoder into a
different model).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .exceptions import CheckpointError
from .utils import filter_state_dict

FORMAT_VERSION = 1
_CHECKPOINT_GLOB = "checkpoint_*.pt"


def _json_safe(value: Any) -> Any:
    """Convert a value to primitives so ``weights_only=True`` can unpickle it.

    ``torch.__version__`` is a ``TorchVersion`` (str subclass) that the safe
    loader does not allow; ``default=str`` coerces it (and any other exotic
    leaf) to a plain string.
    """
    return json.loads(json.dumps(value, default=str, sort_keys=True))


@dataclass
class LoadReport:
    """Outcome of loading a state dict into a model."""

    loaded_keys: int
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys


@dataclass
class ResumeState:
    """What :meth:`CheckpointManager.resume` returns."""

    path: Path
    epoch: int | None
    step: int | None
    metrics: dict[str, Any]
    extra: dict[str, Any]
    model_config: dict[str, Any] | None
    model: nn.Module | None
    optimizer: Any | None
    scheduler: Any | None


class CheckpointManager:
    """Persists and restores model / optimizer / scheduler state.

    Args:
        directory: Directory checkpoints are written to (created on demand).
        keep_last: Number of most-recent checkpoints to retain (``None`` =
            keep every checkpoint).
    """

    def __init__(self, directory: str | Path, keep_last: int | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

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
        extra: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> Path:
        """Write a checkpoint and return its path.

        Args:
            model: The model to persist (its ``config`` is stored too when
                present).
            optimizer / scheduler: Optional objects to include for resume.
            epoch / step / metrics / extra: Training metadata.
            name: Explicit file name (default ``checkpoint_epoch<NNN>.pt``).
        """
        state: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "extra": extra or {},
            "model_config": (
                # JSON mode keeps Path fields as primitives so checkpoints load
                # with torch.load(weights_only=True).
                model.config.model_dump(mode="json") if hasattr(model, "config") else None
            ),
            #: Architecture name + schema version + runtime metadata so the
            #: factory can rebuild the right class and validate compatibility.
            "architecture": (
                getattr(getattr(model, "config", None), "name", None)
            ),
            "architecture_version": (
                getattr(getattr(model, "config", None), "architecture_version", None)
            ),
            "metadata": (
                _json_safe(dict(model.metadata)) if hasattr(model, "metadata") else None
            ),
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        if name is None:
            if epoch is not None:
                name = f"checkpoint_epoch{int(epoch):04d}.pt"
            elif step is not None:
                name = f"checkpoint_step{int(step):06d}.pt"
            else:
                name = "checkpoint.pt"

        path = self.directory / name
        try:
            torch.save(state, path)
        except (OSError, RuntimeError) as exc:
            raise CheckpointError(
                f"failed to write checkpoint: {exc}", detail=str(path)
            ) from exc
        self._prune()
        return path

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Read a checkpoint dict from disk."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise CheckpointError(
                f"checkpoint not found: {checkpoint_path}", detail=str(path)
            )
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CheckpointError(
                f"failed to read checkpoint: {exc}", detail=str(path)
            ) from exc
        if not isinstance(state, dict) or "model_state_dict" not in state:
            raise CheckpointError(
                "checkpoint is missing 'model_state_dict'", detail=str(path)
            )
        return state

    @staticmethod
    def load_state_into(
        model: nn.Module, path: str | Path, *, strict: bool = True
    ) -> LoadReport:
        """Load a checkpoint's weights into ``model``.

        Args:
            model: Destination model.
            path: Checkpoint file.
            strict: When ``False``, loading tolerates missing / unexpected
                keys (partial loading).

        Returns:
            :class:`LoadReport` describing the result.
        """
        state = CheckpointManager.load(path)
        model_state = state["model_state_dict"]
        if strict:
            model.load_state_dict(model_state, strict=True)
            return LoadReport(loaded_keys=len(model_state))
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        return LoadReport(
            loaded_keys=len(model_state) - len(unexpected),
            missing_keys=list(missing),
            unexpected_keys=list(unexpected),
        )

    @staticmethod
    def partial_load(
        model: nn.Module,
        path: str | Path,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> LoadReport:
        """Load a filtered subset of a checkpoint's weights.

        Args:
            model: Destination model.
            path: Checkpoint file.
            include: Regex patterns; only matching keys are loaded.
            exclude: Regex patterns; matching keys are skipped.

        Returns:
            :class:`LoadReport` with missing / unexpected keys.
        """
        state = CheckpointManager.load(path)
        filtered = filter_state_dict(
            state["model_state_dict"], include=include, exclude=exclude
        )
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        return LoadReport(
            loaded_keys=len(filtered) - len(unexpected),
            missing_keys=list(missing),
            unexpected_keys=list(unexpected),
        )

    def resume(
        self,
        path: str | Path,
        model: nn.Module | None = None,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
    ) -> ResumeState:
        """Restore training state for a resumed run.

        Args:
            path: Checkpoint file.
            model / optimizer / scheduler: Optional objects whose state is
                loaded in-place (model weights are loaded strictly).

        Returns:
            :class:`ResumeState` with the training metadata.
        """
        state = self.load(path)
        if model is not None:
            model.load_state_dict(state["model_state_dict"], strict=True)
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        return ResumeState(
            path=Path(path),
            epoch=state.get("epoch"),
            step=state.get("step"),
            metrics=dict(state.get("metrics", {})),
            extra=dict(state.get("extra", {})),
            model_config=state.get("model_config"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _prune(self) -> None:
        if self.keep_last is None:
            return
        checkpoints = sorted(
            self.directory.glob(_CHECKPOINT_GLOB), key=lambda p: p.stat().st_mtime
        )
        for stale in checkpoints[:- self.keep_last]:
            try:
                stale.unlink()
            except OSError:
                continue
