"""Callback implementations for the training engine.

Callbacks observe the training lifecycle via the hooks defined in
:class:`~ai.training.interfaces.Callback`. The built-ins cover early stopping,
model checkpointing, learning-rate logging, console progress and optional
TensorBoard / Weights & Biases integration (both degrade gracefully when the
underlying package is not installed).
"""

from __future__ import annotations

import math
from typing import Any

from .interfaces import Callback
from .utils import is_primary


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric key from the epoch logs (e.g. ``val_loss``).
        mode: ``min`` (lower better) | ``max`` (higher better).
        patience: Epochs without improvement before stopping (``None`` disables).
        min_delta: Minimum improvement required to count as progress.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int | None = 10,
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best = -math.inf if mode == "max" else math.inf
        self.wait = 0
        self.best_epoch: int | None = None
        self.stopped_epoch: int | None = None

    @property
    def should_stop(self) -> bool:
        return self.stopped_epoch is not None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is None:
            logs = {}
        metric = logs.get(self.monitor)
        if metric is None:
            return
        score = float(metric)
        improved = (
            score < self.best - self.min_delta
            if self.mode == "min"
            else score > self.best + self.min_delta
        )
        if improved:
            self.best = score
            self.wait = 0
            self.best_epoch = epoch
        else:
            self.wait += 1
            if self.patience is not None and self.wait >= self.patience:
                self.stopped_epoch = epoch


class ModelCheckpoint(Callback):
    """Save best / latest / periodic checkpoints during training.

    Reads its objects from the bound trainer (set via :meth:`set_trainer`).
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        *,
        save_best: bool = True,
        save_latest: bool = True,
        save_periodic: int | None = None,
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_latest = save_latest
        self.save_periodic = save_periodic
        self.min_delta = float(min_delta)
        self.best = -math.inf if mode == "max" else math.inf
        self.best_path: Any | None = None
        self.best_metrics: dict[str, Any] = {}

    def set_trainer(self, trainer: Any) -> None:
        self.trainer = trainer

    def _metrics(self, logs: dict[str, Any]) -> dict[str, Any]:
        return dict(logs or {})

    def _save_kwargs(self, epoch: int, logs: dict[str, Any]) -> dict[str, Any]:
        trainer = self.trainer
        scheduler = (
            trainer.scheduler_handle.scheduler if trainer.scheduler_handle else None
        )
        return {
            "model": trainer.raw_model,
            "optimizer": trainer.optimizer,
            "scheduler": scheduler,
            "scaler": trainer.scaler,
            "gradnorm": trainer.gradnorm,
            "training_config": trainer.config,
            "epoch": epoch,
            "step": logs.get("step"),
            "metrics": self._metrics(logs),
        }

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if not is_primary():
            return
        if logs is None:
            logs = {}
        kwargs = self._save_kwargs(epoch, logs)

        if self.save_latest:
            self.trainer.checkpoint_manager.save_latest(**kwargs)

        if self.save_best:
            metric = logs.get(self.monitor)
            if metric is not None:
                score = float(metric)
                improved = (
                    score < self.best - self.min_delta
                    if self.mode == "min"
                    else score > self.best + self.min_delta
                )
                if improved:
                    self.best = score
                    self.best_metrics = dict(kwargs["metrics"])
                    self.best_path = self.trainer.checkpoint_manager.save_best(**kwargs)
                    logs["best_path"] = str(self.best_path)

        if self.save_periodic and epoch % self.save_periodic == 0:
            self.trainer.checkpoint_manager.save_periodic(**kwargs)


class LearningRateLogger(Callback):
    """Record the current learning rate into each batch / epoch log."""

    def __init__(self, group: int = 0) -> None:
        self.group = group

    def on_batch_end(self, step: int, logs: dict[str, Any] | None = None) -> None:
        if logs is None:
            logs = {}
        if self.trainer is not None and self.trainer.optimizer is not None:
            logs["lr"] = self._current_lr()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.on_batch_end(0, logs)

    def _current_lr(self) -> float:
        optimizer = self.trainer.optimizer
        scheduler = getattr(self.trainer, "scheduler_handle", None)
        if scheduler is not None and scheduler.scheduler is not None:
            try:
                # Max across param groups (discriminative-LR runs).
                return max(float(lr) for lr in scheduler.scheduler.get_last_lr())
            except Exception:
                pass
        return max(float(group["lr"]) for group in optimizer.param_groups)


class ConsoleLogger(Callback):
    """Print a compact summary at the end of each epoch."""

    def __init__(self, metrics: tuple[str, ...] = ("train_loss", "val_loss")) -> None:
        self.metrics = metrics

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        if is_primary():
            print("[training] run started")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if not is_primary():
            return
        if logs is None:
            logs = {}
        parts = [f"epoch {epoch + 1}"]
        for key in self.metrics:
            value = logs.get(key)
            if value is not None:
                parts.append(f"{key}={float(value):.4f}")
        for key in ("crop/accuracy", "crop/f1", "yield/rmse", "yield/mae"):
            value = logs.get(key)
            if value is not None:
                parts.append(f"{key}={float(value):.4f}")
        print("[training] " + "  ".join(parts))


class TensorBoardCallback(Callback):
    """Log scalars / histograms to TensorBoard (no-op when unavailable)."""

    def __init__(self, log_dir: str = "artifacts/training/tensorboard") -> None:
        self.log_dir = log_dir
        self.writer: Any | None = None

    def set_trainer(self, trainer: Any) -> None:
        self.trainer = trainer
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir)
        except Exception:
            self.writer = None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if self.writer is None or not is_primary():
            return
        for key, value in (logs or {}).items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, float(value), epoch)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class WandbCallback(Callback):
    """Log metrics to Weights & Biases (no-op when unavailable)."""

    def __init__(
        self,
        project: str = "cropfusion",
        entity: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.project = project
        self.entity = entity
        self.init_config = config or {}
        self.run: Any | None = None

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        if not is_primary():
            return
        try:
            import wandb  # type: ignore

            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                config=self.init_config,
                reinit=True,
            )
        except Exception:
            self.run = None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if self.run is None or not is_primary():
            return
        scalars = {
            key: float(value)
            for key, value in (logs or {}).items()
            if isinstance(value, (int, float))
        }
        self.run.log(scalars, step=epoch)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


class HistoryRecorder(Callback):
    """Accumulate per-epoch metrics for later reporting / visualization."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.history.append(dict(logs or {}))


class StagedFineTuning(Callback):
    """Progressive backbone unfreezing for the CropFusion image encoders.

    The image backbones (``ndvi_encoder`` / ``evi_encoder``) start fully
    frozen so only the randomly-initialised fusion / heads train first. At each
    configured epoch a stage unfreezes the matching backbone blocks (e.g.
    ``blocks.6``) so the pretrained trunk is fine-tuned from the top down at a
    low learning rate while early blocks / stem stay frozen.

    Frozen backbone modules are kept in ``eval()`` mode during training (via
    ``on_model_train_mode``) so their BatchNorm statistics and Dropout stay
    inert, mirroring the curriculum's freeze discipline.

    Note:
        Freezing happens *after* the optimizer is built, so param groups
        (``optimizer.backbone_lr_multiplier``) remain stable across stages —
        the optimizer simply receives no gradient for frozen parameters.
    """

    #: Must run on every rank (parameter state is not broadcast).
    all_ranks = True

    #: Module-name prefixes of the image encoders inside CropFusionModel.
    _ENCODER_ATTRIBUTES = ("ndvi_encoder", "evi_encoder")

    def __init__(self, schedule: list[dict[str, Any]] | None = None) -> None:
        # Normalise schedule entries {"epoch": int, "prefixes": [str]}.
        raw = list(schedule or [])
        self.schedule: list[dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            prefixes = [
                str(p) for p in entry.get("prefixes", []) if str(p).strip()
            ]
            try:
                epoch = int(entry.get("epoch", 0))
            except (TypeError, ValueError):
                continue
            self.schedule.append({"epoch": epoch, "prefixes": prefixes})
        self.schedule.sort(key=lambda s: s["epoch"])
        self._applied_epochs: set[int] = set()
        self._backbone_modules: list[tuple[str, Any]] = []
        self._stages_log: list[dict[str, Any]] = []

    def set_trainer(self, trainer: Any) -> None:
        self.trainer = trainer
        self.raw_model = trainer.raw_model
        self._collect_backbone_modules()

    # -- setup ------------------------------------------------------------- #

    def _collect_backbone_modules(self) -> None:
        modules: list[tuple[str, Any]] = []
        for name, module in self.raw_model.named_modules():
            if any(name == attr or name.startswith(f"{attr}.")
                   for attr in self._ENCODER_ATTRIBUTES):
                modules.append((name, module))
        self._backbone_modules = modules

    def _module_matches(self, module_name: str, prefix: str) -> bool:
        if module_name == prefix or module_name.startswith(f"{prefix}."):
            return True
        return f".{prefix}" in module_name or module_name.endswith(f".{prefix}")

    def _backbone_parameters(self) -> list[tuple[str, Any]]:
        params: list[tuple[str, Any]] = []
        for module_name, module in self._backbone_modules:
            for param_name, param in module.named_parameters(recurse=True):
                params.append((f"{module_name}.{param_name}", param))
        return params

    # -- lifecycle ----------------------------------------------------------- #

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        for module_name, module in self._backbone_modules:
            module.requires_grad_(False)
        self._stages_log.append(
            {
                "epoch": 0,
                "action": "freeze_all",
                "frozen": self._count_enabled(False),
                "trainable": self._count_enabled(True),
            }
        )

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        for stage in self.schedule:
            stage_epoch = stage["epoch"]
            if stage_epoch > epoch or stage_epoch in self._applied_epochs:
                continue
            prefixes = stage["prefixes"]
            if not prefixes:
                self._applied_epochs.add(stage_epoch)
                continue
            unfrozen = 0
            for module_name, module in self._backbone_modules:
                if any(self._module_matches(module_name, p) for p in prefixes):
                    module.requires_grad_(True)
                    unfrozen += 1
            self._applied_epochs.add(stage_epoch)
            self._stages_log.append(
                {
                    "epoch": stage_epoch,
                    "action": "unfreeze",
                    "prefixes": list(prefixes),
                    "modules_unfrozen": unfrozen,
                    "frozen": self._count_enabled(False),
                    "trainable": self._count_enabled(True),
                }
            )

    def on_model_train_mode(self) -> None:
        """Keep fully-frozen backbone modules in eval() after ``model.train()``.

        Called by the trainer at the start of every epoch *after* the model is
        switched to training mode, so BatchNorm statistics / Dropout of the
        still-frozen trunk stay inert while the unfrozen fine-tuned blocks
        train normally.
        """
        for module_name, module in self._backbone_modules:
            if module.training and not self._has_trainable_parameter(module):
                module.eval()

    # -- helpers ------------------------------------------------------------- #

    @staticmethod
    def _has_trainable_parameter(module: Any) -> bool:
        return any(p.requires_grad for p in module.parameters(recurse=True))

    def _count_enabled(self, enabled: bool) -> int:
        return sum(
            1
            for _name, param in self._backbone_parameters()
            if param.requires_grad is enabled
        )

    @property
    def stages_log(self) -> list[dict[str, Any]]:
        """Per-transition freeze/unfreeze records (for the run report)."""
        return list(self._stages_log)


class EarlyStopOnNan(Callback):
    """Halt the run when a NaN / Inf appears in the logged loss."""

    def __init__(self, monitor: str = "train_loss") -> None:
        self.monitor = monitor
        self.stopped_epoch: int | None = None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        value = (logs or {}).get(self.monitor)
        if value is not None and math.isnan(float(value)):
            self.stopped_epoch = epoch
