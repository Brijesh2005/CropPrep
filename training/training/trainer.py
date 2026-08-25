"""The training engine.

:class:`Trainer` orchestrates the whole training loop: mixed precision,
gradient clipping / accumulation / checkpointing, NaN detection, learning-rate
scheduling, validation, early stopping, resume and checkpointing — all wired
through the :class:`~ai.training.config.TrainingConfig`.

The trainer is a plain object (not a framework): every collaborator (model,
loaders, loss, optimizer, scheduler, validator, checkpoint manager, callbacks,
logger) is injected. Nothing is read from disk inside the loop — the Phase 4
preprocessing layer feeds ready-made tensors.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from .callbacks import HistoryRecorder
from .checkpoint import TrainingCheckpointManager
from .config import TrainingConfig
from .exceptions import TrainingRunError
from .interfaces import Callback, SchedulerHandle
from .logger import ExperimentLogger
from .losses import GradNormController, MultiTaskLoss
from .optimizers import build_optimizer
from .schedulers import build_scheduler, get_lr
from .utils import (
    Timer,
    all_gather_tensor,
    apply_gradient_checkpointing,
    broadcast_dict,
    configure_determinism,
    get_world_size,
    is_distributed,
    is_primary,
    named_enabled_parameters,
    resolve_device,
    to_device,
)
from .validator import Validator


@dataclass
class TrainingResult:
    """Outcome of a :meth:`Trainer.train` run."""

    epochs: int
    steps: int
    history: list[dict[str, Any]] = field(default_factory=list)
    best_metrics: dict[str, Any] = field(default_factory=dict)
    best_path: str | None = None
    stopped_early: bool = False
    duration_seconds: float = 0.0
    best_epoch: int | None = None
    nan_steps: int = 0
    nan_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "steps": self.steps,
            "best_metrics": self.best_metrics,
            "best_path": self.best_path,
            "stopped_early": self.stopped_early,
            "duration_seconds": self.duration_seconds,
            "best_epoch": self.best_epoch,
            "nan_steps": self.nan_steps,
            "nan_diagnostics": self.nan_diagnostics,
        }


def _default_input_map(
    batch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a Phase 4 batch into model inputs and task targets.

    R5.2.2: passes ``yield_unit_mask`` through to targets so the yield loss
    only receives physical kg/ha observations (NPP-proxy excluded).
    """
    inputs = {k: batch[k] for k in ("tabular", "ndvi", "evi", "temporal_mask")
              if k in batch}
    targets: dict[str, Any] = {}
    if "crop_label" in batch:
        targets["crop"] = batch["crop_label"]
    if "yield_label" in batch:
        targets["yield"] = batch["yield_label"]
    if "yield_unit_mask" in batch:
        targets["yield_unit_mask"] = batch["yield_unit_mask"]
    return inputs, targets


def _outputs_to_dict(out: Any) -> dict[str, torch.Tensor]:
    """Extract task outputs from the model output (dict or CropFusionOutput)."""
    if isinstance(out, dict):
        return dict(out)
    raw = out.as_dict() if hasattr(out, "as_dict") else {}
    result: dict[str, torch.Tensor] = {}
    mapping = {"crop_logits": "crop", "yield_pred": "yield"}
    for key, value in raw.items():
        if key in mapping and value is not None:
            result[mapping[key]] = value
    return result


def _has_nan_or_inf(value: torch.Tensor) -> bool:
    return bool(torch.isnan(value).any().item() or torch.isinf(value).any().item())


def _nan_attrs(value: torch.Tensor) -> dict[str, Any]:
    """Per-tensor NaN/Inf counts (for diagnostics)."""
    value = value.detach().float()
    return {
        "nan": int(torch.isnan(value).sum().item()),
        "inf": int(torch.isinf(value).sum().item()),
        "min": float(value.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).min().item()),
        "max": float(value.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).max().item()),
    }


class Trainer:
    """Train a CropFusion model.

    Args:
        model: The model to train (Phase 5 :class:`CropFusionModel`).
        train_loader: Training DataLoader.
        config: Validated :class:`TrainingConfig`.
        val_loader: Optional validation DataLoader.
        loss_module: A :class:`MultiTaskLoss` (built from ``config`` when None).
        optimizer: Pre-built optimizer (built from ``config`` when None).
        scheduler_handle: Pre-built :class:`SchedulerHandle` (built when None).
        callbacks: Extra callbacks (built-ins are added automatically).
        checkpoint_manager: :class:`TrainingCheckpointManager` (built when None).
        logger: :class:`ExperimentLogger` (built when None).
        validator: :class:`Validator` (built when None).
        device: Compute device (resolved from ``config`` when None).
        input_map: Callable mapping a batch dict to ``(inputs, targets)``.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        config: TrainingConfig,
        *,
        val_loader: Any | None = None,
        loss_module: MultiTaskLoss | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler_handle: SchedulerHandle | None = None,
        callbacks: Sequence[Callback] | None = None,
        checkpoint_manager: TrainingCheckpointManager | None = None,
        logger: ExperimentLogger | None = None,
        validator: Validator | None = None,
        device: torch.device | None = None,
        input_map: Callable[[Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any]]]
        | None = None,
    ) -> None:
        self.config = config
        self.raw_model = model
        self.device = device or resolve_device(config.general.device)
        # Place the model on the compute device *before* the optimizer / DDP /
        # validator are built so every consumer sees device-resident parameters
        # (previously CUDA inputs hit CPU weights on first forward).
        self.raw_model.to(self.device)
        self.input_map = input_map or _default_input_map

        # torch.compile (optional) — wraps the raw model before any
        # distributed wrapper so parameters / state_dict stay shared.
        if config.general.compile:
            if not hasattr(torch, "compile"):
                raise TrainingRunError(
                    "torch.compile requested but unavailable in this PyTorch build",
                    detail=torch.__version__,
                )
            from training.models.runtime import compile_model as _compile_model

            self.raw_model = _compile_model(
                model,
                mode=config.general.compile_mode,
                backend=config.general.compile_backend,
            )

        # Distributed wrapper (no-op on single process).
        self.model: nn.Module = self.raw_model
        if is_distributed() and get_world_size() > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.raw_model,
                device_ids=None if self.device.type == "cpu" else [self.device],
            )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_module = loss_module or MultiTaskLoss(config.loss)
        self.loss_module.to(self.device)
        self.optimizer = optimizer or build_optimizer(self.raw_model, config.optimizer)

        self.steps_per_epoch = max(1, len(train_loader))
        self.scheduler_handle = scheduler_handle or build_scheduler(
            self.optimizer,
            config.scheduler,
            steps_per_epoch=self.steps_per_epoch,
            total_epochs=config.train.epochs,
        )

        # AMP scaler — only meaningful on CUDA (graceful CPU fallback).
        self.amp = config.general.amp and torch.cuda.is_available()
        self.scaler: Any = None
        if self.amp:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)

        # Gradient checkpointing (memory trade-off).
        apply_gradient_checkpointing(self.raw_model, config.general.gradient_checkpointing)

        self.validator = validator or (
            Validator(
                self.raw_model,
                self.loss_module,
                device=self.device,
                metrics_config=config.metrics,
                amp=self.amp,
                amp_dtype=config.general.amp_dtype,
                input_map=self.input_map,
                nan_policy=config.general.nan_policy,
            )
            if val_loader is not None
            else None
        )

        self.checkpoint_manager = checkpoint_manager or TrainingCheckpointManager(
            config.checkpoint.directory, keep_last=config.checkpoint.keep_last
        )
        self.logger = logger

        # GradNorm controller (optional).
        self.gradnorm: GradNormController | None = None
        if config.loss.weighting_mode == "gradnorm":
            self.gradnorm = GradNormController(
                self.raw_model,
                self.loss_module,
                alpha=config.loss.gradnorm_alpha,
            )

        self.callbacks = self._build_callbacks(list(callbacks or []))
        for callback in self.callbacks:
            callback.set_trainer(self)

        self.epoch = 0
        self.global_step = 0
        self.history: list[dict[str, Any]] = []
        # NaN diagnostics (R5.2): every detected NaN step is recorded with its
        # source tensors so instability is never silently hidden by ``skip``.
        self.nan_steps = 0
        self.nan_diagnostics: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _build_callbacks(self, callbacks: list[Callback]) -> list[Callback]:
        cfg = self.config
        built: list[Callback] = []

        from .callbacks import (
            ConsoleLogger,
            EarlyStopping,
            EarlyStopOnNan,
            HistoryRecorder,
            LearningRateLogger,
            ModelCheckpoint,
            TensorBoardCallback,
            WandbCallback,
        )

        built.append(HistoryRecorder())
        built.append(LearningRateLogger())
        built.append(
            ModelCheckpoint(
                monitor=cfg.train.early_stopping_metric,
                mode=cfg.train.early_stopping_mode,
                save_best=cfg.checkpoint.save_best,
                save_latest=cfg.checkpoint.save_latest,
                save_periodic=cfg.checkpoint.save_periodic,
            )
        )
        built.append(
            EarlyStopping(
                monitor=cfg.train.early_stopping_metric,
                mode=cfg.train.early_stopping_mode,
                patience=cfg.train.early_stopping_patience,
                min_delta=cfg.train.early_stopping_min_delta,
            )
        )
        if cfg.general.nan_policy == "stop":
            built.append(EarlyStopOnNan(monitor="train_loss"))
        if cfg.logging.console:
            built.append(ConsoleLogger())
        if cfg.logging.tensorboard:
            built.append(TensorBoardCallback(log_dir=cfg.logging.tensorboard_dir))
        if cfg.logging.wandb:
            built.append(
                WandbCallback(
                    project=cfg.logging.wandb_project,
                    entity=cfg.logging.wandb_entity,
                )
            )
        # User-supplied callbacks run last.
        built.extend(callbacks)
        return built

    def _resolve_callbacks(self) -> list[Callback]:
        return self.callbacks

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #

    def _try_resume(self) -> None:
        cfg = self.config
        if not cfg.checkpoint.resume:
            return
        resume_path = cfg.checkpoint.resume_path
        state = None
        if resume_path:
            state = self.checkpoint_manager.restore(
                resume_path,
                model=self.raw_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler_handle.scheduler if self.scheduler_handle else None,
                scaler=self.scaler,
                gradnorm=self.gradnorm,
            )
        else:
            state = self.checkpoint_manager.resume_latest(
                model=self.raw_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler_handle.scheduler if self.scheduler_handle else None,
                scaler=self.scaler,
                gradnorm=self.gradnorm,
            )
        if state is not None:
            # The checkpoint stores the last *completed* epoch, so the next
            # epoch to run is epoch + 1.
            self.epoch = int(state.epoch or 0) + 1
            self.global_step = int(state.step or 0)
            if self.logger is not None:
                self.logger.info(
                    "resumed training",
                    epoch=self.epoch, step=self.global_step, path=str(state.path),
                )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self) -> TrainingResult:
        """Run the full training loop and return a :class:`TrainingResult`."""
        cfg = self.config
        configure_determinism(cfg.general.deterministic, cfg.general.seed)
        self._try_resume()

        self._fire("on_train_begin")
        timer = Timer().start()
        stopped_early = False
        early_stopper = self._early_stopping()
        best_epoch: int | None = None

        try:
            for epoch in range(self.epoch, cfg.train.epochs):
                self.epoch = epoch
                self._fire("on_epoch_begin", epoch)

                epoch_logs = self._run_epoch(epoch)

                # Validation + scheduler (epoch-period) + checkpoints.
                if self.validator is not None and (
                    (epoch + 1) % cfg.general.validation_frequency == 0
                ):
                    val_result = self.validator.validate(self.val_loader, epoch=epoch)
                    epoch_logs.update(val_result.metrics)
                    epoch_logs["val_duration"] = val_result.duration_seconds
                    if self.scheduler_handle is not None and self.scheduler_handle.requires_metric:
                        metric = epoch_logs.get(self.scheduler_handle.monitor_metric)
                        if metric is None:
                            metric = val_result.loss
                        self.scheduler_handle.step(metric)
                elif self.scheduler_handle is not None and self.scheduler_handle.step_period == "epoch":
                    self.scheduler_handle.step()

                epoch_logs["lr"] = self._current_lr()
                self.history.append(dict(epoch_logs))

                self._fire("on_epoch_end", epoch, epoch_logs)
                if self.logger is not None:
                    self.logger.log_epoch(epoch + 1, epoch_logs)

                if self._early_stopping_triggered(epoch_logs, early_stopper):
                    stopped_early = True
                    best_epoch = early_stopper.best_epoch
                    if self.logger is not None:
                        self.logger.info(
                            "early stopping", epoch=epoch + 1, best=early_stopper.best
                        )
                    break
        except Exception as exc:
            self._fire("on_exception", exc)
            raise

        duration = timer.stop()
        self._fire("on_train_end")

        result = self._finalize_result(
            duration=duration,
            stopped_early=stopped_early,
            best_epoch=best_epoch,
        )
        return result

    # ------------------------------------------------------------------ #
    # One epoch
    # ------------------------------------------------------------------ #

    def _run_epoch(self, epoch: int) -> dict[str, Any]:
        cfg = self.config
        general = cfg.general
        model = self.model
        model.train()
        self._on_model_train_mode()

        running_loss = 0.0
        running_samples = 0
        per_task_sum: dict[str, float] = {}
        step_in_epoch = 0
        logs: dict[str, Any] = {"epoch": epoch + 1}

        autocast = (
            torch.autocast("cuda", dtype=torch.float16 if general.amp_dtype == "float16"
                           else torch.bfloat16)
            if self.amp
            else contextlib.nullcontext()
        )

        self.optimizer.zero_grad(set_to_none=True)
        accum = general.gradient_accumulation_steps
        num_batches = max(1, len(self.train_loader))

        for batch_index, batch in enumerate(self.train_loader):
            batch_size = self._batch_size(batch)
            batch = to_device(batch, self.device)
            inputs, targets = self.input_map(batch)

            with autocast:
                out = model(inputs)
                out_dict = _outputs_to_dict(out)
                if self.gradnorm is not None:
                    per_task = self.loss_module.per_task_losses(out_dict, targets)
                    self.gradnorm.apply(per_task)
                    total, per_task = self.loss_module.combine(per_task)
                else:
                    total, per_task = self.loss_module(out_dict, targets)
                loss_scaled = total / accum

            if self.scaler is not None:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            running_loss += float(total.detach().item()) * batch_size
            running_samples += batch_size
            for name, value in per_task.items():
                per_task_sum[name] = per_task_sum.get(name, 0.0) + float(
                    value.detach().item()
                ) * batch_size

            # NaN / Inf detection.
            nan_detected = general.nan_detection and self._nan_in_state(total.detach())
            if nan_detected:
                diagnostics = self._collect_nan_diagnostics(
                    epoch, batch_index, total, per_task, inputs, targets
                )
                self.nan_steps += 1
                self.nan_diagnostics.append(diagnostics)
                if general.nan_policy == "stop":
                    self._fire("on_exception", TrainingRunError("NaN detected in training state"))
                    raise TrainingRunError(
                        "NaN detected in training state; aborting (nan_policy=stop)",
                        detail=diagnostics,
                    )
                if self.logger is not None:
                    self.logger.warning(
                        "NaN detected; skipping step",
                        epoch=epoch,
                        step=batch_index,
                        nan_steps=self.nan_steps,
                        **{k: v for k, v in diagnostics.items() if k in
                           ("loss", "per_task_losses", "inputs", "grad_params")},
                    )

            last_batch = batch_index == num_batches - 1
            if nan_detected and general.nan_policy in ("warn", "skip"):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if (batch_index + 1) % accum == 0 or last_batch:
                if general.gradient_clip is not None:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    params = self.raw_model.parameters()
                    if general.gradient_clip_type == "value":
                        torch.nn.utils.clip_grad_value_(
                            params, general.gradient_clip
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            params, general.gradient_clip
                        )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                step_in_epoch += 1

                if self.scheduler_handle is not None and self.scheduler_handle.step_period == "step":
                    self.scheduler_handle.step()

                if step_in_epoch % general.log_every == 0:
                    logs.update(
                        {
                            "step": self.global_step,
                            "train_loss": running_loss / max(running_samples, 1),
                            "lr": self._current_lr(),
                        }
                    )
                    for name, value in per_task_sum.items():
                        logs[f"train/{name}_loss"] = value / max(running_samples, 1)
                    self._fire("on_batch_begin", self.global_step, logs)
                    self._fire("on_batch_end", self.global_step, logs)

        logs["train_loss"] = running_loss / max(running_samples, 1)
        for name, value in per_task_sum.items():
            logs[f"train/{name}_loss"] = value / max(running_samples, 1)
        logs["train_samples"] = running_samples
        return logs

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _on_model_train_mode(self) -> None:
        """Hook fired after ``model.train()`` at the start of every epoch.

        Subclasses / callbacks use it to re-apply per-epoch module states that
        ``model.train()`` would otherwise reset — e.g. the curriculum keeps
        frozen modules in ``eval()`` mode so their BatchNorm statistics and
        Dropout stay inert.
        """
        for callback in self.callbacks:
            method = getattr(callback, "on_model_train_mode", None)
            if method is not None and (
                getattr(callback, "all_ranks", False) or is_primary()
            ):
                method()

    @staticmethod
    def _batch_size(batch: Mapping[str, Any]) -> int:
        for key in ("tabular", "ndvi", "evi", "crop_label"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.size(0))
        return 1

    def _nan_in_state(self, loss: torch.Tensor) -> bool:
        if _has_nan_or_inf(loss):
            return True
        for _name, param in named_enabled_parameters(self.raw_model):
            if param.grad is not None and _has_nan_or_inf(param.grad):
                return True
        return False

    def _collect_nan_diagnostics(
        self,
        epoch: int,
        step: int,
        total: torch.Tensor,
        per_task: Mapping[str, torch.Tensor],
        inputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Capture which tensor went non-finite and the surrounding context."""
        diagnostics: dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "loss": _nan_attrs(total),
            "per_task_losses": {
                name: _nan_attrs(value) for name, value in per_task.items()
            },
            "inputs": {
                key: _nan_attrs(value)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            },
            "targets": {
                key: _nan_attrs(value)
                for key, value in targets.items()
                if isinstance(value, torch.Tensor)
            },
            "grad_params": [],
        }
        for name, param in named_enabled_parameters(self.raw_model):
            if param.grad is not None and _has_nan_or_inf(param.grad):
                diagnostics["grad_params"].append(
                    {"name": name, **_nan_attrs(param.grad)}
                )
        return diagnostics

    def _current_lr(self) -> float:
        if self.scheduler_handle is not None:
            try:
                return float(self.scheduler_handle.get_last_lr()[0])
            except Exception:
                pass
        return float(self.optimizer.param_groups[0]["lr"])

    def _early_stopping(self):
        for callback in self.callbacks:
            if callback.__class__.__name__ == "EarlyStopping":
                return callback
        return None

    def _early_stopping_triggered(self, logs: dict[str, Any], early_stopper) -> bool:
        if early_stopper is None:
            return False
        if early_stopper.stopped_epoch is not None:
            return True
        return False

    def _finalize_result(
        self, *, duration: float, stopped_early: bool, best_epoch: int | None
    ) -> TrainingResult:
        best_metrics: dict[str, Any] = {}
        best_path: str | None = None
        for callback in self.callbacks:
            if callback.__class__.__name__ == "ModelCheckpoint":
                if callback.best_path is not None:
                    best_path = str(callback.best_path)
                best_metrics = dict(callback.best_metrics)
                break
        if self.logger is not None:
            summary = {
                "epochs": len(self.history),
                "steps": self.global_step,
                "stopped_early": stopped_early,
                "duration_seconds": duration,
                "best_path": best_path,
            }
            self.logger.finalize(summary)
        return TrainingResult(
            epochs=len(self.history),
            steps=self.global_step,
            history=self.history,
            best_metrics=best_metrics,
            best_path=best_path,
            stopped_early=stopped_early,
            duration_seconds=duration,
            best_epoch=best_epoch,
            nan_steps=self.nan_steps,
            nan_diagnostics=list(self.nan_diagnostics),
        )

    # ------------------------------------------------------------------ #
    # Callback dispatch
    # ------------------------------------------------------------------ #

    def _fire(self, hook: str, *args: Any, **kwargs: Any) -> None:
        for callback in self.callbacks:
            if getattr(callback, "all_ranks", False) or is_primary():
                method = getattr(callback, hook, None)
                if method is not None:
                    method(*args, **kwargs)
