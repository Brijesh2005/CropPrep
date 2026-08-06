"""Specialised training orchestration for the CropFusion model.

:class:`CropFusionTrainer` extends the base :class:`Trainer` with the Phase 7
training strategies:

* **Class-imbalance handling** — per-class weights derived from the training
  set's crop-label frequencies (``balanced`` / ``sqrt_inv`` / ``effective_num``)
  are threaded into the crop loss. No oversampling; focal loss is future work.
* **Curriculum training** — the :class:`~ai.training.curriculum.Curriculum`
  freezes / unfreezes model components across the five stages (tabular → image
  → temporal → fusion → finetune) with automatic transitions.
* **End-of-run reports** — the five report artefacts are generated from the
  training history (:mod:`ai.training.reports`).

Everything else (AMP, gradient handling, schedulers, checkpoints, callbacks,
resume) is inherited from :class:`Trainer`, so a ``CropFusionTrainer`` can be
dropped into any existing run by swapping the class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import TrainingCheckpointManager
from .config import TrainingConfig
from .curriculum import Curriculum, CurriculumCallback, build_curriculum
from .interfaces import Callback, SchedulerHandle
from .logger import ExperimentLogger
from .losses import MultiTaskLoss, build_class_weights
from .reports import default_reports_dir, generate_reports
from .trainer import Trainer, TrainingResult
from .validator import Validator


@dataclass
class CropFusionTrainingResult(TrainingResult):
    """Outcome of a :meth:`CropFusionTrainer.train` run."""

    #: Per-stage freeze reports (``stage`` / ``frozen`` / ``trainable``).
    stages: list[dict[str, Any]] = field(default_factory=list)
    #: Written report artefacts ``{report_type: path}``.
    reports: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        data = super().summary()
        data["stages"] = len(self.stages)
        data["reports"] = dict(self.reports)
        return data


def _infer_num_classes(model: nn.Module) -> int | None:
    """Best-effort crop class count from a CropFusion model config."""
    config = getattr(model, "config", None)
    heads = getattr(config, "heads", None) if config is not None else None
    crop = getattr(heads, "crop", None) if heads is not None else None
    num = getattr(crop, "num_classes", None)
    return int(num) if num else None


def _collect_class_counts(loader: Any, num_classes: int | None) -> torch.Tensor:
    """Count crop-label occurrences across a training loader (one pass)."""
    counts = torch.zeros(max(num_classes or 0, 0), dtype=torch.float32)
    for batch in loader:
        label = batch.get("crop_label") if isinstance(batch, Mapping) else None
        if label is None:
            continue
        label = torch.as_tensor(label, dtype=torch.long).reshape(-1)
        if label.numel() == 0:
            continue
        observed = int(label.max().item()) + 1
        if observed > counts.numel():
            counts = F.pad(counts, (0, observed - counts.numel()))
        counts += torch.bincount(label, minlength=observed).float()[:observed]
    return counts


class CropFusionTrainer(Trainer):
    """Train the CropFusion model with curriculum + class-frequency weighting.

    Args:
        model: The model to train (Phase 5 :class:`CropFusionModel`).
        train_loader: Training DataLoader (iterated once for class-frequency
            statistics when ``loss.class_weight_mode != "none"``).
        config: Validated :class:`TrainingConfig`.

    Keyword args mirror :class:`Trainer` (``val_loader``, ``loss_module``,
    ``optimizer``, ``scheduler_handle``, ``callbacks``, ``checkpoint_manager``,
    ``logger``, ``validator``, ``device``, ``input_map``).
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
        input_map: Callable[
            [Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any]]
        ]
        | None = None,
    ) -> None:
        # -- Class-frequency weights (no oversampling) --------------------- #
        self.class_frequency: torch.Tensor | None = None
        self.class_weights: torch.Tensor | None = None
        if config.loss.class_weight_mode != "none":
            num_classes = _infer_num_classes(model)
            counts = _collect_class_counts(train_loader, num_classes)
            weights = build_class_weights(config.loss, num_classes, counts)
            self.class_frequency = counts
            self.class_weights = weights
            if logger is not None:
                logger.info(
                    "class-frequency weights",
                    mode=config.loss.class_weight_mode,
                    counts=counts.tolist(),
                    weights=(
                        weights.tolist() if weights is not None else None
                    ),
                )

        if loss_module is None:
            loss_module = MultiTaskLoss(
                config.loss, class_weights={"crop": self.class_weights}
            )

        # -- Curriculum ---------------------------------------------------- #
        curriculum = build_curriculum(
            model, config.curriculum, num_epochs=config.train.epochs
        )
        self.curriculum = curriculum
        self.curriculum_callback: CurriculumCallback | None = None
        extra_callbacks: list[Callback] = []
        if config.curriculum.enabled:
            self.curriculum_callback = CurriculumCallback(curriculum)
            extra_callbacks.append(self.curriculum_callback)
        extra_callbacks.extend(list(callbacks or []))

        super().__init__(
            model,
            train_loader,
            config,
            val_loader=val_loader,
            loss_module=loss_module,
            optimizer=optimizer,
            scheduler_handle=scheduler_handle,
            callbacks=extra_callbacks,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            validator=validator,
            device=device,
            input_map=input_map,
        )
        # Move the loss (and its weight buffers) to the compute device.
        self.loss_module.to(self.device)

    # ------------------------------------------------------------------ #
    # Curriculum integration
    # ------------------------------------------------------------------ #

    def _run_epoch(self, epoch: int) -> dict[str, Any]:  # type: ignore[override]
        logs = super()._run_epoch(epoch)
        if (
            self.config.curriculum.enabled
            and self.config.curriculum.log_transitions
            and self.curriculum_callback is not None
            and self.curriculum_callback.current_stage is not None
        ):
            logs["stage"] = self.curriculum_callback.current_stage.name
        return logs

    def _finalize_result(  # type: ignore[override]
        self, *, duration: float, stopped_early: bool, best_epoch: int | None
    ) -> CropFusionTrainingResult:
        result = super()._finalize_result(
            duration=duration, stopped_early=stopped_early, best_epoch=best_epoch
        )
        stages = (
            list(self.curriculum_callback.stages_log)
            if self.curriculum_callback is not None
            else []
        )
        return CropFusionTrainingResult(
            epochs=result.epochs,
            steps=result.steps,
            history=result.history,
            best_metrics=result.best_metrics,
            best_path=result.best_path,
            stopped_early=result.stopped_early,
            duration_seconds=result.duration_seconds,
            best_epoch=result.best_epoch,
            stages=stages,
            reports={},
        )

    def train(self) -> CropFusionTrainingResult:  # type: ignore[override]
        """Run the training loop, then generate the end-of-run reports."""
        result = super().train()
        if self.config.general.reports:
            directory = default_reports_dir(self.config)
            reports = generate_reports(self.config, result, directory=directory)
            result.reports = {key: str(path) for key, path in reports.items()}
        return result


__all__ = ["CropFusionTrainer", "CropFusionTrainingResult"]
