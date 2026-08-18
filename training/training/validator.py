"""Validation loop and model-validation strategies.

* :class:`Validator` — runs the model over a validation DataLoader in eval
  mode, computes the multi-task loss and the configured metrics, and returns a
  :class:`ValidationResult`.
* Cross-validation fold generators — ``holdout``, ``kfold``,
  ``stratified_kfold``, ``spatial`` and ``temporal``. Folds are generated at
  the observation level (STAM ``AgriculturalObservation`` objects), so each
  fold yields ``(train, val)`` observation subsets that feed the Phase 4
  dataset / preprocessor unchanged.
"""

from __future__ import annotations

import contextlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch import nn

from .config import MetricsConfig, ValidationConfig
from .exceptions import ValidationError
from .interfaces import FoldGenerator
from .metrics import MetricsTracker
from .utils import all_gather_tensor, is_distributed, resolve_device


# --------------------------------------------------------------------------- #
# Validation loop
# --------------------------------------------------------------------------- #


@dataclass
class ValidationResult:
    """Outcome of one validation pass."""

    metrics: dict[str, Any] = field(default_factory=dict)
    loss: float = 0.0
    per_task_losses: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    samples: int = 0

    @property
    def val_loss(self) -> float:
        return float(self.metrics.get("val_loss", self.loss))


class Validator:
    """Evaluation-loop engine (no parameter updates).

    Args:
        model: The model to validate.
        loss_module: A :class:`~ai.training.losses.MultiTaskLoss`.
        device: Compute device.
        metrics_config: Validated :class:`MetricsConfig`.
        amp: Enable autocast during validation (fp16 on CUDA).
        amp_dtype: ``float16`` | ``bfloat16``.
        input_map: Optional callable mapping a batch dict to ``(inputs, targets)``.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_module: nn.Module,
        device: torch.device | None = None,
        metrics_config: MetricsConfig | None = None,
        *,
        amp: bool = False,
        amp_dtype: str = "float16",
        input_map: Callable[[Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any]]]
        | None = None,
        nan_policy: str = "stop",
    ) -> None:
        self.model = model
        self.loss_module = loss_module
        self.device = device or resolve_device()
        self.metrics_config = metrics_config or MetricsConfig()
        self.amp = amp and torch.cuda.is_available()
        self.amp_dtype = amp_dtype
        self.input_map = input_map or self._default_input_map
        self.nan_policy = nan_policy

    # -- batch mapping ------------------------------------------------------ #

    @staticmethod
    def _default_input_map(batch: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        inputs = {k: batch[k] for k in ("tabular", "ndvi", "evi", "temporal_mask")
                  if k in batch}
        targets: dict[str, Any] = {}
        if "crop_label" in batch:
            targets["crop"] = batch["crop_label"]
        if "yield_label" in batch:
            targets["yield"] = batch["yield_label"]
        return inputs, targets

    # -- run ---------------------------------------------------------------- #

    @torch.no_grad()
    def validate(
        self,
        dataloader: Any,
        epoch: int | None = None,
    ) -> ValidationResult:
        """Run one validation pass and return aggregated metrics + loss."""
        from .utils import Timer

        model = self.model
        was_training = model.training
        model.eval()

        task_kinds: dict[str, str] = {}
        for name in getattr(self.loss_module, "tasks", {}):
            task_kinds[name] = "classification" if name == "crop" else "regression"
        tracker = MetricsTracker(task_kinds, config=self.metrics_config)

        loss_sum = 0.0
        loss_counts = 0
        per_task_sum: dict[str, float] = defaultdict(float)
        samples = 0
        timer = Timer().start()

        autocast = (
            torch.autocast("cuda", dtype=torch.float16 if self.amp_dtype == "float16"
                           else torch.bfloat16)
            if self.amp
            else contextlib.nullcontext()
        )

        for batch in dataloader:
            batch_size = self._batch_size(batch)
            batch = self._to_device(batch)
            inputs, targets = self.input_map(batch)
            with autocast:
                out = model(inputs)
            out_dict = self._outputs_to_dict(out)

            if self.loss_module is not None:
                _, per_task = self.loss_module(out_dict, targets)
                loss_value = torch.stack(list(per_task.values())).sum()
                loss_sum += float(loss_value.detach().item()) * batch_size
                for name, value in per_task.items():
                    per_task_sum[name] += float(value.detach().item()) * batch_size
                loss_counts += batch_size

            for name in task_kinds:
                if name not in out_dict or name not in targets:
                    continue
                tracker.update(name, out_dict[name].float(), targets[name])

            samples += batch_size

        duration = timer.stop()
        model.train(mode=was_training)

        metrics = tracker.result()
        if loss_counts > 0:
            metrics["val_loss"] = loss_sum / loss_counts
            metrics["val_per_task_loss"] = {
                name: total / loss_counts for name, total in per_task_sum.items()
            }

        # R5.2: surface NaN / Inf validation LOSS instead of silently writing
        # it into the run history (mirrors the trainer's fail-loudly NaN
        # handling). Per-metric NaN (e.g. R² on constant targets) is a
        # legitimate artifact and is NOT treated as a training failure.
        loss_bad = {
            name: value
            for name, value in metrics.items()
            if (name in ("val_loss",) or name.startswith("val_per_task_loss"))
            and isinstance(value, float)
            and not math.isfinite(value)
        }
        if loss_bad and getattr(self, "nan_policy", "stop") == "stop":
            raise ValidationError(
                "NaN/Inf in validation loss",
                detail={"epoch": epoch, "metrics": loss_bad},
            )

        return ValidationResult(
            metrics=metrics,
            loss=metrics.get("val_loss", 0.0),
            per_task_losses=metrics.get("val_per_task_loss", {}),
            duration_seconds=duration,
            samples=samples,
        )

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _batch_size(batch: Mapping[str, Any]) -> int:
        for key in ("tabular", "ndvi", "evi", "crop_label", "yield_label"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.size(0))
        return 0

    def _to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(self.device, non_blocking=True)
            else:
                out[key] = value
        return out

    def _outputs_to_dict(self, out: Any) -> dict[str, torch.Tensor]:
        if isinstance(out, dict):
            return dict(out)
        if hasattr(out, "as_dict"):
            raw = out.as_dict()
        else:
            raw = out
        result: dict[str, torch.Tensor] = {}
        mapping = {"crop_logits": "crop", "yield_pred": "yield"}
        for key, value in raw.items():
            if key in mapping and value is not None:
                result[mapping[key]] = value
            elif isinstance(value, torch.Tensor) and value.dim() > 0 and value.numel() > 0:
                result[key] = value
        return result


# --------------------------------------------------------------------------- #
# Fold generators
# --------------------------------------------------------------------------- #


def _group_key(observation: Any, column: str) -> str:
    admin = getattr(getattr(observation, "location", None), "admin", None)
    if admin is not None and getattr(admin, "village", None):
        return f"{column}:{admin.village}"
    fields = getattr(getattr(observation, "tabular", None), "fields", {}) or {}
    return f"{column}:{fields.get(column, 'unknown')}"


class HoldOutFoldGenerator(FoldGenerator):
    """Single (train, val) hold-out — no cross-validation."""

    strategy = "holdout"

    def __init__(self, seed: int = 42, val_ratio: float = 0.2) -> None:
        self.seed = seed
        self.val_ratio = val_ratio

    def folds(self, observations: Sequence[Any]) -> Iterator[tuple[list[Any], list[Any]]]:
        items = list(observations)
        rng = random.Random(self.seed)
        shuffled = list(items)
        rng.shuffle(shuffled)
        val_n = max(1, int(round(len(shuffled) * self.val_ratio)))
        yield shuffled[val_n:], shuffled[:val_n]


class KFoldFoldGenerator(FoldGenerator):
    """Random K-fold split at the observation level."""

    strategy = "kfold"

    def __init__(self, k: int = 5, seed: int = 42, shuffle: bool = True) -> None:
        self.k = k
        self.seed = seed
        self.shuffle = shuffle

    def folds(self, observations: Sequence[Any]) -> Iterator[tuple[list[Any], list[Any]]]:
        items = list(observations)
        rng = random.Random(self.seed)
        if self.shuffle:
            rng.shuffle(items)
        folds: list[list[Any]] = [[] for _ in range(self.k)]
        for index, item in enumerate(items):
            folds[index % self.k].append(item)
        for i in range(self.k):
            val = folds[i]
            train = [item for j, fold in enumerate(folds) if j != i for item in fold]
            yield train, val


class StratifiedKFoldFoldGenerator(FoldGenerator):
    """Stratified K-fold by the crop attribute (preserves class balance)."""

    strategy = "stratified_kfold"

    def __init__(self, k: int = 5, seed: int = 42, shuffle: bool = True) -> None:
        self.k = k
        self.seed = seed
        self.shuffle = shuffle

    def folds(self, observations: Sequence[Any]) -> Iterator[tuple[list[Any], list[Any]]]:
        by_class: dict[str, list[Any]] = defaultdict(list)
        for obs in observations:
            by_class[str(getattr(obs, "crop", None) or "unknown")].append(obs)

        fold_buckets: list[list[Any]] = [[] for _ in range(self.k)]
        for seed_offset, group in enumerate(by_class.values()):
            rng = random.Random(self.seed + seed_offset)
            if self.shuffle:
                rng.shuffle(group)
            for index, item in enumerate(group):
                fold_buckets[index % self.k].append(item)

        for i in range(self.k):
            val = fold_buckets[i]
            train = [item for j, fold in enumerate(fold_buckets) if j != i for item in fold]
            yield train, val


class SpatialFoldGenerator(FoldGenerator):
    """Group-based K-fold: whole villages/regions go to one fold (no leakage)."""

    strategy = "spatial"

    def __init__(self, k: int = 5, seed: int = 42, group_column: str = "village") -> None:
        self.k = k
        self.seed = seed
        self.group_column = group_column

    def folds(self, observations: Sequence[Any]) -> Iterator[tuple[list[Any], list[Any]]]:
        by_group: dict[str, list[Any]] = defaultdict(list)
        for obs in observations:
            by_group[_group_key(obs, self.group_column)].append(obs)
        groups = sorted(by_group)
        rng = random.Random(self.seed)
        rng.shuffle(groups)

        group_folds: list[list[str]] = [[] for _ in range(self.k)]
        for index, group in enumerate(groups):
            group_folds[index % self.k].append(group)

        for i in range(self.k):
            val = [obs for group in group_folds[i] for obs in by_group[group]]
            train_groups = [g for j, fold in enumerate(group_folds) if j != i for g in fold]
            train = [obs for group in train_groups for obs in by_group[group]]
            yield train, val


class TemporalFoldGenerator(FoldGenerator):
    """Year-based K-fold: whole years go to one fold (no temporal leakage)."""

    strategy = "temporal"

    def __init__(self, k: int = 5, seed: int = 42, temporal_column: str = "year") -> None:
        self.k = k
        self.seed = seed
        self.temporal_column = temporal_column

    def folds(self, observations: Sequence[Any]) -> Iterator[tuple[list[Any], list[Any]]]:
        by_year: dict[int, list[Any]] = defaultdict(list)
        for obs in observations:
            year = getattr(getattr(obs, "temporal", None), "year", None)
            if year is None:
                year = 0
            by_year[int(year)].append(obs)
        years = sorted(by_year)

        year_folds: list[list[int]] = [[] for _ in range(self.k)]
        for index, year in enumerate(years):
            year_folds[index % self.k].append(year)

        for i in range(self.k):
            val = [obs for year in year_folds[i] for obs in by_year[year]]
            train_years = [y for j, fold in enumerate(year_folds) if j != i for y in fold]
            train = [obs for year in train_years for obs in by_year[year]]
            yield train, val


def build_fold_generator(
    config: ValidationConfig | None = None,
    strategy: str | None = None,
) -> FoldGenerator:
    """Build a fold generator from a :class:`ValidationConfig` (or name)."""
    cfg = config or ValidationConfig()
    name = (strategy or cfg.strategy).strip().lower()
    if name == "holdout":
        return HoldOutFoldGenerator(seed=cfg.seed, val_ratio=0.2)
    if name == "kfold":
        return KFoldFoldGenerator(k=cfg.k_folds, seed=cfg.seed, shuffle=cfg.shuffle)
    if name == "stratified_kfold":
        return StratifiedKFoldFoldGenerator(
            k=cfg.k_folds, seed=cfg.seed, shuffle=cfg.shuffle
        )
    if name == "spatial":
        return SpatialFoldGenerator(
            k=cfg.k_folds, seed=cfg.seed, group_column=cfg.group_column
        )
    if name == "temporal":
        return TemporalFoldGenerator(
            k=cfg.k_folds, seed=cfg.seed, temporal_column=cfg.temporal_column
        )
    raise ValidationError(f"unknown validation strategy {name!r}")


def cross_validation_splits(
    observations: Sequence[Any],
    config: ValidationConfig | None = None,
) -> list[tuple[list[Any], list[Any]]]:
    """Materialize all ``(train, val)`` splits for a validation strategy."""
    generator = build_fold_generator(config)
    return list(generator.folds(observations))
