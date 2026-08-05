"""Evaluation metrics for the multi-task model.

Classification: accuracy, precision, recall, F1, top-K accuracy, ROC-AUC
(one-vs-rest) and the confusion matrix.

Regression: RMSE, MAE, MSE, R² and MAPE.

Values are accumulated as tensors across batches and reduced once, on the
primary rank (``torch.distributed`` aware via :mod:`training.training.utils`).
``sklearn`` provides the classification reductions where convenient; the
regression metrics are implemented directly so they stay fast and dependency
free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from .config import MetricsConfig
from .exceptions import MetricError
from .utils import all_gather_tensor, tensor_to_numpy


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def _predicted_class(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1).reshape(-1)


def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: MetricsConfig | None = None,
) -> dict[str, Any]:
    """Compute classification metrics from a batch of logits / labels.

    Returns a dict of metric name -> float (``None`` when undefined).
    """
    config = config or MetricsConfig()
    pred = _predicted_class(logits)
    labels = targets.reshape(-1)
    probs = torch.softmax(logits, dim=-1)

    accuracy = (pred == labels).float().mean().item()
    top_k_acc = _top_k_accuracy(probs, labels, k=config.top_k)

    pred_np = tensor_to_numpy(pred).astype(np.int64)
    labels_np = tensor_to_numpy(labels).astype(np.int64)

    result: dict[str, Any] = {
        "accuracy": float(accuracy),
        f"top{config.top_k}_accuracy": float(top_k_acc),
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "confusion_matrix": None,
        "support": int(labels.numel()),
    }

    if pred_np.size == 0 or labels_np.size == 0:
        return result

    num_classes = int(logits.size(-1))
    try:
        from sklearn.metrics import (
            confusion_matrix as _confusion_matrix,
            precision_recall_fscore_support,
            roc_auc_score,
        )
    except Exception:  # pragma: no cover - sklearn absent
        return result

    n_classes_present = int(np.unique(labels_np).size)
    if n_classes_present >= 1:
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_np, pred_np, average=config.average, zero_division=0
            )
            result["precision"] = float(precision)
            result["recall"] = float(recall)
            result["f1"] = float(f1)
        except Exception:
            pass
        try:
            result["confusion_matrix"] = _confusion_matrix(
                labels_np, pred_np, labels=list(range(num_classes))
            ).tolist()
        except Exception:
            pass

    if config.roc_auc and num_classes >= 2 and n_classes_present >= 2:
        proba_np = tensor_to_numpy(probs)
        try:
            result["roc_auc"] = float(
                roc_auc_score(
                    labels_np,
                    proba_np,
                    multi_class="ovr",
                    average="macro",
                    labels=list(range(num_classes)),
                )
            )
        except Exception:
            result["roc_auc"] = None

    return result


def _top_k_accuracy(probs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    if probs.size(1) < 2:
        return (probs.argmax(-1) == labels).float().mean().item()
    k = min(k, probs.size(1))
    top_k = probs.topk(k, dim=-1).indices
    return (top_k == labels.unsqueeze(1)).any(dim=-1).float().mean().item()


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #


def compute_regression_metrics(
    preds: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Compute regression metrics from predictions and targets."""
    pred = preds.reshape(-1).float()
    target = targets.reshape(-1).float()

    if pred.numel() == 0:
        raise MetricError("cannot compute regression metrics on empty batch")

    errors = target - pred
    mse = float((errors ** 2).mean().item())
    rmse = float(mse ** 0.5)
    mae = float(errors.abs().mean().item())

    # R² = 1 - SS_res / SS_tot (SS_tot == 0 -> constant target -> NaN).
    ss_res = float((errors ** 2).sum().item())
    ss_tot = float(((target - target.mean()) ** 2).sum().item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    # MAPE with a guard against zero/absent targets.
    denom = target.abs()
    mask = denom > 1e-8
    if bool(mask.sum().item() > 0):
        mape = float(
            ((errors.abs() / denom.clamp_min(1e-8)) * mask).sum().item()
            / mask.sum().item()
            * 100.0
        )
    else:
        mape = float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "support": int(pred.numel()),
    }


# --------------------------------------------------------------------------- #
# Accumulators
# --------------------------------------------------------------------------- #


@dataclass
class ClassificationAccumulator:
    """Streaming accumulator for classification metrics."""

    num_classes: int = 1
    config: MetricsConfig = field(default_factory=MetricsConfig)
    logits: list[torch.Tensor] = field(default_factory=list)
    targets: list[torch.Tensor] = field(default_factory=list)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self.logits.append(logits.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def reset(self) -> None:
        self.logits = []
        self.targets = []

    def result(self) -> dict[str, Any]:
        if not self.logits:
            return {}
        logits = torch.cat(self.logits, dim=0)
        targets = torch.cat(self.targets, dim=0)
        return compute_classification_metrics(logits, targets, self.config)


@dataclass
class RegressionAccumulator:
    """Streaming accumulator for regression metrics."""

    preds: list[torch.Tensor] = field(default_factory=list)
    targets: list[torch.Tensor] = field(default_factory=list)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def reset(self) -> None:
        self.preds = []
        self.targets = []

    def result(self) -> dict[str, float]:
        if not self.preds:
            return {}
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)
        return compute_regression_metrics(preds, targets)


# --------------------------------------------------------------------------- #
# Per-task tracker
# --------------------------------------------------------------------------- #


class MetricsTracker:
    """Tracks metrics per task name (``crop`` -> classification,
    ``yield`` -> regression).

    Args:
        tasks: Mapping of task name -> ``"classification"`` | ``"regression"``.
        config: Validated :class:`MetricsConfig`.
    """

    def __init__(
        self,
        tasks: Mapping[str, str],
        config: MetricsConfig | None = None,
    ) -> None:
        self.tasks = dict(tasks)
        self.config = config or MetricsConfig()
        self._accumulators: dict[str, Any] = {}
        for name, kind in self.tasks.items():
            if kind == "classification":
                self._accumulators[name] = ClassificationAccumulator(
                    num_classes=1, config=self.config
                )
            else:
                self._accumulators[name] = RegressionAccumulator()

    def update(
        self,
        task: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Accumulate one batch for ``task``."""
        if task not in self._accumulators:
            raise MetricError(f"unknown metric task {task!r}")
        self._accumulators[task].update(prediction, target)

    def reset(self) -> None:
        for accumulator in self._accumulators.values():
            accumulator.reset()

    def result(self) -> dict[str, Any]:
        """Per-task metric dicts, prefixed for logging (``crop/accuracy``)."""
        out: dict[str, Any] = {}
        for name, accumulator in self._accumulators.items():
            for key, value in accumulator.result().items():
                if key in {"confusion_matrix"}:
                    out[f"{name}/{key}"] = value
                    continue
                out[f"{name}/{key}"] = value
        return out

    def result_flat(self, prefix: str = "val") -> dict[str, Any]:
        """Flatten per-task metrics under a prefix (e.g. ``val/crop/accuracy``)."""
        out: dict[str, Any] = {}
        for name, accumulator in self._accumulators.items():
            for key, value in accumulator.result().items():
                if key == "confusion_matrix":
                    continue
                out[f"{prefix}/{name}/{key}"] = value
        return out


def task_kind(name: str) -> str:
    """Map a task name to its metric kind (``crop`` -> classification)."""
    return "classification" if name == "crop" else "regression"
