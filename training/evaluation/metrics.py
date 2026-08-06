"""Evaluation metrics for the multi-task CropFusion model (Phase R5).

Extends the training-time metrics with the deeper diagnostic surface required
for model evaluation:

* classification — balanced accuracy, per-class precision / recall / F1,
  one-vs-rest ROC-AUC and average precision (AUPRC), confusion matrix;
* regression — median absolute error, signed bias, max absolute error,
  percentiles of the absolute-error distribution, within-tolerance fraction
  and a prediction-error histogram.

All reductions are implemented against numpy (sklearn where convenient) so the
package stays self-contained and reproducible for exported / deployed models.
Values are computed from accumulated logits / predictions / targets, so an
evaluation pass can stream batches then reduce once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from .config import MetricsConfig
from .exceptions import MetricComputationError


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: MetricsConfig | None = None,
) -> dict[str, Any]:
    """Compute extended classification metrics from logits / labels.

    Args:
        logits: ``[N, C]`` logits.
        targets: ``[N]`` integer labels.
        config: Metric settings (``None`` = defaults).

    Returns:
        A dict with scalar metrics plus a ``per_class`` list and the confusion
        matrix.
    """
    config = config or MetricsConfig()
    probs = torch.softmax(logits.detach().float(), dim=-1)
    pred = probs.argmax(dim=-1).reshape(-1)
    labels = targets.detach().reshape(-1)

    pred_np = _to_numpy(pred).astype(np.int64)
    labels_np = _to_numpy(labels).astype(np.int64)
    proba_np = _to_numpy(probs)
    num_classes = int(logits.size(-1))

    accuracy = float((pred == labels).float().mean().item())
    top_k_acc = float(_top_k_accuracy(probs, labels, k=config.top_k))

    result: dict[str, Any] = {
        "accuracy": accuracy,
        f"top{config.top_k}_accuracy": top_k_acc,
        "balanced_accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "auprc": None,
        "confusion_matrix": None,
        "per_class": [],
        "support": int(labels.numel()),
    }

    if pred_np.size == 0 or labels_np.size == 0:
        return result

    try:
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            confusion_matrix as _confusion_matrix,
            precision_recall_fscore_support,
            roc_auc_score,
        )
    except Exception:  # pragma: no cover - sklearn absent
        return result

    n_present = int(np.unique(labels_np).size)
    if n_present >= 1:
        try:
            result["balanced_accuracy"] = float(balanced_accuracy_score(labels_np, pred_np))
        except Exception:
            pass
        try:
            precision, recall, f1, support = precision_recall_fscore_support(
                labels_np, pred_np, average=None, zero_division=0,
                labels=list(range(num_classes)),
            )
            result["per_class"] = [
                {
                    "class": int(cls),
                    "precision": float(precision[cls]),
                    "recall": float(recall[cls]),
                    "f1": float(f1[cls]),
                    "support": int(support[cls]),
                }
                for cls in range(num_classes)
            ]
        except Exception:
            pass
        if result["per_class"]:
            avg = config.average
            if avg == "macro":
                keys = ("precision", "recall", "f1")
                for key in keys:
                    result[key] = float(
                        np.mean([row[key] for row in result["per_class"]])
                    )
            elif avg == "weighted":
                supports = [row["support"] for row in result["per_class"]]
                total = sum(supports) or 1
                for key in ("precision", "recall", "f1"):
                    result[key] = float(
                        np.average([row[key] for row in result["per_class"]],
                                   weights=supports)
                    )
            else:  # micro == global accuracy
                for key in ("precision", "recall", "f1"):
                    result[key] = accuracy
        try:
            result["confusion_matrix"] = _confusion_matrix(
                labels_np, pred_np, labels=list(range(num_classes))
            ).tolist()
        except Exception:
            pass

    if config.roc_auc and num_classes >= 2 and n_present >= 2:
        try:
            result["roc_auc"] = float(
                roc_auc_score(
                    labels_np, proba_np, multi_class="ovr",
                    average="macro", labels=list(range(num_classes)),
                )
            )
        except Exception:
            result["roc_auc"] = None

    if config.pr_curves and num_classes >= 2 and n_present >= 2:
        try:
            result["auprc"] = float(
                average_precision_score(
                    labels_np, proba_np, average="macro",
                )
            )
        except Exception:
            result["auprc"] = None

    return result


def compute_pr_curves(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> list[dict[str, Any]]:
    """Precision-recall curves (one entry per class, one-vs-rest).

    Each entry carries ``class``, ``precision``, ``recall`` and ``threshold``
    arrays (decimated to at most 100 points).
    """
    probs = torch.softmax(logits.detach().float(), dim=-1)
    labels = _to_numpy(targets.detach().reshape(-1)).astype(np.int64)
    proba = _to_numpy(probs)
    num_classes = int(logits.size(-1))

    try:
        from sklearn.metrics import precision_recall_curve
    except Exception as exc:  # pragma: no cover - sklearn absent
        raise MetricComputationError(
            "PR curves require sklearn", detail=exc
        ) from exc

    curves: list[dict[str, Any]] = []
    for cls in range(num_classes):
        binary = (labels == cls).astype(np.int64)
        precision, recall, thresholds = precision_recall_curve(binary, proba[:, cls])
        # Decimate to keep reports light.
        step = max(1, len(precision) // 100)
        curves.append(
            {
                "class": cls,
                "precision": precision[::step].tolist(),
                "recall": recall[::step].tolist(),
                "threshold": thresholds[::step].tolist(),
            }
        )
    return curves


def _top_k_accuracy(probs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    if probs.size(1) < 2:
        return float((probs.argmax(-1) == labels).float().mean().item())
    k = min(k, probs.size(1))
    top_k = probs.topk(k, dim=-1).indices
    return float((top_k == labels.unsqueeze(1)).any(dim=-1).float().mean().item())


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #


def compute_regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    config: MetricsConfig | None = None,
) -> dict[str, Any]:
    """Compute extended regression metrics from predictions / targets.

    Adds median absolute error, signed bias, max absolute error, absolute-error
    percentiles, within-tolerance fraction and the error histogram to the
    standard RMSE / MAE / MSE / R2 / MAPE set.
    """
    config = config or MetricsConfig()
    pred = preds.detach().reshape(-1).float()
    target = targets.detach().reshape(-1).float()

    if pred.numel() == 0:
        raise MetricComputationError(
            "cannot compute regression metrics on empty batch"
        )

    errors = target - pred
    abs_errors = errors.abs()
    mse = float((errors ** 2).mean().item())
    rmse = float(mse ** 0.5)
    mae = float(abs_errors.mean().item())
    median_ae = float(abs_errors.median().item())
    bias = float(errors.mean().item())
    max_ae = float(abs_errors.max().item())

    ss_res = float((errors ** 2).sum().item())
    ss_tot = float(((target - target.mean()) ** 2).sum().item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    denom = target.abs()
    mask = denom > 1e-8
    if bool(mask.sum().item() > 0):
        mape = float(
            ((abs_errors / denom.clamp_min(1e-8)) * mask).sum().item()
            / mask.sum().item() * 100.0
        )
    else:
        mape = float("nan")

    percentiles = config.error_percentiles
    error_quantiles = np.percentile(
        _to_numpy(abs_errors), [p * 100.0 for p in percentiles]
    ).tolist()
    within_tol = float(
        (abs_errors <= (target.abs() * config.tolerance_fraction))
        .float().mean().item()
    )

    counts, edges = np.histogram(
        _to_numpy(errors), bins=config.histogram_bins
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "median_absolute_error": median_ae,
        "bias": bias,
        "max_absolute_error": max_ae,
        "r2": r2,
        "mape": mape,
        "within_tolerance": within_tol,
        "error_percentiles": {
            f"p{int(p * 100)}": float(value)
            for p, value in zip(percentiles, error_quantiles)
        },
        "error_histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
        "support": int(pred.numel()),
    }


# --------------------------------------------------------------------------- #
# Accumulator
# --------------------------------------------------------------------------- #


@dataclass
class EvaluationAccumulator:
    """Streaming accumulator for extended per-task metrics.

    Stores logits / predictions / targets per task and reduces them once via
    :meth:`result`. ``num_classes`` is required for classification tasks.
    """

    config: MetricsConfig = field(default_factory=MetricsConfig)
    logits: list[torch.Tensor] = field(default_factory=list)
    preds: list[torch.Tensor] = field(default_factory=list)
    targets: list[torch.Tensor] = field(default_factory=list)

    def update(self, logits: torch.Tensor | None, preds: torch.Tensor | None,
               targets: torch.Tensor) -> None:
        """Accumulate one batch (logits xor preds, plus targets)."""
        targets = targets.detach().cpu()
        if logits is not None:
            self.logits.append(logits.detach().cpu())
            self.preds.append(logits.detach().cpu().argmax(dim=-1))
        elif preds is not None:
            self.preds.append(preds.detach().cpu())
        self.targets.append(targets)

    def reset(self) -> None:
        self.logits = []
        self.preds = []
        self.targets = []

    @property
    def empty(self) -> bool:
        return not self.targets

    def result(self, kind: str) -> dict[str, Any]:
        """Reduced metrics for ``kind`` (``classification`` | ``regression``)."""
        if self.empty:
            return {}
        targets = torch.cat(self.targets, dim=0)
        if kind == "classification":
            if not self.logits:
                return {}
            logits = torch.cat(self.logits, dim=0)
            return compute_classification_metrics(logits, targets, self.config)
        preds = torch.cat(self.preds, dim=0)
        return compute_regression_metrics(preds, targets, self.config)

    def predictions(self, kind: str) -> dict[str, np.ndarray]:
        """Raw arrays for downstream error analysis."""
        if self.empty:
            return {}
        targets = torch.cat(self.targets, dim=0).reshape(-1)
        out: dict[str, np.ndarray] = {
            "targets": _to_numpy(targets),
        }
        if kind == "classification" and self.logits:
            logits = torch.cat(self.logits, dim=0)
            out["probs"] = torch.softmax(logits.detach().float(), dim=-1).numpy()
            out["preds"] = np.asarray(out["probs"]).argmax(-1).astype(np.int64)
        elif self.preds:
            out["preds"] = _to_numpy(torch.cat(self.preds, dim=0))
        return out
