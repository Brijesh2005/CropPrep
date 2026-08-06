"""Multimodal evaluation driver (Phase R5).

:class:`MultimodalEvaluator` runs a trained :class:`CropFusionModel` over a
Phase-4-style loader in evaluation mode and reduces per-task extended metrics,
PR curves, confusion matrices, raw predictions, shared embeddings and
forward-pass latency into a single :class:`EvaluationOutcome`.

The driver is device-agnostic (CPU / CUDA), seed-stable and keeps the model in
``eval()`` with gradients disabled — the same contract used at inference time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from training.models import CropFusionModel

from .config import EvaluationConfig
from .exceptions import EvaluationError
from .metrics import (
    EvaluationAccumulator,
    compute_pr_curves,
)

_TENSOR_KEYS = ("tabular", "ndvi", "evi", "temporal_mask")


@dataclass
class EvaluationOutcome:
    """Result of one evaluation pass.

    Attributes:
        metrics: Per-task extended metrics (``crop`` / ``yield``).
        pr_curves: Per-task precision-recall curves (classification only).
        predictions: Per-task ``{targets, preds, probs}`` numpy arrays.
        embeddings: Collected shared representations ``[N, D]`` (``None`` when
            disabled or when the pass collected none).
        latency_ms: Forward-pass latency statistics (``mean`` / ``p50`` /
            ``p95`` in milliseconds).
        num_samples: Number of evaluated samples.
        per_class_tables: Per-task per-class rows (classification only).
    """

    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    pr_curves: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    predictions: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    embeddings: np.ndarray | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    per_class_tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    gates: dict[str, np.ndarray] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "pr_curves": self.pr_curves,
            "latency_ms": self.latency_ms,
            "num_samples": self.num_samples,
        }


class MultimodalEvaluator:
    """Evaluate a CropFusion model over a loader.

    Args:
        model: A trained :class:`CropFusionModel`.
        config: Validated :class:`EvaluationConfig` (``None`` = defaults).
        device: Override the configured device (``cpu`` / ``cuda``).
    """

    def __init__(
        self,
        model: CropFusionModel,
        config: EvaluationConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.config = config or EvaluationConfig()
        self.device = device or self.config.general.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tasks = self._discover_tasks()

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def _discover_tasks(self) -> dict[str, str]:
        """Map task name -> metric kind from the model configuration."""
        config = getattr(self.model, "config", None)
        tasks: dict[str, str] = {}
        if config is not None:
            if getattr(config, "crop_enabled", False):
                tasks["crop"] = "classification"
            if getattr(config, "yield_enabled", False):
                tasks["yield"] = "regression"
        if not tasks:
            tasks = {"crop": "classification", "yield": "regression"}
        return tasks

    @property
    def task_names(self) -> list[str]:
        return list(self._tasks)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(self, loader: Any) -> EvaluationOutcome:
        """Evaluate over ``loader`` and return an :class:`EvaluationOutcome`.

        ``loader`` yields Phase-4 batch dicts with the model inputs plus
        ``crop_label`` / ``yield_label`` target keys.

        Raises:
            EvaluationError: When evaluation produces no samples.
        """
        self.model.eval()
        accumulators = {
            name: EvaluationAccumulator(self.config.metrics)
            for name in self._tasks
        }
        embeddings: list[np.ndarray] = []
        gates: dict[str, list[np.ndarray]] = {}
        latencies: list[float] = []
        n_samples = 0

        for batch in loader:
            device_batch = self._to_device(batch)
            start = time.perf_counter()
            out = self.model(device_batch)
            latencies.append((time.perf_counter() - start) * 1000.0)

            batch_size = self._batch_size(device_batch)
            n_samples += batch_size
            for name, kind in self._tasks.items():
                if kind == "classification":
                    logits = out.crop_logits
                    labels = device_batch.get("crop_label")
                else:
                    logits = None
                    preds = out.yield_pred
                    labels = device_batch.get("yield_label")
                if logits is not None or preds is not None:
                    accumulators[name].update(
                        logits,
                        None if logits is not None else preds,
                        labels,
                    )

            if (
                self.config.general.collect_embeddings
                and out.shared_representation is not None
            ):
                embeddings.append(
                    out.shared_representation.detach().cpu().float().numpy()
                )

            model_gates = getattr(out, "gates", {}) or {}
            for gate_name, gate_values in model_gates.items():
                values = gate_values.detach().cpu().float().numpy().reshape(-1)
                gates.setdefault(gate_name, []).append(values)

        if n_samples == 0:
            raise EvaluationError(
                "evaluation produced no samples — check the loader",
                detail={"tasks": self._tasks},
            )

        outcome = EvaluationOutcome(num_samples=n_samples)
        for name, kind in self._tasks.items():
            acc = accumulators[name]
            if acc.empty:
                continue
            outcome.metrics[name] = acc.result(kind)
            outcome.predictions[name] = acc.predictions(kind)
            if kind == "classification" and acc.logits:
                logits = torch.cat(acc.logits, dim=0)
                targets = torch.cat(acc.targets, dim=0)
                try:
                    outcome.pr_curves[name] = compute_pr_curves(logits, targets)
                except Exception:
                    outcome.pr_curves[name] = []
                per_class = outcome.metrics[name].get("per_class", [])
                if per_class:
                    outcome.per_class_tables[name] = per_class

        if embeddings:
            outcome.embeddings = np.concatenate(embeddings, axis=0)

        if gates:
            outcome.gates = {
                name: np.concatenate(values, axis=0) for name, values in gates.items()
            }

        outcome.latency_ms = {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        }
        return outcome

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Move model-input tensors to the evaluation device."""
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if key in _TENSOR_KEYS and isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    @staticmethod
    def _batch_size(batch: Mapping[str, Any]) -> int:
        for key in _TENSOR_KEYS:
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.size(0))
        for key in ("crop_label", "yield_label"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.size(0))
        return 0
