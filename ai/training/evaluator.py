"""Final evaluation of a trained model.

:class:`Evaluator` runs a test pass and reports:

* per-task metrics (classification + regression),
* the combined multi-task score,
* the confusion matrix (classification),
* inference latency percentiles and throughput,
* GPU / CPU memory usage,
* parameter count and model size,
* collected predictions / targets and the shared representation (for
  visualization and feature-distribution plots).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import contextlib

import numpy as np
import torch
from torch import nn

from .config import MetricsConfig
from .metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    task_kind,
)
from .utils import (
    Timer,
    all_gather_tensor,
    count_parameters,
    estimate_parameter_memory,
    resolve_device,
    tensor_to_numpy,
)


@dataclass
class EvaluationResult:
    """Full evaluation output for one model."""

    metrics: dict[str, Any] = field(default_factory=dict)
    per_task_loss: dict[str, float] = field(default_factory=dict)
    confusion_matrix: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    throughput: dict[str, float] = field(default_factory=dict)
    memory: dict[str, float] = field(default_factory=dict)
    multi_task_score: float = 0.0
    predictions: dict[str, Any] = field(default_factory=dict)
    feature_embeddings: np.ndarray | None = None
    feature_labels: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "metrics": self.metrics,
            "per_task_loss": self.per_task_loss,
            "confusion_matrix": self.confusion_matrix,
            "latency_ms": self.latency,
            "throughput": self.throughput,
            "memory": self.memory,
            "multi_task_score": self.multi_task_score,
        }
        if self.predictions:
            out["predictions"] = {k: v.tolist() if hasattr(v, "tolist") else v
                                  for k, v in self.predictions.items()}
        return out


class Evaluator:
    """Evaluate a trained model over a test loader.

    Args:
        model: The trained model.
        device: Compute device.
        metrics_config: Validated :class:`MetricsConfig`.
        input_map: Callable mapping a batch to ``(inputs, targets)``.
        amp: Enable autocast during evaluation (fp16 on CUDA).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        metrics_config: MetricsConfig | None = None,
        *,
        input_map: Callable[[Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any]]]
        | None = None,
        amp: bool = False,
    ) -> None:
        self.model = model
        self.device = device or resolve_device()
        self.metrics_config = metrics_config or MetricsConfig()
        self.amp = amp and torch.cuda.is_available()
        self.input_map = input_map or self._default_input_map

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

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: Any,
        loss_module: nn.Module | None = None,
        *,
        collect_embeddings: bool = True,
    ) -> EvaluationResult:
        """Run the test pass and aggregate metrics + artifacts."""
        model = self.model
        was_training = model.training
        model.eval()

        tasks: dict[str, str] = {}
        if loss_module is not None:
            for name in getattr(loss_module, "tasks", {}):
                tasks[name] = task_kind(name)
        else:
            tasks = {"crop": "classification", "yield": "regression"}

        collected: dict[str, list[torch.Tensor]] = {"crop": [], "yield": []}
        target_accum: dict[str, list[torch.Tensor]] = {"crop": [], "yield": []}
        loss_sum = 0.0
        loss_counts = 0
        per_task_sum: dict[str, float] = {}
        embeddings: list[torch.Tensor] = []
        embedding_labels: list[torch.Tensor] = []

        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
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
            if loss_module is not None:
                _, per_task = loss_module(out_dict, targets)
                loss_sum += float(torch.stack(list(per_task.values())).sum().item()) * batch_size
                loss_counts += batch_size
                for name, value in per_task.items():
                    per_task_sum[name] = per_task_sum.get(name, 0.0) + float(
                        value.detach().item()
                    ) * batch_size

            for name in tasks:
                if name in out_dict and name in targets:
                    collected[name].append(out_dict[name].float().detach().cpu())
                    target_accum[name].append(targets[name].detach().cpu())

            if collect_embeddings:
                shared = getattr(out, "shared_representation", None)
                if shared is None and isinstance(out, dict):
                    shared = out.get("shared_representation")
                if shared is not None:
                    embeddings.append(shared.float().detach().cpu())
                    if "crop" in targets:
                        embedding_labels.append(targets["crop"].long().detach().cpu())

        model.train(mode=was_training)

        # -- Metrics ----------------------------------------------------- #
        metrics: dict[str, Any] = {}
        confusion: dict[str, Any] = {}
        predictions: dict[str, Any] = {}
        for name, kind in tasks.items():
            if not collected.get(name):
                continue
            logits = torch.cat(collected[name], dim=0)
            targets_flat = torch.cat(target_accum[name], dim=0)
            if kind == "classification":
                result = compute_classification_metrics(
                    logits, targets_flat, self.metrics_config
                )
                for key, value in result.items():
                    metrics[f"{name}/{key}"] = value
                confusion[name] = result.get("confusion_matrix")
                predictions[f"{name}_pred"] = tensor_to_numpy(
                    logits.argmax(dim=-1)
                )
            else:
                result = compute_regression_metrics(logits, targets_flat)
                for key, value in result.items():
                    metrics[f"{name}/{key}"] = value
                predictions[f"{name}_pred"] = tensor_to_numpy(logits.reshape(-1))
            predictions[f"{name}_target"] = tensor_to_numpy(targets_flat.reshape(-1))

        per_task_loss = (
            {name: total / max(loss_counts, 1) for name, total in per_task_sum.items()}
            if loss_counts > 0
            else {}
        )
        metrics["test_loss"] = loss_sum / max(loss_counts, 1) if loss_counts > 0 else 0.0

        # -- Multi-task score --------------------------------------------- #
        multi_task_score = self._multi_task_score(metrics, tasks, predictions)

        # -- Feature embeddings ------------------------------------------- #
        emb = None
        emb_labels = None
        if collect_embeddings and embeddings:
            emb = torch.cat(embeddings, dim=0).numpy()
            if embedding_labels:
                emb_labels = torch.cat(embedding_labels, dim=0).numpy()

        return EvaluationResult(
            metrics=metrics,
            per_task_loss=per_task_loss,
            confusion_matrix=confusion,
            predictions=predictions,
            feature_embeddings=emb,
            feature_labels=emb_labels,
            multi_task_score=self._multi_task_score(metrics, tasks, predictions),
        )

    # ------------------------------------------------------------------ #
    # Benchmark
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def benchmark(
        self,
        sample_batch: Mapping[str, torch.Tensor],
        *,
        iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Measure inference latency and resource usage.

        Args:
            sample_batch: A batch matching the model's input contract
                (e.g. ``model.sample_batch(...)``).
            iterations: Passes used for the latency distribution.
            warmup_iterations: Passes discarded before timing.

        Returns:
            ``(latency, memory, throughput)`` dicts. Latency keys:
            ``mean_ms``, ``p50_ms``, ``p95_ms``, ``p99_ms``. Memory keys:
            ``parameters``, ``model_size_mb``, ``gpu_memory_mb``,
            ``cpu_rss_mb``. Throughput keys: ``samples_per_second``,
            ``batches_per_second``.
        """
        model = self.model
        was_training = model.training
        model.eval()

        batch = self._to_device(sample_batch)
        inputs, _ = self.input_map(batch)

        for _ in range(warmup_iterations):
            model(inputs)

        latencies: list[float] = []
        timer = Timer().start()
        for _ in range(iterations):
            start = time.perf_counter()
            model(inputs)
            latencies.append((time.perf_counter() - start) * 1000.0)
        total_s = timer.stop()

        latencies = np.asarray(latencies)
        throughput = {
            "samples_per_second": (
                iterations * int(batch_size_from(inputs)) / total_s if total_s > 0 else 0.0
            ),
            "batches_per_second": iterations / total_s if total_s > 0 else 0.0,
        }

        latency = {
            "mean_ms": float(latencies.mean()),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }

        memory = {
            "parameters": int(count_parameters(model)),
            "model_size_mb": float(estimate_parameter_memory(model) / (1024 ** 2)),
            "gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                if torch.cuda.is_available()
                else 0.0
            ),
            "cpu_rss_mb": _cpu_rss_mb(),
        }

        model.train(mode=was_training)
        return latency, memory, throughput

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _multi_task_score(
        self,
        metrics: dict[str, Any],
        tasks: dict[str, str],
        predictions: dict[str, Any],
    ) -> float:
        """Weighted blend of classification accuracy and regression quality."""
        crop_score = 1.0
        if "crop" in tasks and metrics.get("crop/accuracy") is not None:
            crop_score = float(metrics["crop/accuracy"])

        yield_score = 1.0
        if "yield" in tasks:
            rmse = metrics.get("yield/rmse")
            if rmse is not None:
                targets = predictions.get("yield_target")
                std = float(np.std(targets)) if targets is not None and len(targets) > 1 else 1.0
                nrmse = float(rmse) / max(std, 1e-6)
                yield_score = max(0.0, min(1.0, 1.0 - nrmse))
        return float(0.5 * crop_score + 0.5 * yield_score)

    @staticmethod
    def _batch_size(batch: Mapping[str, Any]) -> int:
        for key in ("tabular", "ndvi", "evi", "crop_label"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.size(0))
        return 1

    def _to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}

    @staticmethod
    def _outputs_to_dict(out: Any) -> dict[str, torch.Tensor]:
        if isinstance(out, dict):
            return dict(out)
        raw = out.as_dict() if hasattr(out, "as_dict") else {}
        result: dict[str, torch.Tensor] = {}
        mapping = {"crop_logits": "crop", "yield_pred": "yield"}
        for key, value in raw.items():
            if key in mapping and value is not None:
                result[mapping[key]] = value
        return result


def batch_size_from(inputs: Mapping[str, torch.Tensor]) -> int:
    for value in inputs.values():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            return int(value.size(0))
    return 1


def _cpu_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 ** 2))
    except Exception:
        return 0.0
