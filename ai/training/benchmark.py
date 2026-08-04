"""Inference / training benchmark.

:class:`Benchmark` measures training speed, validation speed, inference speed,
GPU / CPU memory and model size for a trained model. Reports are plain dicts
suitable for JSON logging and the completion report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import nn

from .config import MetricsConfig
from .evaluator import Evaluator
from .losses import MultiTaskLoss
from .optimizers import build_optimizer
from .utils import Timer, count_parameters, estimate_parameter_memory, resolve_device


@dataclass
class BenchmarkReport:
    """Result of a :class:`Benchmark` run."""

    training: dict[str, float] = field(default_factory=dict)
    validation: dict[str, float] = field(default_factory=dict)
    inference: dict[str, float] = field(default_factory=dict)
    memory: dict[str, float] = field(default_factory=dict)
    model: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training": self.training,
            "validation": self.validation,
            "inference": self.inference,
            "memory": self.memory,
            "model": self.model,
        }


class Benchmark:
    """Measure training / validation / inference throughput and resources.

    Args:
        model: The model to benchmark.
        device: Compute device.
        batch_size: Samples per benchmark batch.
        iterations: Timed passes for each measurement.
        warmup_iterations: Passes discarded before timing.
        metrics_config: Validated :class:`MetricsConfig`.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        *,
        batch_size: int = 32,
        iterations: int = 50,
        warmup_iterations: int = 5,
        metrics_config: MetricsConfig | None = None,
    ) -> None:
        self.model = model
        self.device = device or resolve_device()
        self.batch_size = batch_size
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.metrics_config = metrics_config or MetricsConfig()
        self.evaluator = Evaluator(model, device=self.device, metrics_config=self.metrics_config)

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #

    def run(
        self,
        train_loader: Any | None = None,
        val_loader: Any | None = None,
        loss_module: MultiTaskLoss | None = None,
        *,
        measure_training: bool = True,
        measure_inference: bool = True,
        sample_batch: Mapping[str, torch.Tensor] | None = None,
    ) -> BenchmarkReport:
        """Run every enabled measurement and return a :class:`BenchmarkReport`."""
        report = BenchmarkReport()
        report.model = self._measure_model()

        if sample_batch is None and hasattr(self.model, "sample_batch"):
            sample_batch = self.model.sample_batch(batch_size=self.batch_size)

        if measure_inference and sample_batch is not None:
            latency, memory, throughput = self.evaluator.benchmark(
                sample_batch,
                iterations=self.iterations,
                warmup_iterations=self.warmup_iterations,
            )
            report.inference = {**latency, **throughput}
            report.memory = memory

        if measure_training and train_loader is not None:
            loss_module = loss_module or MultiTaskLoss()
            report.training = self._measure_training(train_loader, loss_module)
            if val_loader is not None:
                report.validation = self._measure_validation(val_loader, loss_module)

        return report

    # ------------------------------------------------------------------ #
    # Measurements
    # ------------------------------------------------------------------ #

    def _measure_model(self) -> dict[str, float]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return {
            "parameters": float(count_parameters(self.model)),
            "trainable_parameters": float(sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )),
            "model_size_mb": float(estimate_parameter_memory(self.model) / (1024 ** 2)),
        }

    def _measure_training(
        self, loader: Any, loss_module: MultiTaskLoss
    ) -> dict[str, float]:
        model = self.model
        was_training = model.training
        model.train()
        optimizer = build_optimizer(model)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        samples = 0
        timer = Timer().start()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            if index >= self.iterations:
                break
            batch = self._to_device(batch)
            inputs, targets = self._split(batch)
            out = model(inputs)
            out_dict = self._outputs(out)
            total, _ = loss_module(out_dict, targets)
            total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            samples += self._batch_size(batch)
        elapsed = timer.stop()
        model.train(mode=was_training)

        result: dict[str, float] = {
            "samples_per_second": samples / elapsed if elapsed > 0 else 0.0,
            "batches_per_second": samples / (self.batch_size * max(elapsed, 1e-9)),
            "elapsed_seconds": elapsed,
        }
        if torch.cuda.is_available():
            result["peak_gpu_memory_mb"] = float(
                torch.cuda.max_memory_allocated() / (1024 ** 2)
            )
        return result

    def _measure_validation(
        self, loader: Any, loss_module: MultiTaskLoss
    ) -> dict[str, float]:
        model = self.model
        was_training = model.training
        model.eval()
        samples = 0
        timer = Timer().start()
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= self.iterations:
                    break
                batch = self._to_device(batch)
                inputs, targets = self._split(batch)
                model(inputs)
                samples += self._batch_size(batch)
        elapsed = timer.stop()
        model.train(mode=was_training)
        return {
            "samples_per_second": samples / elapsed if elapsed > 0 else 0.0,
            "elapsed_seconds": elapsed,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}

    @staticmethod
    def _split(batch: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        inputs = {k: batch[k] for k in ("tabular", "ndvi", "evi", "temporal_mask")
                  if k in batch}
        targets: dict[str, Any] = {}
        if "crop_label" in batch:
            targets["crop"] = batch["crop_label"]
        if "yield_label" in batch:
            targets["yield"] = batch["yield_label"]
        return inputs, targets

    @staticmethod
    def _outputs(out: Any) -> dict[str, torch.Tensor]:
        if isinstance(out, dict):
            return dict(out)
        raw = out.as_dict() if hasattr(out, "as_dict") else {}
        result: dict[str, torch.Tensor] = {}
        mapping = {"crop_logits": "crop", "yield_pred": "yield"}
        for key, value in raw.items():
            if key in mapping and value is not None:
                result[mapping[key]] = value
        return result

    @staticmethod
    def _batch_size(batch: Mapping[str, Any]) -> int:
        for key in ("tabular", "ndvi", "evi", "crop_label"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.size(0))
        return 1
