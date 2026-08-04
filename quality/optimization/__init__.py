"""Inference optimization toolkit (Phase 11).

Wrappers that accelerate a trained :class:`~ai.models.cropfusion.CropFusionModel`
at inference time without touching the architecture:

* eager — the baseline forward pass,
* autocast — fp16 / bf16 mixed-precision context,
* compiled — ``torch.compile`` (inductor) with a graceful fallback,
* onnx — ONNX Runtime session exported via :class:`ai.models.exporter.ModelExporter`.

Every wrapper implements the same ``predict`` interface and returns plain
numpy arrays, so swapping runtimes is a one-line change and every runtime can
be cross-validated against the eager baseline.
"""

from __future__ import annotations

from .batch import BatchInferenceEngine
from .benchmark import OptimizationBenchmark
from .runtime import OptimizedRuntime, run_autocast

__all__ = [
    "BatchInferenceEngine",
    "OptimizationBenchmark",
    "OptimizedRuntime",
    "run_autocast",
]
