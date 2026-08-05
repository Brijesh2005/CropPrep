"""Optimized runtime wrappers around a CropFusionModel.

``OptimizedRuntime`` exposes a uniform ``predict(batch) -> dict`` API across
eager / autocast / compiled / onnx modes. ``run_autocast`` is a context
manager used by both the runtime and the benchmark.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Mapping

import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    import onnxruntime  # noqa: F401
except ImportError:  # pragma: no cover
    onnxruntime = None

MODE_NAMES = ("eager", "autocast", "compiled", "onnx")

_OUTPUT_KEYS = ("crop_logits", "yield_pred", "shared_representation")


def run_autocast(device: torch.device, *, dtype: torch.dtype | None = None) -> Any:
    """Return an autocast context manager suitable for ``device``.

    Falls back to a no-op context on CPU or when autocast is unavailable so
    mixed-precision code runs anywhere without special-casing.
    """
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype or torch.float16)
    if device.type == "cpu" and hasattr(torch, "autocast") and torch.__version__ >= "2.1":
        return torch.autocast("cpu", dtype=dtype or torch.bfloat16)
    return contextlib.nullcontext()


class OptimizedRuntime:
    """Run a model through any supported optimisation backend.

    Args:
        model: A :class:`CropFusionModel` (or any ``nn.Module`` whose
            ``forward(batch_dict)`` returns an object with ``crop_logits`` /
            ``yield_pred`` attributes).
        device: Target device.
        mode: One of ``eager``, ``autocast``, ``compiled``, ``onnx``.
        autocast_dtype: Dtype used by mixed precision.
        compile_options: kwargs forwarded to ``torch.compile``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | None = None,
        mode: str = "eager",
        autocast_dtype: torch.dtype | None = None,
        compile_options: Mapping[str, Any] | None = None,
        onnx_path: str | None = None,
        sample_batch: Mapping[str, Any] | None = None,
    ) -> None:
        if mode not in MODE_NAMES:
            raise ValueError(f"mode must be one of {MODE_NAMES}, got {mode!r}")
        self.device = device or (next(model.parameters()).device)
        self.mode = mode
        self.autocast_dtype = autocast_dtype
        self.model = model.eval()
        self._eager_model = model.eval()

        if mode == "compiled":
            self.model = self._compile(compile_options or {})
        elif mode == "onnx":
            from .onnx_runtime import OnnxRuntimeEngine

            self._onnx = OnnxRuntimeEngine(
                model, onnx_path=onnx_path, sample_batch=sample_batch
            )

    def _compile(self, options: Mapping[str, Any]) -> nn.Module:
        try:
            return torch.compile(self.model, **options)
        except Exception:  # pragma: no cover - compiler availability varies
            return self.model

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Run inference on a model batch and return numpy outputs."""
        if self.mode == "onnx":
            return self._onnx.predict(batch)
        prepared = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        if self.mode == "autocast":
            with run_autocast(self.device, dtype=self.autocast_dtype):
                out = self.model(prepared)
        else:
            out = self._forward(prepared)
        return _outputs_to_dict(out)

    def _forward(self, prepared: Mapping[str, torch.Tensor]):
        """Run the model, falling back to eager when ``torch.compile`` fails lazily."""
        try:
            return self.model(prepared)
        except Exception:
            if self.mode != "compiled" or self.model is self._eager_model:
                raise
            self.model = self._eager_model
            return self.model(prepared)

    @torch.no_grad()
    def predict_proba(self, batch: Mapping[str, Any]) -> Any:
        """Softmax crop probabilities (``[B, K]`` numpy)."""
        outputs = self.predict(batch)
        logits = outputs.get("crop_logits")
        if logits is None:
            raise ValueError("model has no crop head")
        import numpy as np

        import torch as _torch

        probs = _torch.softmax(_torch.from_numpy(logits).float(), dim=-1)
        return probs.numpy()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def device_name(self) -> str:
        return str(self.device)


def _outputs_to_dict(out: Any) -> dict[str, Any]:
    """Normalise a ``CropFusionOutput`` / dict into numpy keyed by output name."""
    import numpy as np

    if isinstance(out, dict):
        raw = out
    elif hasattr(out, "as_dict"):
        raw = out.as_dict()
    else:
        raw = vars(out)
    result: dict[str, Any] = {}
    for key in _OUTPUT_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        value = value.detach().cpu()
        if value.dtype in (torch.bfloat16, torch.float16):
            value = value.float()
        result[key] = np.asarray(value)
    return result
