"""ONNX Runtime inference engine for CropFusion models.

Exports a model to ONNX via :class:`ai.models.exporter.ModelExporter`, builds
an :class:`onnxruntime.InferenceSession`, and runs batch dicts through it.
Includes a parity check against the eager PyTorch forward so ONNX regression
is caught immediately.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from training.models.exporter import ModelExporter

_OUTPUT_KEYS = ("crop_logits", "yield_pred", "shared_representation")


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout/stderr so torch.onnx's emoji logging survives cp1252."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if stream is None or not hasattr(stream, "buffer"):
            continue
        try:
            if stream.encoding.lower() in ("cp1252", "latin-1", "windows-1252"):
                wrapped = io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace")
                setattr(sys, stream_name, wrapped)
        except (AttributeError, ValueError):  # pragma: no cover
            pass


class OnnxRuntimeEngine:
    """Run a CropFusionModel through ONNX Runtime."""

    def __init__(
        self,
        model: nn.Module,
        *,
        onnx_path: str | Path | None = None,
        opset: int | None = None,
        providers: list[str] | None = None,
        sample_batch: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("onnxruntime is required for ONNX inference") from exc

        self.model = model
        self.sample = (
            dict(sample_batch)
            if sample_batch is not None
            else model.sample_batch(batch_size=1) if hasattr(model, "sample_batch") else None
        )
        self._input_names = _enabled_input_names(model)

        if onnx_path is None:
            onnx_path = _default_export_path(model, opset=opset)
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            _ensure_utf8_stdout()
            ModelExporter(model, sample_batch=self.sample).export_onnx(
                self.onnx_path, opset=opset, dynamic_batch=False
            )

        available = onnxruntime.get_available_providers()
        chosen = [p for p in (providers or ["CPUExecutionProvider"]) if p in available]
        self.session = onnxruntime.InferenceSession(
            str(self.onnx_path), providers=chosen or None
        )
        self._output_names = [out.name for out in self.session.get_outputs()]

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Run a batch through ONNX and return numpy outputs keyed by name."""
        feeds = {}
        for name in self._input_names:
            value = batch.get(name)
            if value is None:
                continue
            feeds[name] = (
                value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            )
        results = self.session.run(self._output_names, feeds)
        mapping = dict(zip(self._output_names, results))
        return {key: mapping[key] for key in _OUTPUT_KEYS if key in mapping}

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def assert_parity(
        self,
        batch: Mapping[str, Any],
        *,
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ) -> dict[str, float]:
        """Compare ONNX outputs to the eager torch forward; returns max errors."""
        eager = _eager_outputs(self.model, batch)
        onnx = self.predict(batch)
        errors: dict[str, float] = {}
        for key, expected in eager.items():
            actual = onnx.get(key)
            if actual is None:
                continue
            expected = np.asarray(expected)
            actual = np.asarray(actual)
            if expected.shape != actual.shape:
                raise AssertionError(f"{key}: shape {expected.shape} != {actual.shape}")
            abs_err = float(np.max(np.abs(expected - actual)))
            denom = float(np.max(np.abs(expected))) or 1.0
            rel_err = abs_err / denom
            if not (abs_err <= atol + rtol * denom):
                raise AssertionError(
                    f"{key}: ONNX vs torch mismatch abs={abs_err:.6f} rel={rel_err:.6f}"
                )
            errors[key] = abs_err
        return errors

    def warmup(self, iterations: int = 3) -> None:
        if self.sample is not None:
            for _ in range(iterations):
                self.predict(self.sample)


def _enabled_input_names(model: nn.Module) -> list[str]:
    names: list[str] = []
    if getattr(model, "use_tabular", False):
        names.append("tabular")
    if getattr(model, "use_image", False):
        if getattr(model, "ndvi_encoder", None) is not None:
            names.append("ndvi")
        if getattr(model, "evi_encoder", None) is not None:
            names.append("evi")
        names.append("temporal_mask")
    return names


def _default_export_path(model: nn.Module, *, opset: int | None) -> Path:
    from tempfile import mkdtemp

    version = getattr(model.config, "name", "model") if hasattr(model, "config") else "model"
    return Path(mkdtemp(prefix="cropfusion_onnx_")) / f"{version}_op{opset or 17}.onnx"


def _eager_outputs(model: nn.Module, batch: Mapping[str, Any]) -> dict[str, Any]:
    prepared = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
    out = model(prepared)
    raw = out.as_dict() if hasattr(out, "as_dict") else vars(out)
    return {
        key: np.asarray(raw[key].detach().cpu()) for key in _OUTPUT_KEYS if raw.get(key) is not None
    }
