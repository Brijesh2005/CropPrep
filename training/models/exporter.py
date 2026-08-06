"""Model export — TorchScript, ONNX and (future) TensorRT.

``ModelExporter`` wraps a :class:`~ai.models.cropfusion.CropFusionModel` so
its dict-based forward becomes a plain tensor-in/tensor-out graph, then
exports it to TorchScript (traced) or ONNX. TensorRT is a documented future
target: the entry point validates the environment and raises a clear
:class:`~ai.models.exceptions.MissingDependencyError` until the dependency is
available — no silent no-op.
"""

from __future__ import annotations

import keyword
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import nn

from .exceptions import ExportError, MissingDependencyError

#: Export inputs, in forward order (subset used per modality).
_INPUT_NAMES = ("tabular", "ndvi", "evi", "temporal_mask")


def _patch_fx_keyword_codegen() -> None:
    """Let torch.fx quote Python-keyword attribute names (e.g. ``yield``).

    ``torch.fx.graph._format_target`` only treats non-identifier segments as
    needing ``getattr`` quoting, so a submodule registered under a Python
    keyword (CropFusion registers its yield head as ``heads._heads.yield``)
    generates invalid module code during ONNX export. Quoting keywords too is
    purely a superset of the original behaviour.
    """
    try:
        import torch.fx.graph as fx_graph
    except ImportError:  # pragma: no cover - torch without fx
        return
    original = fx_graph._format_target

    def _format_target(base: str, target: str) -> str:
        elements = target.split(".")
        rendered = base
        for element in elements:
            if not element.isidentifier() or keyword.iskeyword(element):
                rendered = f'getattr({rendered}, "{element}")'
            else:
                rendered = f"{rendered}.{element}"
        return rendered

    fx_graph._format_target = _format_target
    _format_target.__name__ = getattr(original, "__name__", "_format_target")


_patch_fx_keyword_codegen()


class _ExportWrapper(nn.Module):
    """Tensor-in / tensor-out adapter around a CropFusionModel."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        tabular: torch.Tensor | None = None,
        ndvi: torch.Tensor | None = None,
        evi: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return self.model.forward_export(
            tabular=tabular, ndvi=ndvi, evi=evi, temporal_mask=temporal_mask
        )


class ModelExporter:
    """Export helper for :class:`~ai.models.cropfusion.CropFusionModel`.

    Args:
        model: A :class:`CropFusionModel`.
        sample_batch: Batch dict used to derive example inputs
            (defaults to :meth:`CropFusionModel.sample_batch`).
    """

    def __init__(self, model: nn.Module, sample_batch: Mapping[str, Any] | None = None) -> None:
        self.model = model
        self.sample = (
            dict(sample_batch)
            if sample_batch is not None
            else model.sample_batch()
        )
        self.wrapper = _ExportWrapper(model)

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    @contextmanager
    def _eval_guard(self) -> Iterator[None]:
        """Run export in eval mode with input validation disabled.

        Validation is a Python-time pre-check, not part of the compute graph;
        leaving it on during tracing turns shape checks into trace-time
        constants and spams tracer warnings.
        """
        was_training = self.model.training
        was_validate = bool(getattr(self.model.config, "validate_inputs", False))
        self.model.eval()
        if was_validate:
            self.model.config.validate_inputs = False
        try:
            yield
        finally:
            self.model.config.validate_inputs = was_validate
            self.model.train(mode=was_training)

    def _export_args(self) -> tuple[torch.Tensor, ...]:
        """Example inputs as a positional tuple (ordered per enabled modality).

        The wrapper's ``forward`` accepts any prefix of
        ``(tabular, ndvi, evi, temporal_mask)``, so only the inputs the model
        actually consumes are traced — this keeps TorchScript / ONNX graphs
        free of unused inputs.
        """
        uses_tabular = bool(getattr(self.model, "use_tabular", False))
        uses_image = bool(getattr(self.model, "use_image", False))
        args: list[torch.Tensor] = []
        if uses_tabular:
            args.append(self.sample["tabular"])
        if uses_image:
            args.append(self.sample["ndvi"])
            args.append(self.sample["evi"])
            args.append(self.sample["temporal_mask"])
        return tuple(args)

    def _input_names(self) -> list[str]:
        uses_tabular = bool(getattr(self.model, "use_tabular", False))
        uses_image = bool(getattr(self.model, "use_image", False))
        names: list[str] = []
        if uses_tabular:
            names.append("tabular")
        if uses_image:
            names.extend(("ndvi", "evi", "temporal_mask"))
        return names

    def _output_names(self) -> list[str]:
        """Output names matching the order of :meth:`CropFusionModel.forward_export`."""
        head_names = list(self.model.heads.names) if hasattr(self.model, "heads") else []
        names: list[str] = []
        if "crop" in head_names:
            names.append("crop_logits")
        if "yield" in head_names:
            names.append("yield_pred")
        names.append("shared_representation")
        return names

    # ------------------------------------------------------------------ #
    # TorchScript
    # ------------------------------------------------------------------ #

    def export_torchscript(
        self, path: str | Path, *, mode: str | None = None
    ) -> Path:
        """Export the model as a traced TorchScript module.

        Args:
            path: Destination ``.pt`` file.
            mode: ``trace`` (default) — scripting is reserved for models whose
                forward is fully scriptable.

        Raises:
            ExportError: If tracing or saving fails.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        export_mode = mode or "trace"
        if export_mode != "trace":
            raise ExportError(
                "only 'trace' TorchScript export is supported for the "
                "dict-based forward; scripting requires a tensor-only forward",
                detail=export_mode,
            )

        args = self._export_args()
        with self._eval_guard():
            try:
                with torch.no_grad():
                    traced = torch.jit.trace(
                        self.wrapper, args, check_trace=False
                    )
                torch.jit.save(traced, out)
            except (RuntimeError, TypeError) as exc:
                raise ExportError(
                    f"TorchScript export failed: {exc}", detail=str(out)
                ) from exc
        return out

    # ------------------------------------------------------------------ #
    # ONNX
    # ------------------------------------------------------------------ #

    def export_onnx(
        self,
        path: str | Path,
        *,
        opset: int | None = None,
        dynamic_batch: bool = True,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
    ) -> Path:
        """Export the model to ONNX.

        Args:
            path: Destination ``.onnx`` file.
            opset: ONNX opset version (defaults to the model config).
            dynamic_batch: Mark the batch axis dynamic.
            input_names / output_names / dynamic_axes: Overrides for the
                ONNX graph metadata.

        Raises:
            MissingDependencyError: When the ``onnx`` package is unavailable.
            ExportError: When export fails.
        """
        try:
            import onnx  # noqa: F401  (validates availability)
        except ImportError as exc:  # pragma: no cover - env dependent
            raise MissingDependencyError(
                "ONNX export requires the 'onnx' package "
                "(pip install onnx); model export skipped"
            ) from exc

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        config = getattr(self.model, "config", None)
        opset_version = opset or (config.export.onnx_opset if config else 17)

        args = self._export_args()
        names_in = input_names or self._input_names()
        names_out = output_names or self._output_names()

        axes: dict[str, dict[int, str]] = {}
        if dynamic_batch:
            for name in names_in:
                axes[name] = {0: "batch"}
            for name in names_out:
                axes[name] = {0: "batch"}
            for name in ("ndvi", "evi", "temporal_mask"):
                if name in names_in:
                    axes[name][1] = "time"
        if dynamic_axes is not None:
            axes = dynamic_axes

        with self._eval_guard():
            fast_path_was = torch.backends.mha.get_fastpath_enabled()
            if fast_path_was:
                torch.backends.mha.set_fastpath_enabled(False)
            try:
                with torch.no_grad():
                    torch.onnx.export(
                        self.wrapper,
                        args,
                        str(out),
                        opset_version=opset_version,
                        input_names=names_in,
                        output_names=names_out,
                        dynamic_axes=axes if axes else None,
                        dynamo=False,
                    )
            except Exception as exc:
                raise ExportError(
                    f"ONNX export failed: {exc}", detail=str(out)
                ) from exc
            finally:
                if fast_path_was:
                    torch.backends.mha.set_fastpath_enabled(True)
        return out

    # ------------------------------------------------------------------ #
    # TensorRT (future)
    # ------------------------------------------------------------------ #

    def export_tensorrt(
        self, path: str | Path, *, onnx_path: str | Path | None = None, **_: Any
    ) -> Path:
        """Convert an ONNX graph to a TensorRT engine (future target).

        TensorRT deployment is planned for the serving phase. Until the
        ``tensorrt`` package and a GPU runtime are available this entry point
        validates the environment and reports exactly what is missing — it
        never silently produces nothing.

        Raises:
            MissingDependencyError: Always, until ``tensorrt`` is available.
        """
        try:
            import tensorrt  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependent
            raise MissingDependencyError(
                "TensorRT export is a planned Phase 6+ deployment feature and "
                "requires the 'tensorrt' package + a CUDA runtime; export the "
                "model to ONNX first (ModelExporter.export_onnx)"
            ) from exc
        raise MissingDependencyError(
            "TensorRT runtime present but engine build is not implemented in "
            "this phase; use the ONNX export and trtexec for now"
        )
