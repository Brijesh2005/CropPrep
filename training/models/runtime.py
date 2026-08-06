"""Model runtime helpers — precision, device, compile, parallelism.

Everything here applies *execution* settings to an already-built
:class:`~ai.models.cropfusion.CropFusionModel`; it never changes the
architecture. It is the deployment counterpart to
:class:`~ai.models.config.RuntimeConfig` and backs the ModelFactory
``apply_runtime`` helper (also reused by the Phase 6 training loop):

* dtype conversion for AMP (``float16`` / ``bfloat16``) with float32
  normalisation layers preserved,
* explicit device placement,
* ``torch.compile``,
* gradient checkpointing,
* single-node (``nn.DataParallel``) and distributed (``DDP``) parallelism.

Every failure raises a :class:`~ai.models.exceptions.ModelError` subclass with
a stable machine-readable code.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from .exceptions import MissingDependencyError, ModelConfigurationError

#: Normalisation layers that stay in float32 under reduced precision (their
#: statistics are computed in fp16/bfloat16-trained models anyway).
_NORM_MODULES = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)

_PRECISION_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_DTYPE_PRECISIONS: dict[torch.dtype, str] = {
    dtype: name for name, dtype in _PRECISION_DTYPES.items()
}


# --------------------------------------------------------------------------- #
# Device / dtype resolution
# --------------------------------------------------------------------------- #


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a device string to ``torch.device`` (``None`` = CUDA or CPU)."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def dtype_from_precision(precision: str | None) -> torch.dtype:
    """Map a precision name (``float32|float16|bfloat16``) to ``torch.dtype``.

    Raises:
        ModelConfigurationError: For any other value.
    """
    key = (precision or "float32").strip().lower()
    if key not in _PRECISION_DTYPES:
        raise ModelConfigurationError(
            f"Unsupported precision {precision!r}; choose from "
            "float32 | float16 | bfloat16",
            detail=key,
        )
    return _PRECISION_DTYPES[key]


def precision_from_dtype(dtype: torch.dtype) -> str:
    """Reverse-map a ``torch.dtype`` to a precision name (default float32)."""
    return _DTYPE_PRECISIONS.get(dtype, "float32")


# --------------------------------------------------------------------------- #
# Automatic mixed precision
# --------------------------------------------------------------------------- #


@contextmanager
def amp_context(
    precision: str | None = None,
    device: str | torch.device | None = None,
) -> Iterator[None]:
    """Run a block under ``torch.autocast`` for the configured precision.

    ``float32`` is a no-op. ``float16`` on CPU is only supported by some
    PyTorch builds; an unsupported combination raises
    :class:`ModelConfigurationError` instead of failing mid-forward.

    Example::

        with amp_context("bfloat16", "cpu"):
            loss = model(batch)
    """
    dtype = dtype_from_precision(precision) if precision else torch.float32
    if dtype == torch.float32:
        yield
        return
    dev = resolve_device(device)
    if dev.type not in ("cpu", "cuda"):
        raise ModelConfigurationError(
            f"autocast is only supported on CPU/CUDA, got {dev.type}",
            detail=str(dev),
        )
    try:
        with torch.autocast(device_type=dev.type, dtype=dtype):
            yield
    except (ValueError, RuntimeError) as exc:
        raise ModelConfigurationError(
            f"autocast failed for precision {precision} on {dev.type}: {exc}",
            detail=precision,
        ) from exc


# --------------------------------------------------------------------------- #
# Precision / device conversion
# --------------------------------------------------------------------------- #


def apply_precision(model: nn.Module, precision: str) -> nn.Module:
    """Convert a model's floating-point parameters and buffers to a dtype.

    Normalisation layers (LayerNorm / BatchNorm) stay in float32 for numeric
    stability. When the model carries a ``config.runtime`` the precision is
    recorded there too.

    Args:
        model: Any ``nn.Module`` (normally a :class:`CropFusionModel`).
        precision: ``float32`` | ``float16`` | ``bfloat16``.

    Returns:
        The same ``model`` (mutated in place).
    """
    dtype = dtype_from_precision(precision)
    if dtype == torch.float32:
        model.float()
    else:
        for module in model.modules():
            if isinstance(module, _NORM_MODULES):
                module.float()
            else:
                module.to(dtype=dtype)
    runtime = getattr(getattr(model, "config", None), "runtime", None)
    if runtime is not None:
        runtime.precision = precision
    return model


def move_to_device(model: nn.Module, device: str | torch.device | None) -> nn.Module:
    """Move a model to a device (``None`` = auto CPU/CUDA)."""
    return model.to(resolve_device(device))


# --------------------------------------------------------------------------- #
# torch.compile
# --------------------------------------------------------------------------- #


def compile_model(
    model: nn.Module,
    mode: str = "default",
    backend: str | None = None,
) -> nn.Module:
    """Compile a model with ``torch.compile``.

    Args:
        model: The module to compile.
        mode: ``default`` | ``reduce-overhead`` | ``max-autotune`` |
            ``max-autotune-no-cudagraphs``.
        backend: Explicit compile backend (``inductor`` default; ``eager`` /
            ``aot_eager`` useful for testing).

    Raises:
        MissingDependencyError: When ``torch.compile`` is unavailable.
        ModelConfigurationError: When compilation itself fails.
    """
    if not hasattr(torch, "compile"):
        raise MissingDependencyError(
            "torch.compile is unavailable in this PyTorch build",
            detail=torch.__version__,
        )
    try:
        return torch.compile(model, mode=mode, backend=backend)
    except Exception as exc:
        raise ModelConfigurationError(
            f"torch.compile failed (mode={mode}, backend={backend}): {exc}",
            detail=mode,
        ) from exc


# --------------------------------------------------------------------------- #
# Gradient checkpointing
# --------------------------------------------------------------------------- #


def enable_gradient_checkpointing(
    model: nn.Module, enabled: bool = True
) -> nn.Module:
    """Enable / disable activation checkpointing on every transformer stack.

    Targets any submodule exposing ``set_gradient_checkpointing`` (the
    TabTransformer, TemporalTransformer and shared encoder). Recompute
    checkpointing only activates in training mode, so eval / export are
    unaffected.

    Returns:
        The same ``model``.
    """
    for module in model.modules():
        setter = getattr(module, "set_gradient_checkpointing", None)
        if callable(setter):
            setter(bool(enabled))
    runtime = getattr(getattr(model, "config", None), "runtime", None)
    if runtime is not None:
        runtime.gradient_checkpointing = bool(enabled)
    return model


# --------------------------------------------------------------------------- #
# Data parallelism
# --------------------------------------------------------------------------- #


def wrap_data_parallel(
    model: nn.Module,
    device_ids: list[int] | tuple[int, ...] | None = None,
    output_device: int | None = None,
) -> nn.DataParallel:
    """Wrap a model in ``torch.nn.DataParallel`` (single node, multi GPU).

    Raises:
        ModelConfigurationError: When no CUDA device is available — the wrapper
            cannot replicate onto GPUs that do not exist, and a silent CPU
            fallback would mask the misconfiguration.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ModelConfigurationError(
            "nn.DataParallel requires at least one CUDA device",
            detail="cuda unavailable",
        )
    device_ids = list(device_ids) if device_ids else list(range(torch.cuda.device_count()))
    return nn.DataParallel(model, device_ids=device_ids, output_device=output_device)


def wrap_distributed(
    model: nn.Module,
    local_rank: int | None = None,
    device: str | torch.device | None = None,
    *,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
) -> "torch.nn.parallel.DistributedDataParallel":
    """Wrap a model in distributed ``DistributedDataParallel``.

    Requires an initialised process group. The model is moved to the local
    rank's device; on CPU-based process groups ``device_ids`` is left ``None``
    (DDP runs entirely on CPU).

    Raises:
        ModelConfigurationError: When the process group is not initialised.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise ModelConfigurationError(
            "torch.distributed is not initialized; call "
            "torch.distributed.init_process_group(...) before wrapping",
            detail="distributed not initialized",
        )
    rank = (
        local_rank
        if local_rank is not None
        else torch.distributed.get_rank()
    )
    dev = resolve_device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", rank % torch.cuda.device_count())
    model = model.to(dev)
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dev.index] if dev.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
    )


# --------------------------------------------------------------------------- #
# RuntimeConfig convenience
# --------------------------------------------------------------------------- #


def apply_runtime(model: nn.Module, runtime: Any | None = None) -> nn.Module:
    """Apply a :class:`RuntimeConfig` to a built model in one call.

    Order: precision → device → gradient checkpointing → ``torch.compile`` →
    data-parallel wrappers. ``runtime`` defaults to ``model.config.runtime``.

    Returns:
        The configured model (possibly a compiled / wrapped version).
    """
    if runtime is None:
        runtime = getattr(getattr(model, "config", None), "runtime", None)
    if runtime is None:
        return model

    if getattr(runtime, "precision", "float32") != "float32":
        apply_precision(model, runtime.precision)
    if getattr(runtime, "device", None):
        move_to_device(model, runtime.device)
    if getattr(runtime, "gradient_checkpointing", False):
        enable_gradient_checkpointing(model, True)
    if getattr(runtime, "compile", False):
        model = compile_model(model, mode=getattr(runtime, "compile_mode", "default"))
    if getattr(runtime, "data_parallel", False):
        model = wrap_data_parallel(model)
    if getattr(runtime, "distributed", False):
        model = wrap_distributed(model, local_rank=getattr(runtime, "local_rank", None))
    return model
