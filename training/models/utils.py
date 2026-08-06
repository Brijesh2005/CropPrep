"""Shared helpers for the AI model package.

Includes parameter / memory accounting, model summaries, activation and
positional-encoding building blocks, and tensor mask helpers. Pure building
blocks — no model logic lives here.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from .exceptions import ModelConfigurationError, ShapeMismatchError

__all__ = [
    "count_parameters",
    "parameter_summary",
    "layer_summary",
    "architecture_report",
    "estimate_parameter_memory",
    "estimate_activation_memory",
    "model_summary",
    "get_activation",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "build_positional_encoding",
    "build_key_padding_mask",
    "masked_mean",
    "resolve_backbone_name",
    "is_power_of_two",
]


# --------------------------------------------------------------------------- #
# Parameter / memory accounting
# --------------------------------------------------------------------------- #


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count module parameters (optionally only trainable ones)."""
    return sum(
        p.numel() for p in module.parameters() if not trainable_only or p.requires_grad
    )


def parameter_summary(module: nn.Module) -> dict[str, int]:
    """Return ``{total, trainable, frozen}`` parameter counts."""
    total = count_parameters(module)
    trainable = count_parameters(module, trainable_only=True)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def layer_summary(module: nn.Module) -> list[dict[str, Any]]:
    """Recursive per-module layer summary.

    Returns one row per leaf ``nn.Module`` in the tree with its class name,
    parameter count, trainable count and output shape placeholder.
    """
    rows: list[dict[str, Any]] = []

    def _walk(prefix: str, mod: nn.Module) -> None:
        for name, child in mod.named_children():
            path = f"{prefix}.{name}" if prefix else name
            child_params = count_parameters(child)
            child_trainable = count_parameters(child, trainable_only=True)
            rows.append(
                {
                    "name": path,
                    "type": type(child).__name__,
                    "params": child_params,
                    "trainable": child_trainable,
                }
            )
            _walk(path, child)

    _walk("", module)
    return rows


def _shape_of(value: Any) -> Any:
    """Collapse a forward-pass value into shape / type descriptors.

    Handles tensors, nested tuples / lists / dicts and dataclasses (e.g.
    :class:`~ai.models.fusion_engine.FusionOutput`) so per-module hooks can
    describe any intermediate without materialising it.
    """
    if isinstance(value, torch.Tensor):
        return tuple(int(size) for size in value.shape)
    if isinstance(value, (tuple, list)):
        return [_shape_of(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _shape_of(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: _shape_of(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    return type(value).__name__


def architecture_report(
    module: nn.Module, forward_fn: Any | None = None
) -> list[dict[str, Any]]:
    """Trace real input / output shapes through every named submodule.

    Runs one forward pass with per-module hooks, then joins the captured
    shapes with parameter counts. Handles modules whose ``forward`` takes
    non-tensor arguments via the zero-arg ``forward_fn`` (e.g. a
    :class:`CropFusionModel` forward that takes a batch dict)::

        rows = architecture_report(model, forward_fn=lambda: model(batch))

    Args:
        module: The module to inspect.
        forward_fn: Zero-arg callable running one forward pass (required for
            dict / masked forwards). When ``None`` the module is called with no
            arguments.

    Returns:
        One row per named submodule: ``name``, ``type``, ``params``,
        ``trainable``, ``input_shapes`` and ``output_shapes``.
    """
    rows: list[dict[str, Any]] = []

    def _make_hook(name: str, mod: nn.Module) -> Any:
        def _hook(_mod: nn.Module, args: Any, output: Any) -> None:
            rows.append(
                {
                    "name": name,
                    "type": type(mod).__name__,
                    "input_shapes": [_shape_of(arg) for arg in (args or ())],
                    "output_shapes": _shape_of(output),
                }
            )

        return _hook

    hooks: list[Any] = []
    for name, mod in module.named_modules():
        if name:
            hooks.append(mod.register_forward_hook(_make_hook(name, mod)))
    try:
        with torch.no_grad():
            if forward_fn is not None:
                forward_fn()
            else:
                module()
    finally:
        for hook in hooks:
            hook.remove()

    counts = {
        name: {
            "params": count_parameters(mod),
            "trainable": count_parameters(mod, trainable_only=True),
        }
        for name, mod in module.named_modules()
        if name
    }
    for row in rows:
        meta = counts.get(row["name"], {})
        row["params"] = meta.get("params", 0)
        row["trainable"] = meta.get("trainable", 0)
    return rows


def estimate_parameter_memory(module: nn.Module, *, bytes_per_element: int = 4) -> int:
    """Estimated memory (bytes) for storing all parameters as float32."""
    return count_parameters(module) * bytes_per_element


def estimate_activation_memory(
    module: nn.Module,
    sample_input: Sequence[torch.Tensor] | torch.Tensor | None = None,
    *,
    forward_fn: Any | None = None,
    bytes_per_element: int = 4,
) -> int:
    """Estimated activation memory (bytes) via a single forward pass.

    Registers forward hooks that sum the byte-size of every output tensor;
    the estimate is upper-bounded by the largest observed value to account
    for peaks during backpropagation (activations are held for the backward
    pass of the whole graph).

    Args:
        module: The module to measure.
        sample_input: Positional input tensors (used when ``forward_fn`` is
            ``None`` and the module's ``forward`` takes tensors).
        forward_fn: Optional zero-arg callable that runs one forward pass —
            required when the module's ``forward`` takes non-tensor arguments
            (e.g. :class:`~ai.models.cropfusion.CropFusionModel` takes a
            batch dict).
    """
    peak = 0
    count = 0
    was_training = module.training
    module.eval()

    def _hook(_mod: nn.Module, _args: Any, output: Any) -> None:
        nonlocal peak, count
        tensors = output if isinstance(output, (tuple, list)) else [output]
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                peak = max(peak, tensor.numel() * bytes_per_element)
                count += 1

    hooks = [m.register_forward_hook(_hook) for m in module.modules()]
    try:
        with torch.no_grad():
            if forward_fn is not None:
                forward_fn()
            else:
                if not isinstance(sample_input, (tuple, list)):
                    sample_input = (sample_input,)
                module(*sample_input)
    finally:
        for hook in hooks:
            hook.remove()
        module.train(mode=was_training)
    if count == 0:
        return 0
    # Backward keeps the largest layer's activations; a single-batch pass sees
    # roughly one layer's worth at a time, so the peak is a fair estimate.
    return peak


def model_summary(
    module: nn.Module,
    sample_input: Sequence[torch.Tensor] | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Assemble a compact model summary dict (params + layers + memory)."""
    params = parameter_summary(module)
    memory: dict[str, Any] = {
        "parameters_bytes": estimate_parameter_memory(module),
        "parameters_mb": round(estimate_parameter_memory(module) / (1024**2), 4),
    }
    if sample_input is not None:
        memory["activation_bytes"] = estimate_activation_memory(module, sample_input)
        memory["activation_mb"] = round(memory["activation_bytes"] / (1024**2), 4)
    return {
        "parameter_summary": params,
        "parameter_count": params["total"],
        "layer_summary": layer_summary(module),
        "memory_estimate": memory,
    }


# --------------------------------------------------------------------------- #
# Activation helpers
# --------------------------------------------------------------------------- #


def get_activation(name: str) -> nn.Module:
    """Resolve an activation name to a module instance.

    Supported: ``relu``, ``gelu``, ``silu`` (swish), ``tanh``, ``leaky_relu``.
    """
    key = (name or "relu").strip().lower().replace("-", "_")
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }
    if key not in activations:
        raise ModelConfigurationError(
            f"Unsupported activation {name!r}; choose from {sorted(activations)}",
            detail=key,
        )
    return activations[key]()


# --------------------------------------------------------------------------- #
# Positional encodings
# --------------------------------------------------------------------------- #


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, base: float = 10000.0) -> None:
        super().__init__()
        if d_model <= 0 or d_model % 2 != 0:
            raise ModelConfigurationError(
                "Sinusoidal positional encoding requires a positive even d_model",
                detail=d_model,
            )
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(base) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``[B, T, D]`` (truncates to T)."""
        if x.size(1) > self.max_len:
            raise ShapeMismatchError(
                f"Sequence length {x.size(1)} exceeds positional encoding max_len "
                f"{self.max_len}"
            )
        return x + self.pe[:, : x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """Learnable positional encoding over a fixed maximum length."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        if d_model <= 0:
            raise ModelConfigurationError(
                "Learned positional encoding requires a positive d_model", detail=d_model
            )
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, d_model)
        #: Pre-computed position indices so tracing / ONNX export stays static.
        self.register_buffer("position_ids", torch.arange(max_len, dtype=torch.long))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``[B, T, D]`` (truncates to T)."""
        if x.size(1) > self.max_len:
            raise ShapeMismatchError(
                f"Sequence length {x.size(1)} exceeds positional encoding max_len "
                f"{self.max_len}"
            )
        positions = self.position_ids[: x.size(1)].unsqueeze(0)  # [1, T]
        return x + self.embedding(positions)


def build_positional_encoding(
    name: str, d_model: int, max_len: int
) -> nn.Module | None:
    """Build a positional encoding module from ``none|sinusoidal|learned``."""
    key = (name or "none").strip().lower()
    if key in ("", "none"):
        return None
    if key == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len=max_len)
    if key == "learned":
        return LearnedPositionalEncoding(d_model, max_len=max_len)
    raise ModelConfigurationError(
        f"Unsupported positional encoding {name!r}; choose from "
        "none|sinusoidal|learned",
        detail=key,
    )


# --------------------------------------------------------------------------- #
# Mask helpers
# --------------------------------------------------------------------------- #


def build_key_padding_mask(
    mask: torch.Tensor | None, *, has_cls: bool
) -> torch.Tensor | None:
    """Convert a float validity mask ``[B, T]`` (1=real, 0=padding) into an
    ``nn.Transformer``-style key-padding mask ``[B, T']`` (True=ignored).

    When ``has_cls`` the CLS position (column 0) is always valid, so a
    ``False`` column is prepended.
    """
    if mask is None:
        return None
    padding = mask < 0.5  # [B, T] bool — True where padding
    if has_cls:
        padding = torch.cat(
            [torch.zeros_like(padding[:, :1]), padding], dim=1
        )
    return padding


def masked_mean(
    x: torch.Tensor, mask: torch.Tensor | None, dim: int = 1
) -> torch.Tensor:
    """Mean over ``dim`` restricted to real (``mask == 1``) positions."""
    if mask is None:
        return x.mean(dim=dim)
    expanded = mask.unsqueeze(-1).expand_as(x)
    denom = mask.sum(dim=dim, keepdim=True).clamp(min=1.0)
    return (x * expanded.to(x.dtype)).sum(dim=dim) / denom


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #


def resolve_backbone_name(base: str | None, override: str | None) -> str | None:
    """Resolve a per-modality backbone override, falling back to the base."""
    return override or base


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def freeze_matching(module: nn.Module, patterns: Sequence[str]) -> list[str]:
    """Freeze every parameter whose name matches any regex pattern.

    Returns the list of frozen parameter names.
    """
    compiled = [re.compile(p) for p in patterns]
    frozen: list[str] = []
    for name, param in module.named_parameters():
        if any(c.search(name) for c in compiled):
            param.requires_grad_(False)
            frozen.append(name)
    return frozen


def filter_state_dict(
    state: OrderedDict[str, torch.Tensor],
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Filter a state dict by regex include / exclude patterns."""
    include_pat = [re.compile(p) for p in (include or [])]
    exclude_pat = [re.compile(p) for p in (exclude or [])]
    kept = OrderedDict()
    for key, value in state.items():
        if include_pat and not any(p.search(key) for p in include_pat):
            continue
        if any(p.search(key) for p in exclude_pat):
            continue
        kept[key] = value
    return kept


def iterable_len(values: Iterable[Any]) -> int:
    """Return the length of a sized iterable without importing numpy."""
    return len(values)
