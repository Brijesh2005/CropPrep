"""Shared helpers for the preprocessing pipeline (tensor conversion, padding)."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

from .exceptions import PreprocessingError


def to_float_tensor(values: Any, *, device: str | None = None) -> Any:
    """Convert an array-like into a float32 tensor (lazy torch import)."""
    torch = _torch()
    array = np.asarray(values, dtype="float32")
    tensor = torch.from_numpy(array)
    return tensor.to(device) if device else tensor


def to_long_tensor(values: Any, *, device: str | None = None) -> Any:
    """Convert an array-like into an int64 tensor."""
    torch = _torch()
    array = np.asarray(values, dtype="int64")
    tensor = torch.from_numpy(array)
    return tensor.to(device) if device else tensor


def pad_sequence_tensors(
    tensors: Sequence[Any],
    max_length: int,
    *,
    pad_value: float = 0.0,
    pad_side: str = "right",
    truncation: str = "tail",
) -> tuple[Any, Any]:
    """Pad/truncate a sequence of tensors to ``max_length``.

    Returns ``(stacked_tensor, mask_tensor)`` where mask is 1 for real
    positions and 0 for padding. All tensors must share the same shape.
    """
    torch = _torch()
    if not tensors:
        return torch.zeros(0), torch.zeros(0)

    shape = tuple(tensors[0].shape)
    items = list(tensors)

    # Truncate.
    if len(items) > max_length:
        if truncation == "head":
            items = items[len(items) - max_length:]
        else:  # tail
            items = items[:max_length]

    count = len(items)
    mask = np.ones(max_length, dtype="float32")
    mask[count:] = 0.0

    # Pad.
    if count < max_length:
        filler = torch.full(shape, pad_value, dtype=items[0].dtype)
        padding = [filler] * (max_length - count)
        if pad_side == "left":
            items = padding + items
        else:  # right
            items = items + padding

    stacked = torch.stack(items, dim=0) if items else torch.zeros(0)
    return stacked, to_float_tensor(mask)


def safe_mean(values: Sequence[float]) -> float | None:
    cleaned = [v for v in values if v is not None and not _is_nan(v)]
    return float(np.mean(cleaned)) if cleaned else None


def safe_std(values: Sequence[float]) -> float | None:
    cleaned = [v for v in values if v is not None and not _is_nan(v)]
    return float(np.std(cleaned)) if len(cleaned) > 1 else 0.0


def is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def unique_counts(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def stable_hash(*parts: Any) -> str:
    import hashlib

    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()[:16]


def require_fitted(pipeline: Any, name: str) -> None:
    from .exceptions import FitError

    if not getattr(pipeline, "fitted", False):
        raise FitError(f"{name} has not been fitted; call fit() first")


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise PreprocessingError(
            "PyTorch is required by the preprocessing pipeline; install with "
            "`pip install torch`"
        ) from exc
