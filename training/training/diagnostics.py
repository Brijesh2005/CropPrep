"""Training-time diagnostics for the CropFusion pipeline.

R5.2-mandated observability that runs inside the training loop itself:

* :func:`profile_batch` — a one-time multimodal batch fingerprint
  (``B / T / C / H / W``, real-vs-zero-filled frame counts, finiteness) so a
  run can prove the exact tensor contract that reached the model.
* :func:`assert_image_batch_shape` — hard assertion that every image tensor is
  ``[B, T, 1, H, W]`` with ``H == W == expected_hw`` and at least one real
  timestep (fail-loudly, never suppressed).
* :func:`nan_source_hooks` — attach forward hooks for one pass to attribute a
  NaN / Inf to the first module output that produced it (mirrors the
  fail-loudly NaN policy: it *describes* the source, it does not swallow it).

All helpers are framework-agnostic (``torch`` only) and safe under ``no_grad``
and autocast.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Mapping

import torch
from torch import nn

__all__ = [
    "profile_batch",
    "assert_image_batch_shape",
    "nan_source_hooks",
    "tensor_stats",
]


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    """Per-tensor finiteness summary: ``{nan, inf, min, max, finite}``."""
    value = tensor.detach().float()
    finite = bool(torch.isfinite(value).all().item())
    if value.numel() == 0:
        return {"nan": 0, "inf": 0, "min": None, "max": None, "finite": finite}
    return {
        "nan": int(torch.isnan(value).sum().item()),
        "inf": int(torch.isinf(value).sum().item()),
        "min": float(value.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).min().item()),
        "max": float(value.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).max().item()),
        "finite": finite,
    }


def _image_frames(batch: Mapping[str, Any], key: str) -> tuple[int, int, dict[str, Any]]:
    """Count real vs zero-filled frames for one image stream.

    Returns ``(real_frames, zero_filled_frames, stats_and_shape)``.
    """
    tensor = batch.get(key)
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return 0, 0, {"present": False, **tensor_stats(torch.zeros(1))}
    shape = [int(size) for size in tensor.shape]
    stats = {"present": True, "shape": shape}
    real = int((tensor.abs().sum(dim=(2, 3, 4)) > 0).count_nonzero().item())
    total = int(tensor.size(0)) * int(tensor.size(1))
    return real, max(0, total - real), stats


def profile_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Fingerprint a Phase 4 batch dict.

    Captures the tensor contract that actually reached the model:

    * per-image-stream shape and real (nonzero) vs zero-filled frame counts,
    * the temporal mask coverage,
    * finiteness of every input tensor.

    This is how a run proves imagery really flowed (vs the zero-fill fallback)
    and that the inputs were finite before any model arithmetic.
    """
    profile: dict[str, Any] = {}

    for key in ("tabular", "ndvi", "evi", "temporal_mask", "crop_label", "yield_label"):
        tensor = batch.get(key)
        if not isinstance(tensor, torch.Tensor):
            continue
        shape = [int(size) for size in tensor.shape]
        finite = bool(torch.isfinite(tensor).all().item())
        entry: dict[str, Any] = {"shape": shape, "finite": finite}
        if tensor.dtype.is_floating_point:
            stats = tensor_stats(tensor)
            entry.update({"nan": stats["nan"], "inf": stats["inf"],
                          "min": stats["min"], "max": stats["max"]})
        profile[key] = entry

    ndvi_real, ndvi_zero, ndvi_stat = _image_frames(batch, "ndvi")
    evi_real, evi_zero, evi_stat = _image_frames(batch, "evi")
    profile["ndvi_frames"] = {"real": ndvi_real, "zero_filled": ndvi_zero, **ndvi_stat}
    profile["evi_frames"] = {"real": evi_real, "zero_filled": evi_zero, **evi_stat}

    mask = batch.get("temporal_mask")
    if isinstance(mask, torch.Tensor):
        mask_float = mask.float()
        profile["mask"] = {
            "mean": float(mask_float.mean().item()) if mask_float.numel() else None,
            "min": float(mask_float.min().item()) if mask_float.numel() else None,
            "ones": int((mask > 0.5).count_nonzero().item()),
            "shape": [int(size) for size in mask.shape],
        }

    profile["batch_size"] = None
    for key in ("tabular", "ndvi", "evi", "crop_label"):
        tensor = batch.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
            profile["batch_size"] = int(tensor.size(0))
            break
    return profile


def assert_image_batch_shape(
    batch: Mapping[str, Any],
    expected_hw: int | None,
    *,
    error_type: type[Exception] = AssertionError,
    detail: str = "",
) -> dict[str, Any]:
    """Assert every image stream is ``[B, T, 1, H, W]`` with H == W == hw.

    Raises ``error_type`` (default :class:`AssertionError`) when:

    * an expected image stream is missing or non-finite,
    * the channel dim is not 1,
    * ``expected_hw`` is provided and the frame isn't centred on it,
    * the spatial dims are inconsistent between NDVI and EVI,
    * the sequence is empty (``T == 0``).

    Returns the :func:`profile_batch` fingerprint on success.
    """
    profile = profile_batch(batch)
    present: list[torch.Tensor] = []

    def _fail(message: str) -> None:
        raise error_type(message + (f" ({detail})" if detail else ""))

    for key in ("ndvi", "evi"):
        tensor = batch.get(key)
        if tensor is None or not isinstance(tensor, torch.Tensor):
            if key == "ndvi" and batch.get("ndvi") is None and batch.get("evi") is None:
                continue
            if tensor is None:
                continue
            _fail(f"{key} is not a tensor")
        if tensor.dim() != 5:
            _fail(f"{key} must be [B, T, 1, H, W], got {tuple(tensor.shape)}")
        if int(tensor.size(2)) != 1:
            _fail(f"{key} must be single-channel (C=1), got C={tensor.size(2)}")
        if int(tensor.size(1)) == 0:
            _fail(f"{key} has an empty sequence (T=0)")
        if not bool(torch.isfinite(tensor).all().item()):
            _fail(f"{key} contains non-finite values (NaN/Inf in input)")
        present.append(tensor)

    if len(present) >= 2:
        if present[0].size(1) != present[1].size(1):
            _fail(
                f"ndvi/evi sequence lengths differ "
                f"{tuple(present[0].size(1))} vs {tuple(present[1].size(1))}"
            )
        if tuple(present[0].shape[2:]) != tuple(present[1].shape[2:]):
            _fail(
                f"ndvi/evi spatial dims differ {tuple(present[0].shape[2:])} "
                f"vs {tuple(present[1].shape[2:])}"
            )

    mask = batch.get("temporal_mask")
    if present and not isinstance(mask, torch.Tensor):
        _fail("temporal_mask must be present when image streams are enabled")
    if isinstance(mask, torch.Tensor) and present:
        if mask.dim() != 2:
            _fail(f"temporal_mask must be [B, T], got {tuple(mask.shape)}")
        if int(mask.size(0)) != present[0].size(0) or int(mask.size(1)) != present[0].size(1):
            _fail(
                f"temporal_mask {tuple(mask.shape)} does not match image "
                f"sequence {tuple(present[0].shape[:2])}"
            )
        if int((mask > 0.5).sum().item()) == 0:
            _fail("temporal_mask has no real timesteps (all padding)")

    if expected_hw is not None:
        for tensor in present:
            height, width = int(tensor.size(3)), int(tensor.size(4))
            if (height, width) != (expected_hw, expected_hw):
                _fail(
                    f"image stream must be [B, T, C, {expected_hw}, {expected_hw}], "
                    f"got spatial {height}x{width}"
                )

    profile["image_assert_passed"] = True
    if expected_hw is not None:
        profile["expected_hw"] = int(expected_hw)
    return profile


@contextlib.contextmanager
def nan_source_hooks(model: nn.Module) -> Iterator[list[dict[str, Any]]]:
    """Context manager that records the first non-finite output per module.

    Registering hooks does not run a forward; wrap the *real* forward pass
    (e.g. ``with nan_source_hooks(model) as sources: out = model(batch)``) and
    inspect ``sources`` afterwards. Each entry is
    ``{"module", "type", "shape", "nan", "inf", "input_finite", "origin"}`` for
    the first tensor that broke:

    * ``origin == "created"`` — the module received only finite inputs and its
      output went non-finite first (this module is the ROOT CAUSE locus).
    * ``origin == "propagated"`` — a non-finite input flowed straight through.

    The trace is diagnostic-only: it never raises, never mutates the graph and
    is safe under ``no_grad`` / autocast. The finiteness checks themselves run
    under ``no_grad`` so the hooks add no autograd nodes to the traced forward
    (essential under gradient checkpointing, where the operator graph of the
    original forward must match the recomputation).
    """
    results: list[dict[str, Any]] = []
    handles: list[Any] = []
    # Shared state: stop gathering after the first non-finite output so the
    # trace pinpoints the *source* module rather than every downstream
    # propagation of the same NaN.
    state = {"hit": False}

    def _extract(tensors: Any) -> list[torch.Tensor]:
        if isinstance(tensors, torch.Tensor):
            return [tensors]
        if isinstance(tensors, (tuple, list)):
            return [t for t in tensors if isinstance(t, torch.Tensor)]
        if isinstance(tensors, Mapping):
            return [t for t in tensors.values() if isinstance(t, torch.Tensor)]
        if hasattr(tensors, "__dataclass_fields__"):
            return [
                t for t in tensors.__dict__.values() if isinstance(t, torch.Tensor)
            ]
        return []

    def _make_hook(name: str) -> Any:
        def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            if state["hit"]:
                return
            for tensor in _extract(output):
                if tensor is None or not tensor.dtype.is_floating_point:
                    continue
                with torch.no_grad():
                    finite = bool(torch.isfinite(tensor).all().item())
                if not finite:
                    # Was the breakage produced HERE (finite in -> non-finite
                    # out) or received as a non-finite input (propagated)?
                    input_tensors = [
                        t for t in _extract(_inputs)
                        if t is not None and t.dtype.is_floating_point
                    ]
                    with torch.no_grad():
                        input_finite = all(
                            bool(torch.isfinite(t).all().item())
                            for t in input_tensors
                        )
                    with torch.no_grad():
                        nan_count = int(torch.isnan(tensor).count_nonzero().item())
                        inf_count = int(torch.isinf(tensor).count_nonzero().item())
                    results.append(
                        {
                            "module": name,
                            "type": type(_module).__name__,
                            "shape": [int(size) for size in tensor.shape],
                            "nan": nan_count,
                            "inf": inf_count,
                            "input_finite": bool(input_finite),
                            "origin": "created" if input_finite else "propagated",
                        }
                    )
                    state["hit"] = True
                    return

        return _hook

    for name, module in model.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_hook(_make_hook(name)))

    try:
        yield results
    finally:
        for handle in handles:
            handle.remove()