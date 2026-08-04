"""Shared helpers for the training package.

* Seed control + deterministic training.
* Device resolution with a graceful CPU fallback.
* Distributed (DDP) helpers that no-op on single-process runs.
* Git hash / environment introspection for experiment tracking.
* Gradient checkpointing, timing and tensor conversions.

Pure building blocks — no training-engine logic lives here.
"""

from __future__ import annotations

import contextlib
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .exceptions import TrainingRunError

__all__ = [
    "set_seed",
    "configure_determinism",
    "resolve_device",
    "to_device",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "is_primary",
    "setup_distributed",
    "cleanup_distributed",
    "all_gather_tensor",
    "broadcast_dict",
    "get_git_hash",
    "get_git_branch",
    "get_environment_info",
    "Timer",
    "format_duration",
    "MovingAverage",
    "apply_gradient_checkpointing",
    "tensor_to_numpy",
    "named_enabled_parameters",
    "compute_grad_norm",
    "count_parameters",
    "estimate_parameter_memory",
]


# --------------------------------------------------------------------------- #
# Seeds & determinism
# --------------------------------------------------------------------------- #


def set_seed(seed: int) -> None:
    """Seed torch / numpy / python randomness deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool, seed: int) -> None:
    """Apply seed control and (optionally) deterministic algorithms."""
    set_seed(seed)
    if not enabled:
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # some builds reject the switch
        pass


# --------------------------------------------------------------------------- #
# Device resolution
# --------------------------------------------------------------------------- #


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve the training device, falling back to CPU when CUDA is missing.

    ``"auto"`` picks CUDA when available, else CPU. Explicit ``"cuda"`` on a
    machine without CUDA downgrades to CPU with a warning (graceful fallback).
    """
    if isinstance(device, torch.device):
        return device
    requested = (device or "auto").strip().lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested in {"cpu", "cpu:0"}:
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print(
                "[training] WARNING: requested device 'cuda' but CUDA is "
                "unavailable — falling back to CPU."
            )
            return torch.device("cpu")
        return torch.device(requested)
    raise TrainingRunError(f"Unsupported device {device!r}")


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move every tensor in a batch (or nested mapping) to ``device``."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, Mapping):
            out[key] = to_device(value, device)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #


def _from_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def is_distributed() -> bool:
    """Whether torch.distributed is initialized and spans > 1 process."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return torch.distributed.get_world_size()


def is_primary() -> bool:
    """The rank-0 process (the only one that writes shared artifacts)."""
    return get_rank() == 0


def setup_distributed(
    *,
    backend: str = "nccl",
    init_method: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> bool:
    """Initialize the distributed process group (single-process no-op).

    Auto-detects ``RANK`` / ``WORLD_SIZE`` (torchrun / mpirun) when not
    passed explicitly. Returns ``True`` when distributed training is active.

    Raises:
        TrainingRunError: If the requested backend is unavailable (e.g. NCCL
            on a CPU build) — the caller should fall back to single-process.
    """
    if is_distributed():
        return True
    if not torch.distributed.is_available():
        return False

    rank = _from_env("RANK", 0) if rank is None else rank
    world_size = _from_env("WORLD_SIZE", 1) if world_size is None else world_size
    if world_size <= 1:
        return False

    if backend == "nccl" and not torch.cuda.is_available():
        raise TrainingRunError(
            "NCCL backend requires CUDA; use backend='gloo' for CPU DDP",
            detail=backend,
        )
    init = init_method or f"env://"
    try:
        torch.distributed.init_process_group(
            backend=backend, init_method=init, rank=rank, world_size=world_size
        )
    except Exception as exc:
        raise TrainingRunError(f"failed to initialize process group: {exc}") from exc
    return True


def cleanup_distributed() -> None:
    if is_distributed():
        torch.distributed.destroy_process_group()


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a tensor across ranks and concatenate on dim 0 (no-op alone)."""
    if not is_distributed() or get_world_size() == 1:
        return tensor
    world = get_world_size()
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    torch.distributed.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def broadcast_dict(values: dict[str, float], src: int = 0) -> dict[str, float]:
    """Broadcast a small float dict from rank ``src`` to every rank."""
    if not is_distributed() or get_world_size() == 1:
        return dict(values)
    keys = sorted(values)
    data = torch.tensor([values[k] for k in keys], dtype=torch.float32)
    torch.distributed.broadcast(data, src=src)
    return {key: float(value) for key, value in zip(keys, data.tolist())}


# --------------------------------------------------------------------------- #
# Environment / git introspection
# --------------------------------------------------------------------------- #


def _run_git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(Path.cwd()),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def get_git_hash() -> str | None:
    """Best-effort HEAD commit hash (``None`` when not a git repository)."""
    return _run_git("rev-parse", "HEAD")


def get_git_branch() -> str | None:
    return _run_git("rev-parse", "--abbrev-ref", "HEAD")


def get_environment_info() -> dict[str, Any]:
    """Python / PyTorch / hardware environment summary for experiment logs."""
    import platform

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "world_size": get_world_size(),
        "rank": get_rank(),
    }


# --------------------------------------------------------------------------- #
# Timing & metrics helpers
# --------------------------------------------------------------------------- #


class Timer:
    """A small context-manager / wall-clock timer."""

    def __init__(self) -> None:
        self._start: float | None = None
        self.elapsed: float = 0.0

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
            self._start = None
        return self.elapsed

    def reset(self) -> "Timer":
        self.elapsed = 0.0
        self._start = None
        return self

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def format_duration(seconds: float) -> str:
    """Format a duration as ``H:MM:SS`` (or ``S.SS`` for sub-minute)."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


class MovingAverage:
    """Simple exponential / arithmetic moving average for loss tracking."""

    def __init__(self, window: int | None = None) -> None:
        self.window = window
        self._values: list[float] = []

    def update(self, value: float) -> float:
        self._values.append(float(value))
        if self.window is not None and len(self._values) > self.window:
            self._values.pop(0)
        return self.value

    @property
    def value(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def reset(self) -> None:
        self._values = []


# --------------------------------------------------------------------------- #
# Gradient checkpointing
# --------------------------------------------------------------------------- #


def apply_gradient_checkpointing(model: nn.Module, enabled: bool) -> None:
    """Wrap the memory-heavy encoder forwards with ``torch.utils.checkpoint``.

    The CropFusion image encoders (NDVI / EVI timm backbones) hold the bulk of
    the activation memory; recomputing their forward during the backward pass
    trades a small amount of compute for a large reduction in peak memory.

    Wrapping is applied only when ``enabled`` and is idempotent (a second call
    leaves already-wrapped modules untouched).
    """
    if not enabled:
        return

    def _wrap(module: nn.Module) -> None:
        if getattr(module, "_cropfusion_checkpointed", False):
            return
        original_forward = module.forward

        def checkpointed_forward(*args: Any, **kwargs: Any) -> Any:
            # ``original_forward`` is the bound method; the first positional
            # arg is the real input tensor.
            return torch.utils.checkpoint.checkpoint(
                original_forward, *args, **kwargs, use_reentrant=False
            )

        module.forward = checkpointed_forward  # type: ignore[method-assign]
        module._cropfusion_checkpointed = True  # type: ignore[attr-defined]

    for name in ("ndvi_encoder", "evi_encoder", "tab_encoder"):
        sub = getattr(model, name, None)
        if sub is not None and hasattr(sub, "forward"):
            _wrap(sub)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Move a tensor to CPU and convert to a NumPy array."""
    if not isinstance(tensor, torch.Tensor):
        return np.asarray(tensor)
    return tensor.detach().cpu().numpy()


def named_enabled_parameters(
    module: nn.Module, *, require_grad: bool = True
) -> Iterator[tuple[str, nn.Parameter]]:
    """Iterate ``(name, param)`` for parameters that require gradients."""
    for name, param in module.named_parameters():
        if not require_grad or param.requires_grad:
            yield name, param


def compute_grad_norm(
    model: nn.Module, norm_type: float = 2.0
) -> float:
    """Total gradient norm over all trainable parameters (before clipping)."""
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.detach().norm(norm_type).item()
            total += param_norm ** norm_type
    return float(total ** (1.0 / norm_type))


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count module parameters (optionally only trainable ones)."""
    return sum(
        p.numel() for p in module.parameters() if not trainable_only or p.requires_grad
    )


def estimate_parameter_memory(module: nn.Module, *, bytes_per_element: int = 4) -> int:
    """Estimated memory (bytes) for storing all parameters as float32."""
    return count_parameters(module) * bytes_per_element
