"""Shared helpers for the explainability package.

* :class:`AttentionCapture` — captures self-attention weights from the
  Phase 5 transformer modules via forward pre-hooks (robust across PyTorch
  versions, independent of module internals).
* GradCAM target-layer discovery on the timm backbones.
* Single-sample batch construction from a Phase 4 sample dict.
* Feature / crop-class name mapping and yield inverse-scaling through the
  fitted preprocessor.
* Heatmap normalisation / resize and background sampling.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .exceptions import ExplainabilityError, CamError


# --------------------------------------------------------------------------- #
# Attention capture
# --------------------------------------------------------------------------- #


class AttentionCapture:
    """Capture self-attention weights from ``nn.MultiheadAttention`` modules.

    PyTorch's ``TransformerEncoderLayer`` calls ``self_attn`` with
    ``need_weights=False`` (fast SDPA path) and discards the weights. Rather
    than monkeypatching module internals, a forward **pre-hook** records the
    exact inputs; :meth:`weights` then recomputes the attention with
    ``need_weights=True``. This is robust across PyTorch versions.
    """

    def __init__(self) -> None:
        self._captures: dict[str, list[tuple[tuple, dict]]] = {}
        self._modules: dict[str, nn.Module] = {}
        self._hooks: list[Any] = []

    def register(self, module: nn.Module, name: str) -> None:
        """Install a capturing pre-hook on ``module`` under ``name``."""
        self._modules[name] = module

        def hook(_mod: nn.Module, args: tuple, kwargs: dict) -> None:
            self._captures.setdefault(name, []).append((args, dict(kwargs)))

        self._hooks.append(
            module.register_forward_pre_hook(hook, with_kwargs=True)
        )

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._captures = {}
        self._modules = {}

    def captured_names(self) -> list[str]:
        return [name for name in self._modules if name in self._captures]

    def weights(
        self,
        name: str,
        *,
        headwise: bool = False,
        index: int = -1,
    ) -> torch.Tensor | None:
        """Attention weights for one captured call.

        Args:
            name: The registered name.
            headwise: Return per-head weights ``[B, H, T, T]`` (slower).
            index: Which captured call (``-1`` = last).

        Returns:
            ``[B, T, T]`` (averaged) or ``[B, H, T, T]`` (headwise) weights,
            or ``None`` when nothing was captured.
        """
        captures = self._captures.get(name)
        module = self._modules.get(name)
        if not captures or module is None:
            return None
        args, kwargs = captures[index]
        kw = dict(kwargs)
        kw["need_weights"] = True
        kw["average_attn_weights"] = not headwise
        with torch.no_grad():
            _output, weights = module(*args, **kw)
        return weights.detach() if weights is not None else None


# --------------------------------------------------------------------------- #
# GradCAM target layer
# --------------------------------------------------------------------------- #


def find_last_spatial_conv(
    backbone: nn.Module, input_size: tuple[int, int] | None = None
) -> tuple[str, nn.Conv2d]:
    """Find the deepest ``nn.Conv2d`` whose output still has spatial extent.

    The final conv layers of a timm backbone (and ``conv_head``) often emit
    ``1x1`` maps; a useful GradCAM target needs ``H, W > 1``. Candidates are
    probed from deepest to shallowest and the first with spatial extent is
    returned.
    """
    candidates: list[tuple[str, nn.Conv2d]] = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            candidates.append((name, module))
    if not candidates:
        raise CamError("backbone has no Conv2d layer", detail=type(backbone).__name__)

    size = input_size or (32, 32)
    # Probe the deepest candidates until one has spatial extent > 1.
    for name, module in reversed(candidates):
        if "conv_head" in name:
            continue
        hw = _probe_conv_spatial(backbone, module, size)
        if hw is not None and hw[0] > 1 and hw[1] > 1:
            return name, module
    # Fall back to the deepest non-head conv.
    for name, module in reversed(candidates):
        if "conv_head" not in name:
            return name, module
    return candidates[-1]


def _probe_conv_spatial(
    backbone: nn.Module, conv: nn.Conv2d, input_size: tuple[int, int]
) -> tuple[int, int] | None:
    activations: dict[str, torch.Tensor] = {}

    def hook(_mod: nn.Module, _args: Any, output: torch.Tensor) -> None:
        activations["a"] = output

    handle = conv.register_forward_hook(hook)
    was_training = backbone.training
    backbone.eval()
    try:
        with torch.no_grad():
            backbone(torch.zeros(1, 3, *input_size))
    except Exception:
        return None
    finally:
        handle.remove()
        backbone.train(mode=was_training)
    if "a" not in activations:
        return None
    return tuple(activations["a"].shape[-2:])  # type: ignore[return-value]


def probe_spatial_shape(backbone: nn.Module, input_size: tuple[int, int]) -> tuple[int, int]:
    """Run a zero probe to learn the target layer's spatial output size."""
    name, conv = find_last_spatial_conv(backbone)
    activations: dict[str, torch.Tensor] = {}

    def hook(_mod: nn.Module, _args: Any, output: torch.Tensor) -> None:
        activations["a"] = output

    handle = conv.register_forward_hook(hook)
    was_training = backbone.training
    backbone.eval()
    try:
        with torch.no_grad():
            backbone(torch.zeros(1, 3, *input_size))
    finally:
        handle.remove()
        backbone.train(mode=was_training)
    if "a" not in activations:
        raise CamError("failed to probe GradCAM target layer", detail=name)
    return activations["a"].shape[-2:]


# --------------------------------------------------------------------------- #
# Single-sample batch
# --------------------------------------------------------------------------- #


def single_sample_batch(
    sample: Mapping[str, Any], device: torch.device | None = None
) -> dict[str, torch.Tensor]:
    """Turn a Phase 4 sample dict into a ``batch_size=1`` model batch."""
    batch: dict[str, torch.Tensor] = {}
    for key in ("tabular", "ndvi", "evi", "temporal_mask"):
        value = sample.get(key)
        if value is None or not isinstance(value, torch.Tensor):
            continue
        # Per-sample tensors always gain a leading batch dimension.
        value = value.unsqueeze(0)
        if device is not None:
            value = value.to(device)
        batch[key] = value
    return batch


def outputs_to_task(out: Any) -> dict[str, torch.Tensor]:
    """Extract task outputs (``crop`` / ``yield``) from the model output."""
    raw = out.as_dict() if hasattr(out, "as_dict") else out
    if isinstance(raw, dict):
        result: dict[str, torch.Tensor] = {}
        mapping = {"crop_logits": "crop", "yield_pred": "yield"}
        for key, value in raw.items():
            if key in mapping and value is not None:
                result[mapping[key]] = value
        return result
    return {}


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if not isinstance(tensor, torch.Tensor):
        return np.asarray(tensor)
    return tensor.detach().cpu().numpy()


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalise a heatmap to ``[0, 1]`` (handles flat maps)."""
    arr = np.asarray(heatmap, dtype="float64")
    low, high = float(arr.min()), float(arr.max())
    if high - low < 1e-12:
        return np.zeros_like(arr)
    return (arr - low) / (high - low)


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear-resize a heatmap to ``(height, width)``."""
    from torch.nn import functional as F

    target = np.asarray(heatmap, dtype="float32")
    tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


# --------------------------------------------------------------------------- #
# Preprocessor helpers
# --------------------------------------------------------------------------- #


def feature_names(preprocessor: Any) -> list[str]:
    """Ordered tabular feature names matching the model's ``[F]`` input."""
    tabular = getattr(preprocessor, "tabular", None)
    if tabular is not None:
        names = list(getattr(tabular, "feature_names", []) or [])
        if names:
            return names
    return [f"feature_{i}" for i in range(_feature_dim(preprocessor))]


def _feature_dim(preprocessor: Any) -> int:
    config = getattr(preprocessor, "config", None)
    if config is None:
        return 0
    return len(getattr(config.tabular, "numeric_features", []) or []) + len(
        getattr(config.tabular, "categorical_features", []) or []
    )


def crop_classes(preprocessor: Any) -> list[str]:
    """Crop class names (ordered by label-encoder index)."""
    label = getattr(preprocessor, "label", None)
    encoder = getattr(label, "crop_encoder", None)
    classes = getattr(encoder, "classes_", None)
    if classes is not None:
        return [str(c) for c in classes]
    return [f"crop_{i}" for i in range(int(getattr(label, "num_classes", 0) or 0))]


def inverse_scale_yield(preprocessor: Any, value: Any) -> float:
    """Map a scaled yield prediction back to physical units (t/ha)."""
    scaler = getattr(getattr(preprocessor, "label", None), "yield_scaler", None)
    if scaler is not None and hasattr(scaler, "inverse_transform"):
        arr = np.asarray([[float(value)]], dtype="float64")
        return float(np.asarray(scaler.inverse_transform(arr))[0, 0])
    return float(value)


def scale_yield(preprocessor: Any, value: Any) -> float:
    """Map a physical yield value into the scaled space used by the model."""
    scaler = getattr(getattr(preprocessor, "label", None), "yield_scaler", None)
    if scaler is not None and hasattr(scaler, "transform"):
        arr = np.asarray([[float(value)]], dtype="float64")
        return float(np.asarray(scaler.transform(arr))[0, 0])
    return float(value)


# --------------------------------------------------------------------------- #
# Background sampling
# --------------------------------------------------------------------------- #


def sample_background(
    samples: Sequence[Mapping[str, Any]], size: int, seed: int = 42
) -> list[dict[str, torch.Tensor]]:
    """Sample ``size`` reference samples for SHAP / IG baselines."""
    if size <= 0:
        return []
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(samples), size=min(size, len(samples)), replace=False)
    return [samples[int(i)] for i in indices]


def compute_probability(logits: torch.Tensor) -> np.ndarray:
    """Softmax probabilities for classification logits."""
    return to_numpy(torch.softmax(logits.float(), dim=-1))


def top_k_indices(values: Sequence[float], k: int) -> list[int]:
    """Indices of the ``k`` largest values, in descending order."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return order[:k]
