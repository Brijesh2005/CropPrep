"""Per-timestep NDVI/EVI feature fusion.

Fuses the NDVI and EVI encoder outputs into a single per-timestep stream
instead of naively concatenating. Four configurable methods:

* ``concat`` — concatenate then project.
* ``weighted_sum`` — softmax-normalised learned scalar mix.
* ``learnable`` (default) — gated residual MLP over the concatenation.
* ``attention`` — softmax attention over the two modality vectors.

Inputs ``[B, T, D_ndvi]`` / ``[B, T, D_evi]`` -> output ``[B, T, out_dim]``.
"""

from __future__ import annotations

import torch
from torch import nn

from .exceptions import ModelConfigurationError
from .utils import get_activation

_FUSION_METHODS = ("concat", "weighted_sum", "learnable", "attention")

#: Default fusion working width when ``hidden_dim`` is not set — capped so a
#: wide timm backbone (e.g. 1280 for EfficientNetV2-S) does not force a
#: multi-million-parameter fusion block; 512 is ample for per-timestep
#: modality mixing.
_DEFAULT_FUSION_WIDTH = 512


class ImageFusion(nn.Module):
    """Fuse NDVI and EVI features at every timestep.

    Args:
        ndvi_dim: NDVI encoder feature width.
        evi_dim: EVI encoder feature width.
        method: One of ``concat`` | ``weighted_sum`` | ``learnable`` |
            ``attention``.
        hidden_dim: Fusion output width (``None`` = ``max(ndvi_dim, evi_dim)``).
        dropout: Dropout applied to the fused output.
        activation: Activation used inside the fusion blocks.

    Attributes:
        out_dim: Width of the fused sequence fed to the temporal transformer.
    """

    def __init__(
        self,
        ndvi_dim: int,
        evi_dim: int,
        *,
        method: str = "learnable",
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if method not in _FUSION_METHODS:
            raise ModelConfigurationError(
                f"unknown image fusion method {method!r}; choose from "
                f"{_FUSION_METHODS}"
            )
        self.method = method

        if hidden_dim is not None:
            base_dim = hidden_dim
        else:
            base_dim = min(max(ndvi_dim, evi_dim), _DEFAULT_FUSION_WIDTH)
        if base_dim < 1:
            raise ModelConfigurationError(
                "image fusion hidden_dim must be positive", detail=base_dim
            )
        act = get_activation(activation)

        # Single-stream mode: one of the modalities is disabled (ablation).
        # A simple projector replaces the two-stream mixing so the temporal
        # transformer still receives a ``[B, T, out_dim]`` stream.
        self.single_stream = ndvi_dim == 0 or evi_dim == 0
        if self.single_stream:
            stream_dim = ndvi_dim if ndvi_dim > 0 else evi_dim
            self.ndvi_proj = None
            self.evi_proj = None
            self.out_dim = base_dim
            self.combine = nn.Sequential(
                nn.Linear(stream_dim, self.out_dim), act, nn.Dropout(dropout)
            )
            return

        self.ndvi_proj = (
            nn.Linear(ndvi_dim, base_dim) if ndvi_dim != base_dim else nn.Identity()
        )
        self.evi_proj = (
            nn.Linear(evi_dim, base_dim) if evi_dim != base_dim else nn.Identity()
        )
        self.out_dim = base_dim

        if method == "concat":
            self.combine = nn.Sequential(
                nn.Linear(2 * base_dim, self.out_dim),
                act,
                nn.Dropout(dropout),
            )
        elif method == "weighted_sum":
            self.mix = nn.Parameter(torch.tensor([0.5, 0.5]))
            self.combine = nn.Sequential(
                nn.Linear(base_dim, self.out_dim),
                act,
                nn.Dropout(dropout),
            )
        elif method == "learnable":
            self.proj_a = nn.Linear(2 * base_dim, 2 * base_dim)
            self.proj_b = nn.Linear(2 * base_dim, 2 * base_dim)
            self.gate = nn.Linear(2 * base_dim, 2 * base_dim)
            self.activation = act
            self.combine = nn.Sequential(
                nn.Linear(2 * base_dim, self.out_dim),
                act,
                nn.Dropout(dropout),
            )
        elif method == "attention":
            self.score = nn.Linear(2 * base_dim, 2)
            self.combine = nn.Sequential(
                nn.Linear(base_dim, self.out_dim),
                act,
                nn.Dropout(dropout),
            )

    # -- nn.Module ---------------------------------------------------------- #

    def forward(  # type: ignore[override]
        self, ndvi: torch.Tensor, evi: torch.Tensor
    ) -> torch.Tensor:
        """Fuse NDVI and EVI features.

        Args:
            ndvi: ``[B, T, D_ndvi]`` NDVI per-timestep features.
            evi: ``[B, T, D_evi]`` EVI per-timestep features.

        Returns:
            ``[B, T, out_dim]`` fused per-timestep features.
        """
        if self.single_stream:
            stream = ndvi if ndvi is not None else evi
            return self.combine(stream)

        ndvi_p = self.ndvi_proj(ndvi)
        evi_p = self.evi_proj(evi)

        if self.method == "concat":
            return self.combine(torch.cat([ndvi_p, evi_p], dim=-1))

        if self.method == "weighted_sum":
            weights = torch.softmax(self.mix, dim=0)
            mixed = weights[0] * ndvi_p + weights[1] * evi_p
            return self.combine(mixed)

        if self.method == "learnable":
            h = torch.cat([ndvi_p, evi_p], dim=-1)
            projected = self.proj_b(self.activation(self.proj_a(h)))
            gate = torch.sigmoid(self.gate(h))
            z = gate * projected + (1.0 - gate) * h
            return self.combine(z)

        # attention
        h = torch.cat([ndvi_p, evi_p], dim=-1)
        weights = torch.softmax(self.score(h), dim=-1)  # [B, T, 2]
        mixed = weights[..., 0:1] * ndvi_p + weights[..., 1:2] * evi_p
        return self.combine(mixed)
