"""timm backbone adapter: single-channel patch sequences -> feature vectors.

``TimmImageEncoder`` wraps a `timm` model (``num_classes=0`` removes the
classifier head) and adapts the Phase 4 image contract:

* input ``[B, T, 1, H, W]`` single-channel vegetation-index patches,
* single channel -> 3 channels (``repeat`` or a learnable 1x1 ``conv``),
* optional resize to the backbone's native resolution,
* per-timestep feature vectors ``[B, T, D]``.

The real output width is probed with a single forward pass at construction —
``timm.num_features`` can under-report the pooled width for some backbones
(e.g. ``mobilenetv3_small_050``), so the adapter never trusts it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .exceptions import ModelConfigurationError, ShapeMismatchError
from .interfaces import ImageEncoder


class TimmImageEncoder(ImageEncoder):
    """Per-timestep encoder built on a timm backbone.

    Args:
        backbone: A timm model name (e.g. ``efficientnetv2_s``).
        pretrained: Load ImageNet pretrained weights (needs network on first
            run).
        input_size: Square edge patches are resized to before the backbone.
            ``None`` uses the backbone's native resolution.
        channel_expansion: ``repeat`` (tile the single channel to 3) or
            ``conv`` (learnable 1x1 conv).
        freeze_backbone: Freeze every backbone weight after construction.
        drop_path_rate: Stochastic depth rate (applied when supported).

    Attributes:
        feature_dim: Width of each timestep's feature vector (probed).
    """

    def __init__(
        self,
        backbone: str,
        *,
        pretrained: bool = False,
        input_size: int | None = None,
        channel_expansion: str = "repeat",
        freeze_backbone: bool = False,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        import timm  # deferred: only required when an image encoder is built

        if channel_expansion not in ("repeat", "conv"):
            raise ModelConfigurationError(
                "channel_expansion must be 'repeat' or 'conv'",
                detail=channel_expansion,
            )

        kwargs: dict[str, Any] = {"num_classes": 0}
        if drop_path_rate and drop_path_rate > 0:
            kwargs["drop_path_rate"] = float(drop_path_rate)

        try:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, **kwargs
            )
        except Exception as exc:  # unknown name / invalid arg
            raise ModelConfigurationError(
                f"Failed to build timm backbone {backbone!r}: {exc}",
                detail=backbone,
            ) from exc

        native = self.backbone.default_cfg.get("input_size")
        if native is None or len(native) < 3:
            raise ModelConfigurationError(
                f"timm backbone {backbone!r} does not declare an input size",
                detail=native,
            )
        self._native_hw = (int(native[1]), int(native[2]))
        self.input_size: tuple[int, int] = (
            (input_size, input_size) if input_size is not None else self._native_hw
        )
        if self.input_size[0] < 1 or self.input_size[1] < 1:
            raise ModelConfigurationError(
                "input_size must be a positive integer", detail=self.input_size
            )

        self.channel_expansion = channel_expansion
        self.expand_conv: nn.Conv2d | None = (
            nn.Conv2d(1, 3, kernel_size=1, bias=False)
            if channel_expansion == "conv"
            else None
        )

        # Probe the real pooled output width (num_features can under-report).
        # BatchNorm backbones need eval mode for a single-sample probe.
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                probe = self.backbone(torch.zeros(1, 3, *self.input_size))
        finally:
            self.backbone.train(mode=was_training)
        if probe.dim() != 2:
            raise ModelConfigurationError(
                f"timm backbone {backbone!r} returned unpooled features "
                f"{tuple(probe.shape)}; only backbones that pool to a vector "
                "are supported"
            )
        self.feature_dim = int(probe.shape[-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        #: Per-timestep activation checkpointing (active only while training).
        #: Keeps the peak activation memory bounded to ``B`` frames instead of
        #: ``B * T`` so long sequences do not blow up GPU memory.
        self._checkpoint_timesteps = False

    # -- nn.Module ---------------------------------------------------------- #

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Toggle per-timestep activation checkpointing (train mode only)."""
        self._checkpoint_timesteps = bool(enabled)

    def _expand_and_resize(self, flat: torch.Tensor) -> torch.Tensor:
        """Channel-expand (C=1 -> 3) + resize for a flat ``[N, C, H, W]`` batch."""
        channels = flat.size(1)
        if channels == 1:
            if self.channel_expansion == "repeat":
                flat = flat.expand(-1, 3, -1, -1)
            else:
                flat = self.expand_conv(flat)
        elif channels != 3:
            raise ShapeMismatchError(
                f"TimmImageEncoder expects C=1 (or C=3), got C={channels}"
            )
        if (flat.size(2), flat.size(3)) != self.input_size:
            flat = F.interpolate(
                flat,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
        return flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Encode a single-channel image sequence.

        Args:
            x: ``[B, T, 1, H, W]`` (or ``[B, 3, H, W]`` for direct 3-channel).

        Returns:
            ``[B, T, feature_dim]`` per-timestep features.
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ShapeMismatchError(
                f"TimmImageEncoder expects [B, T, C, H, W], got {tuple(x.shape)}"
            )
        batch, timesteps, channels, height, width = x.shape
        if timesteps == 0:
            raise ShapeMismatchError(
                f"TimmImageEncoder received an empty sequence (T=0)"
            )

        # Fast path (eval / export / checkpointing disabled): encode the whole
        # batch in one shot — identical behaviour to previous versions.
        if timesteps == 1 or not (self.training and self._checkpoint_timesteps):
            flat = self._expand_and_resize(
                x.reshape(batch * timesteps, channels, height, width)
            )
            features = self.backbone(flat)  # [B*T, feature_dim]
            return features.reshape(batch, timesteps, self.feature_dim)

        # Memory-safe path: checkpoint each timestep's backbone forward so only
        # ``B`` frames hold live activations at any moment (recomputed on the
        # backward pass). The temporal mask already relies on per-timestep
        # independence, so this decomposition is exact.
        frame_features: list[torch.Tensor] = []
        for t in range(timesteps):
            flat_t = self._expand_and_resize(x[:, t])  # [B, 3, H, W]
            if not flat_t.is_contiguous():
                # The ``repeat`` path materialises a zero-stride broadcast view
                # (``expand``). Feeding that view to the cuDNN conv stack makes
                # the implicit contiguity copy / algorithm selection differ
                # between the checkpointed forward and its backward recompute,
                # which can trip ``torch.utils.checkpoint.CheckpointError``
                # ("different number of tensors saved") under FP16 AMP on GPU.
                flat_t = flat_t.contiguous()
            features_t = torch.utils.checkpoint.checkpoint(
                self._predict,
                self.backbone,
                flat_t,
                use_reentrant=False,
            )  # [B, feature_dim]
            frame_features.append(features_t)
        return torch.stack(frame_features, dim=1)

    @staticmethod
    def _predict(backbone: nn.Module, flat: torch.Tensor) -> torch.Tensor:
        """Stateless encode of one flat frame: ``backbone(flat)``.

        Kept pure (no ``self`` reads/writes, no expansion or resize) so the
        non-reentrant checkpoint recomputation replays an *identical*
        computation — a requirement for ``torch.utils.checkpoint``'s
        saved-tensor consistency check.
        """
        return backbone(flat)
