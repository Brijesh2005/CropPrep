"""Image augmentations for the training set only.

Augmentations operate on ``[T, C, H, W]`` image tensors (or single patches)
so the same transform can be applied per-sequence or per-patch. They are
never applied to validation/test data — the master pipeline gates them on the
split.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .config import AugmentationConfig

#: ImageNet-ish constants used for brightness/contrast jitter scaling.
_BRIGHTNESS_FACTOR = 0.3
_CONTRAST_FACTOR = 0.3


@dataclass(slots=True)
class ImageAugmentation:
    """Random geometric + photometric augmentation of image tensors.

    Args:
        config: Augmentation settings (disabled by default).
        seed: Optional RNG seed for reproducible runs.
    """

    config: AugmentationConfig
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    enabled: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.enabled = self.config.enabled and (
            self.config.flip_horizontal
            or self.config.flip_vertical
            or bool(self.config.rotation_degrees)
            or self.config.random_crop
            or self.config.brightness_jitter > 0
            or self.config.contrast_jitter > 0
            or self.config.noise_std > 0
        )

    def __call__(self, tensor: Any) -> Any:
        """Apply augmentation to a ``[T, C, H, W]`` (or ``[C, H, W]``) tensor."""
        if not self.enabled:
            return tensor

        import torch

        squeeze = tensor.dim() == 3
        if squeeze:
            tensor = tensor.unsqueeze(0)
        work = tensor

        if self.config.flip_horizontal and self._rng.random() < 0.5:
            work = torch.flip(work, dims=[-1])
        if self.config.flip_vertical and self._rng.random() < 0.5:
            work = torch.flip(work, dims=[-2])
        if self.config.rotation_degrees:
            angle = self._rng.choice(self.config.rotation_degrees)
            if angle != 0:
                work = _rotate(work, angle)
        if self.config.random_crop:
            work = _random_crop(work, self.config.crop_fraction, self._rng)
        if self.config.brightness_jitter > 0:
            factor = 1.0 + self._rng.uniform(-1, 1) * _BRIGHTNESS_FACTOR * self.config.brightness_jitter
            work = work * factor
        if self.config.contrast_jitter > 0:
            factor = 1.0 + self._rng.uniform(-1, 1) * _CONTRAST_FACTOR * self.config.contrast_jitter
            mean = work.mean(dim=(-2, -1), keepdim=True)
            work = (work - mean) * factor + mean
        if self.config.noise_std > 0:
            noise = torch.randn_like(work) * self.config.noise_std
            work = work + noise

        return work[0] if squeeze else work


def _rotate(tensor: Any, angle: float) -> Any:
    """Rotate a ``[T, C, H, W]`` tensor by a multiple of 90 degrees."""
    import torch

    rotations = int(round(angle / 90.0)) % 4
    return torch.rot90(tensor, k=rotations, dims=[-2, -1])


def _random_crop(tensor: Any, fraction: float, rng: random.Random) -> Any:
    """Randomly crop ``fraction`` of the spatial extent, then resize back."""
    import torch
    import torch.nn.functional as fn

    _, _, height, width = tensor.shape
    crop_h = max(1, int(height * fraction))
    crop_w = max(1, int(width * fraction))
    top = rng.randint(0, max(0, height - crop_h))
    left = rng.randint(0, max(0, width - crop_w))
    cropped = tensor[:, :, top: top + crop_h, left: left + crop_w]
    return fn.interpolate(
        cropped, size=(height, width), mode="bilinear", align_corners=False
    )
