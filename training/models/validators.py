"""Input and configuration validation for the AI model.

``validate_batch`` enforces the Phase 4 batch contract (shapes / dtypes /
value ranges) before a forward pass, and ``expected_batch_shapes`` documents
the exact tensor contract the architecture expects. Both raise
:class:`~ai.models.exceptions.ModelError` subclasses on violation.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .config import ModelConfig
from .exceptions import ModelConfigurationError, ModelInputError, ShapeMismatchError

__all__ = ["validate_batch", "expected_batch_shapes", "validate_model_config"]


def validate_model_config(config: ModelConfig) -> None:
    """Cross-field semantic checks beyond pydantic field validation."""
    if config.uses_tabular:
        if config.tabular.embedding_dim % config.tabular.num_heads != 0:
            raise ModelConfigurationError(
                "tabular.embedding_dim must be divisible by tabular.num_heads"
            )
    if config.uses_image:
        temporal = config.temporal
        if temporal.d_model % temporal.num_heads != 0:
            raise ModelConfigurationError(
                "temporal.d_model must be divisible by temporal.num_heads"
            )
        if temporal.embedding_dim < 1:
            raise ModelConfigurationError("temporal.embedding_dim must be positive")
    if config.uses_tabular and config.uses_image:
        if config.cross_attention.num_heads < 1:
            raise ModelConfigurationError(
                "cross_attention.num_heads must be at least 1"
            )
    if config.shared_encoder.d_model % config.shared_encoder.num_heads != 0:
        raise ModelConfigurationError(
            "shared_encoder.d_model must be divisible by shared_encoder.num_heads"
        )


def expected_batch_shapes(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Document the exact batch tensor shapes for a model config.

    ``0`` denotes a free batch dimension; ``None`` a per-sample-dependent one.
    Only the enabled modalities appear (ablation-aware).
    """
    shapes: dict[str, tuple[int, ...]] = {}
    if config.uses_tabular:
        shapes["tabular"] = (0, config.tabular_feature_dim)
    if config.uses_image:
        if config.image_encoder.enable_ndvi:
            shapes["ndvi"] = (0, config.temporal.max_len, 1, None, None)
        if config.image_encoder.enable_evi:
            shapes["evi"] = (0, config.temporal.max_len, 1, None, None)
        shapes["temporal_mask"] = (0, config.temporal.max_len)
    shapes["crop_label"] = (0,)
    shapes["yield_label"] = (0,)
    return shapes


def validate_batch(batch: Mapping[str, Any], config: ModelConfig) -> None:
    """Validate a Phase 4 batch dict against the model configuration.

    Args:
        batch: A batch as produced by
            :func:`ai.preprocessing.dataloader.collate_samples`.
        config: The model configuration describing the expected contract.

    Raises:
        ModelInputError: On missing keys or invalid dtypes.
        ShapeMismatchError: On inconsistent tensor shapes.
    """
    if not isinstance(batch, Mapping):
        raise ModelInputError(
            "batch must be a mapping (dict) of tensors", detail=type(batch).__name__
        )

    if config.uses_tabular:
        tabular = batch.get("tabular")
        if tabular is None:
            raise ModelInputError("batch is missing 'tabular' (tabular branch enabled)")
        if not isinstance(tabular, torch.Tensor):
            raise ModelInputError("batch['tabular'] must be a torch.Tensor")
        if tabular.dim() != 2:
            raise ShapeMismatchError(
                f"tabular must be [B, F], got {tuple(tabular.shape)}"
            )
        expected_f = config.tabular_feature_dim
        if tabular.size(1) != expected_f:
            raise ShapeMismatchError(
                f"tabular feature dim {tabular.size(1)} != config "
                f"{expected_f} (numeric_dim + categorical cardinalities)",
                detail={"got": tabular.size(1), "expected": expected_f},
            )

    if config.uses_image:
        present_image_keys: list[str] = []
        for key, enabled in (
            ("ndvi", config.image_encoder.enable_ndvi),
            ("evi", config.image_encoder.enable_evi),
        ):
            if not enabled:
                continue
            tensor = batch.get(key)
            if tensor is None:
                raise ModelInputError(
                    f"batch is missing '{key}' (image stream enabled)"
                )
            if not isinstance(tensor, torch.Tensor):
                raise ModelInputError(f"batch['{key}'] must be a torch.Tensor")
            if tensor.dim() != 5:
                raise ShapeMismatchError(
                    f"{key} must be [B, T, 1, H, W], got {tuple(tensor.shape)}"
                )
            if tensor.size(2) != 1:
                raise ShapeMismatchError(
                    f"{key} must be single-channel (C=1), got C={tensor.size(2)}"
                )
            present_image_keys.append(key)

        mask = batch.get("temporal_mask")
        if mask is None:
            raise ModelInputError(
                "batch is missing 'temporal_mask' (image branch enabled)"
            )
        if not isinstance(mask, torch.Tensor):
            raise ModelInputError("batch['temporal_mask'] must be a torch.Tensor")
        if mask.dim() != 2:
            raise ShapeMismatchError(
                f"temporal_mask must be [B, T], got {tuple(mask.shape)}"
            )
        batch_t = batch[present_image_keys[0]].size(1)
        max_len = config.temporal.max_len
        if batch_t > max_len:
            raise ShapeMismatchError(
                f"sequence length {batch_t} exceeds temporal.max_len {max_len}; "
                "increase temporal.max_len in the model config"
            )
        for key in present_image_keys:
            if batch[key].size(1) != batch_t:
                raise ShapeMismatchError(
                    f"{key} sequence length {batch[key].size(1)} != {batch_t}"
                )
        if mask.size(1) != batch_t:
            raise ShapeMismatchError(
                f"temporal_mask length {mask.size(1)} != image sequence "
                f"length {batch_t}"
            )
        batch_dims = {int(batch[key].size(0)) for key in present_image_keys} | {
            int(mask.size(0))
        }
        if len(batch_dims) != 1:
            raise ShapeMismatchError("ndvi/evi/temporal_mask batch dims differ")
