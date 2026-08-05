"""Temporal pipeline: variable-length sequences -> fixed-length tensors + mask.

Handles variable sequence length, missing/duplicate dates, ordering, padding
and truncation. Produces the ``(ndvi_seq, evi_seq, temporal_mask)`` triple
consumed by the AI module — where mask is 1 for real observations and 0 for
padding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import TemporalConfig
from .exceptions import FitError, SampleRejectedError
from .interfaces import Pipeline
from .logger import get_logger
from .utils import pad_sequence_tensors

logger = get_logger("temporal")


class TemporalPipeline(Pipeline):
    """Per-modality pipeline for temporal sequence assembly.

    Args:
        config: Temporal processing settings.
    """

    def __init__(self, config: TemporalConfig | None = None) -> None:
        self.config = config or TemporalConfig()
        self.fitted = False
        self.seq_length_stats: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, samples: Sequence[Any]) -> "TemporalPipeline":
        """Record sequence-length statistics from the training samples."""
        lengths = [int(s.num_observations()) for s in samples]
        if lengths:
            self.seq_length_stats = {
                "min": float(min(lengths)),
                "max": float(max(lengths)),
                "mean": float(np.mean(lengths)),
                "count": float(len(lengths)),
            }
        self.fitted = True
        logger.info(
            "Temporal pipeline fitted",
            extra={"max_observations": self.config.max_observations,
                   "stats": self.seq_length_stats},
        )
        return self

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #

    def transform_sequence(
        self,
        ndvi_tensors: Sequence[Any],
        evi_tensors: Sequence[Any],
        dates: Sequence[Any],
    ) -> tuple[Any, Any, Any]:
        """Assemble ordered, padded NDVI/EVI sequences + a temporal mask.

        Args:
            ndvi_tensors: Per-date ``[1, H, W]`` tensors (None entries become
                zero-fill).
            evi_tensors: Per-date ``[1, H, W]`` tensors.
            dates: Parallel observation dates.

        Returns:
            ``(ndvi_seq, evi_seq, mask)`` with shapes
            ``[max_observations, 1, H, W]``, ``[max_observations, 1, H, W]``
            and ``[max_observations]``.

        Raises:
            SampleRejectedError: When the sample has fewer than
                ``min_observations`` observations.
        """
        self._require_fitted()
        pairs = list(zip(ndvi_tensors, evi_tensors, dates))

        # Sort by date.
        if self.config.sort_by_date:
            pairs.sort(key=lambda item: _date_key(item[2]))

        # Drop duplicate dates (keep the first).
        if self.config.drop_duplicate_dates:
            seen: set = set()
            deduped: list = []
            for item in pairs:
                key = _date_key(item[2])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            pairs = deduped

        if len(pairs) < self.config.min_observations:
            raise SampleRejectedError(
                f"Sequence has {len(pairs)} observations; minimum "
                f"{self.config.min_observations} required"
            )

        max_len = self.config.max_observations
        # Reference shape for zero-fill (from either index's tensors).
        ref = next(
            (t for t in [*ndvi_tensors, *evi_tensors] if t is not None), None
        )
        ndvi_items = [p[0] if p[0] is not None else _zeros_like(ref) for p in pairs]
        evi_items = [p[1] if p[1] is not None else _zeros_like(ref) for p in pairs]

        ndvi_seq, _ = pad_sequence_tensors(
            ndvi_items, max_len, pad_value=self.config.pad_value,
            pad_side=self.config.pad_mode, truncation=self.config.truncation,
        )
        evi_seq, _ = pad_sequence_tensors(
            evi_items, max_len, pad_value=self.config.pad_value,
            pad_side=self.config.pad_mode, truncation=self.config.truncation,
        )
        mask = _build_mask(len(pairs), max_len, self.config.pad_mode)
        return ndvi_seq, evi_seq, mask

    def transform(self, observation: Any) -> Any:
        raise NotImplementedError(
            "TemporalPipeline operates on assembled tensors; use "
            "transform_sequence() from the master pipeline / dataset"
        )

    def validate(self, observation: Any) -> list[Any]:
        issues: list[str] = []
        if observation.num_observations() < self.config.min_observations:
            issues.append(
                f"too_few_observations({observation.num_observations()})"
            )
        return issues

    def summary(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "max_observations": self.config.max_observations,
            "min_observations": self.config.min_observations,
            "pad_mode": self.config.pad_mode,
            "truncation": self.config.truncation,
            "sequence_length_stats": self.seq_length_stats,
        }

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        state = {
            "max_observations": self.config.max_observations,
            "min_observations": self.config.min_observations,
            "pad_mode": self.config.pad_mode,
            "truncation": self.config.truncation,
            "seq_length_stats": self.seq_length_stats,
        }
        (out / "temporal_pipeline.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, directory: str | Path) -> "TemporalPipeline":
        state = json.loads((Path(directory) / "temporal_pipeline.json").read_text(encoding="utf-8"))
        config = TemporalConfig(
            max_observations=state["max_observations"],
            min_observations=state["min_observations"],
            pad_mode=state["pad_mode"],
            truncation=state["truncation"],
        )
        pipeline = cls(config)
        pipeline.seq_length_stats = state["seq_length_stats"]
        pipeline.fitted = True
        return pipeline

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise FitError("TemporalPipeline has not been fitted")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _date_key(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _zeros_like(reference: Any) -> Any:
    """A zero tensor matching the shape of a reference tensor."""
    import torch

    if reference is not None:
        return torch.zeros_like(reference)
    return torch.zeros(1, 1, 1, 1)


def _build_mask(count: int, max_len: int, pad_side: str) -> Any:
    from .utils import to_float_tensor

    mask = np.zeros(max_len, dtype="float32")
    mask[:count] = 1.0
    if pad_side == "left":
        mask = np.roll(mask, max_len - count) if count < max_len else mask
    return to_float_tensor(mask)
