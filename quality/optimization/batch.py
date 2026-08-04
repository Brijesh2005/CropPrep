"""Batched inference — amortise forward-pass cost across many locations.

The production engine predicts one location at a time; batch inference
collects several preprocessed samples and runs them through a single forward
pass, which is dramatically faster per sample on GPUs and vectorised CPUs.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


class BatchInferenceEngine:
    """Run many samples through one model forward pass."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | None = None,
        max_batch: int = 64,
    ) -> None:
        self.model = model.eval()
        self.device = device or (next(model.parameters()).device)
        self.max_batch = max_batch

    @torch.no_grad()
    def predict(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Predict for a list of preprocessed samples.

        Args:
            samples: Each element is a batch-style dict (may hold tensors with
                a leading batch axis or not).

        Returns:
            One output dict per input sample. Every output array keeps a
            leading batch axis of 1.
        """
        results: list[dict[str, Any]] = []
        for start in range(0, len(samples), self.max_batch):
            chunk = samples[start : start + self.max_batch]
            results.extend(self._predict_chunk(chunk))
        return results

    def _predict_chunk(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        batch = _collate(samples, self.device)
        out = self.model(batch)
        raw = out.as_dict() if hasattr(out, "as_dict") else vars(out)

        keys = [k for k in ("crop_logits", "yield_pred", "shared_representation") if raw.get(k) is not None]
        size = int(list(batch.values())[0].size(0))
        per_sample: list[dict[str, Any]] = []
        for i in range(size):
            item: dict[str, Any] = {}
            for key in keys:
                item[key] = np.asarray(raw[key][i : i + 1].detach().cpu())
            per_sample.append(item)
        return per_sample


def _collate(samples: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    """Concatenate samples on the batch axis (all tensors share shapes)."""
    keys = [k for k in samples[0] if isinstance(samples[0][k], torch.Tensor)]
    batch: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [s[key] for s in samples]
        tensors = [t if t.dim() > 0 else t.unsqueeze(0) for t in tensors]
        if tensors[0].dim() == 1:
            tensors = [t.unsqueeze(0) for t in tensors]
        batch[key] = torch.cat(tensors, dim=0).to(device)
    return batch
