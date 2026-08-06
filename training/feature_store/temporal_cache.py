"""TemporalSequenceCache — cache assembled NDVI/EVI temporal sequences.

Wraps whatever already builds these sequences (STAM's sequence_builder /
training.feature_engineering.temporal) so the assembly logic itself is
untouched. This module only avoids re-running that assembly every epoch.
"""

from __future__ import annotations

from typing import Any, Callable

from .invalidation import CacheFingerprint
from .manager import FeatureStoreManager

_NAMESPACE = "temporal_sequences"


class TemporalSequenceCache:
    """Caches, per observation, the ordered NDVI/EVI sequence + metadata.

    The cached payload is an ``.npz`` archive with keys:
    ``ndvi``, ``evi``, ``timestamps``, ``quality``, ``cloud``, ``mask``.
    """

    def __init__(
        self,
        store: FeatureStoreManager,
        sequence_builder: Callable[[Any], dict[str, Any]],
        *,
        feature_version: str = "1",
    ) -> None:
        """
        Args:
            store: Shared FeatureStoreManager.
            sequence_builder: Callable ``(observation) -> dict`` returning at
                least ``ndvi, evi, timestamps, quality, cloud, mask`` arrays.
                Pass your existing STAM sequence builder unchanged.
        """
        self.store = store
        self.sequence_builder = sequence_builder
        self.feature_version = feature_version

    def get_sequence(self, observation_id: str, observation: Any, *, dataset_version: str) -> dict[str, Any]:
        fp = CacheFingerprint(
            dataset_version=dataset_version,
            feature_version=self.feature_version,
        ).digest()

        if self.store.exists(_NAMESPACE, observation_id, fingerprint=fp):
            return self.store.get(_NAMESPACE, observation_id, backend="numpy_npz")

        sequence = self.sequence_builder(observation)
        payload = {
            "ndvi": sequence["ndvi"],
            "evi": sequence["evi"],
            "timestamps": sequence["timestamps"],
            "quality": sequence.get("quality"),
            "cloud": sequence.get("cloud"),
            "mask": sequence["mask"],
        }
        # npz cannot store None values — drop absent optional fields.
        payload = {k: v for k, v in payload.items() if v is not None}
        self.store.put(_NAMESPACE, observation_id, payload, backend="numpy_npz", fingerprint=fp)
        return payload
