"""ObservationCache — avoid repeated STAM observation reconstruction.

STAM already has its own SQLite-backed cache (training/stam/cache.py) for
spatial/temporal lookups. This class is a thin, feature-store-backed layer
specifically for the fully-assembled ``AgriculturalObservation`` objects (and
their historical context) so the *training loop* never re-triggers STAM's
assembly path once an observation has been built for a given epoch-0 pass.

It composes with, and does not replace, STAM's own cache.
"""

from __future__ import annotations

from typing import Any, Callable

from .invalidation import CacheFingerprint
from .manager import FeatureStoreManager

_NAMESPACE = "metadata"  # observations are metadata-shaped (json-serializable)


class ObservationCache:
    """Caches assembled observations + historical context by observation id."""

    def __init__(
        self,
        store: FeatureStoreManager,
        observation_builder: Callable[[str], Any],
        *,
        serialize: Callable[[Any], dict[str, Any]],
        deserialize: Callable[[dict[str, Any]], Any],
        feature_version: str = "1",
    ) -> None:
        """
        Args:
            observation_builder: Callable ``(observation_id) -> AgriculturalObservation``.
                Pass STAM's existing resolver/builder unchanged.
            serialize / deserialize: Convert AgriculturalObservation <-> dict,
                since the feature store's JSON backend needs plain data.
                Use ``AgriculturalObservation.to_dict`` / ``.from_dict`` (or
                equivalents) already provided by STAM — do not reimplement.
        """
        self.store = store
        self.observation_builder = observation_builder
        self.serialize = serialize
        self.deserialize = deserialize
        self.feature_version = feature_version

    def get_observation(self, observation_id: str, *, dataset_version: str) -> Any:
        fp = CacheFingerprint(
            dataset_version=dataset_version,
            feature_version=self.feature_version,
        ).digest()

        if self.store.exists(_NAMESPACE, observation_id, fingerprint=fp):
            data = self.store.get(_NAMESPACE, observation_id, backend="json")
            return self.deserialize(data)

        observation = self.observation_builder(observation_id)
        self.store.put(_NAMESPACE, observation_id, self.serialize(observation),
                        backend="json", fingerprint=fp)
        return observation

    def warm_cache(self, observation_ids: list[str], *, dataset_version: str) -> dict[str, int]:
        before = dict(self.store.stats())
        for obs_id in observation_ids:
            self.get_observation(obs_id, dataset_version=dataset_version)
        after = self.store.stats()
        return {
            "computed": after["writes"] - before.get("writes", 0),
            "reused": after["hits"] - before.get("hits", 0),
        }
