"""Feature Store — R5 training-speed optimization (additive, non-breaking).

This package is purely additive. It does not modify Dataset Manager, STAM,
Feature Builders, or any model/training code. It exists to avoid recomputing
expensive per-epoch work (image embeddings, temporal/tabular features,
observation lookups, image patches) by caching results to disk on first
computation and loading them on every subsequent access.

Layout on disk (created under ``training/feature_store/`` by default,
configurable via ``training/config/performance.yaml``)::

    feature_store/
        image_embeddings/   # <cache_key>.npy  (768-d EfficientNet embeddings)
        temporal_sequences/ # <cache_key>.npz  (NDVI/EVI/mask/timestamps)
        tabular_cache/       # <cache_key>.parquet
        metadata/            # <cache_key>.json  (per-entry metadata sidecar)
        manifests/           # manifest.json (versions, checksums, config hash)

All caches are namespaced and keyed off content + config, so changing the
dataset version, image backbone, patch size, or preprocessing config
automatically invalidates only the affected entries (see ``invalidation.py``).

Nothing here changes model outputs, training objectives, loss functions or
evaluation metrics — it only removes redundant recomputation.
"""

from __future__ import annotations

from .manager import FeatureStoreManager, FeatureStoreConfig
from .image_cache import ImageEmbeddingGenerator
from .temporal_cache import TemporalSequenceCache
from .tabular_cache import TabularFeatureCache
from .observation_cache import ObservationCache
from .patch_cache import PatchCache
from .invalidation import CacheInvalidator, CacheFingerprint

__all__ = [
    "FeatureStoreManager",
    "FeatureStoreConfig",
    "ImageEmbeddingGenerator",
    "TemporalSequenceCache",
    "TabularFeatureCache",
    "ObservationCache",
    "PatchCache",
    "CacheInvalidator",
    "CacheFingerprint",
]
