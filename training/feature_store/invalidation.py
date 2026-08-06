"""Cache invalidation — fingerprints that gate reuse of cached features.

A cache entry is reused ONLY when its stored fingerprint matches the current
fingerprint for that namespace. The fingerprint is a hash of exactly the
inputs that legitimately change the cached output:

* dataset version   (Dataset Manager's dataset version string)
* image backbone    (EfficientNet variant + checkpoint hash, image caches only)
* patch size         (image / patch caches only)
* preprocessing config (normalization, augmentation-affecting settings)
* feature version    (bump manually to force a full rebuild)

Nothing else (e.g. batch size, learning rate, epoch number) should ever be
part of a fingerprint — including them would invalidate caches for no reason.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .manager import FeatureStoreManager


@dataclass
class CacheFingerprint:
    dataset_version: str
    feature_version: str = "1"
    image_backbone: str | None = None
    patch_size: int | None = None
    preprocessing_hash: str | None = None

    def digest(self) -> str:
        payload = {
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "image_backbone": self.image_backbone,
            "patch_size": self.patch_size,
            "preprocessing_hash": self.preprocessing_hash,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:24]

    @staticmethod
    def hash_config(config: dict[str, Any]) -> str:
        """Stable hash of an arbitrary preprocessing/model config dict."""
        blob = json.dumps(config, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:24]


class CacheInvalidator:
    """Convenience wrapper: compares current vs stored fingerprints and
    invalidates whole namespaces when they drift, rather than requiring
    every caller to special-case fingerprint mismatches.
    """

    def __init__(self, store: FeatureStoreManager) -> None:
        self.store = store
        self._last_fingerprint: dict[str, str] = {}

    def check_and_invalidate(self, namespace: str, fingerprint: CacheFingerprint) -> bool:
        """Invalidate `namespace` if its recorded fingerprint changed.

        Returns True if invalidation occurred.
        """
        digest = fingerprint.digest()
        sentinel_key = f"__fingerprint__:{namespace}"
        entries = self.store._manifest.get("entries", {})  # noqa: SLF001 (same package)
        mkey = self.store._manifest_key("manifests", sentinel_key)
        stored = entries.get(mkey, {}).get("fingerprint")
        if stored == digest:
            return False
        removed = self.store.invalidate(namespace)
        self.store.put("manifests", sentinel_key, {"fingerprint": digest}, backend="json",
                        fingerprint=digest)
        if removed:
            import logging
            logging.getLogger("training.feature_store").info(
                "Invalidated %d cached entries in '%s' (fingerprint changed: %s -> %s)",
                removed, namespace, stored, digest,
            )
        return True
