"""PatchCache — memory-mapped 224x224 image patch cache.

Distinct from ImageEmbeddingGenerator: this caches the *raw extracted patch*
(pre-EfficientNet), useful when you need patches for more than one purpose
(e.g. embedding + explainability visualisation) without re-reading the
GeoTIFF and re-cropping every time. Uses ``np.memmap`` so large patch sets
don't need to fit in RAM at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .invalidation import CacheFingerprint
from .manager import FeatureStoreManager

_NAMESPACE = "image_embeddings"  # patches live alongside embeddings; own subfolder
_SUBDIR = "patches"


class PatchCache:
    """Caches extracted image patches (NDVI, EVI, or future indices) as
    memory-mapped ``.npy`` files.
    """

    def __init__(
        self,
        store: FeatureStoreManager,
        patch_extractor: Callable[..., Any],
        *,
        patch_size: int = 224,
        feature_version: str = "1",
    ) -> None:
        """
        Args:
            patch_extractor: Callable ``(image_path, lon, lat, size, index) -> np.ndarray``.
                Pass the existing STAM/dataset_manager patch extractor unchanged.
            index: vegetation index name, e.g. "ndvi", "evi" (extensible to future indices).
        """
        self.store = store
        self.patch_extractor = patch_extractor
        self.patch_size = patch_size
        self.feature_version = feature_version
        self._dir = Path(self.store.root) / _NAMESPACE / _SUBDIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _mmap_path(self, image_path: str, lon: float, lat: float, index: str) -> Path:
        import hashlib
        key = f"{image_path}|{lon:.5f}|{lat:.5f}|{index}|{self.patch_size}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._dir / f"{digest}.npy"

    def get_patch(
        self,
        image_path: str,
        lon: float,
        lat: float,
        *,
        index: str = "ndvi",
        dataset_version: str,
    ) -> Any:
        """Return the cached patch as a memory-mapped array, extracting +
        writing it once if not already cached.
        """
        import numpy as np

        path = self._mmap_path(image_path, lon, lat, index)
        if path.exists():
            return np.load(path, mmap_mode="r")

        patch = self.patch_extractor(image_path, lon, lat, size=self.patch_size, index=index)
        patch = np.asarray(patch)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, patch)
        return np.load(path, mmap_mode="r")

    def invalidate_all(self) -> int:
        """Delete all cached patches (call after patch size or backbone changes)."""
        removed = 0
        for f in self._dir.glob("*.npy"):
            f.unlink(missing_ok=True)
            removed += 1
        return removed
