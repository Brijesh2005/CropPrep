"""ImageEmbeddingGenerator — compute-once, reuse-forever EfficientNet embeddings.

Pipeline: GeoTIFF -> patch extraction -> EfficientNet -> 768-d embedding ->
Feature Store. Wraps your EXISTING EfficientNet module and patch extractor
(passed in, never reimplemented here) so the model architecture is untouched.

Embeddings are recomputed only when the image, backbone, or patch config
change (see ``invalidation.py``). During curriculum stages where EfficientNet
is frozen, this lets an entire epoch skip the CNN forward pass entirely.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .invalidation import CacheFingerprint
from .manager import FeatureStoreManager

logger = logging.getLogger("training.feature_store.image")

_NAMESPACE = "image_embeddings"


class ImageEmbeddingGenerator:
    """Generates and caches 768-d image embeddings.

    Args:
        store: Shared FeatureStoreManager.
        backbone_forward: Callable ``(patch_tensor) -> embedding_tensor``.
            Pass e.g. ``lambda x: efficientnet(x)`` — this class never
            constructs or modifies the EfficientNet model itself.
        patch_extractor: Callable ``(image_path, lon, lat, size) -> patch``.
            Pass your existing STAM/dataset_manager patch extractor.
        backbone_id: String identifying backbone + checkpoint (used in the
            fingerprint so a fine-tuned checkpoint invalidates old embeddings).
    """

    def __init__(
        self,
        store: FeatureStoreManager,
        backbone_forward: Callable[[Any], Any],
        patch_extractor: Callable[..., Any],
        *,
        backbone_id: str,
        patch_size: int = 224,
    ) -> None:
        self.store = store
        self.backbone_forward = backbone_forward
        self.patch_extractor = patch_extractor
        self.backbone_id = backbone_id
        self.patch_size = patch_size

    def _key(self, image_path: str, lon: float, lat: float, date_str: str) -> str:
        return f"{image_path}|{lon:.5f}|{lat:.5f}|{date_str}"

    def get_embedding(
        self,
        image_path: str,
        lon: float,
        lat: float,
        date_str: str,
        *,
        dataset_version: str,
        feature_version: str = "1",
    ) -> Any:
        """Return the cached embedding, computing + caching it on first call."""
        fp = CacheFingerprint(
            dataset_version=dataset_version,
            feature_version=feature_version,
            image_backbone=self.backbone_id,
            patch_size=self.patch_size,
        ).digest()
        key = self._key(image_path, lon, lat, date_str)

        if self.store.exists(_NAMESPACE, key, fingerprint=fp):
            return self.store.get(_NAMESPACE, key, backend="numpy")

        patch = self.patch_extractor(image_path, lon, lat, size=self.patch_size)
        embedding = self.backbone_forward(patch)
        embedding_np = self._to_numpy(embedding)
        self.store.put(_NAMESPACE, key, embedding_np, backend="numpy", fingerprint=fp,
                        extra={"image_path": image_path, "date": date_str})
        return embedding_np

    def warm_cache(
        self,
        observations: list[Any],
        *,
        image_path_fn: Callable[[Any], str],
        coord_fn: Callable[[Any], tuple[float, float]],
        date_fn: Callable[[Any], str],
        dataset_version: str,
    ) -> dict[str, int]:
        """Precompute embeddings for a batch of observations (e.g. before
        epoch 0 of a frozen-backbone curriculum stage). Returns hit/miss counts.
        """
        before = dict(self.store.stats())
        for obs in observations:
            lon, lat = coord_fn(obs)
            self.get_embedding(
                image_path_fn(obs), lon, lat, date_fn(obs),
                dataset_version=dataset_version,
            )
        after = self.store.stats()
        return {
            "computed": after["writes"] - before.get("writes", 0),
            "reused": after["hits"] - before.get("hits", 0),
        }

    @staticmethod
    def _to_numpy(tensor: Any) -> Any:
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        return tensor
