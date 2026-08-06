"""TabularFeatureCache — cache normalized tabular features per observation.

Wraps the existing training.feature_engineering.tabular builder /
preprocessing.tabular_pipeline; this module only stores the already-computed
result so scaling/encoding does not re-run every epoch.
"""

from __future__ import annotations

from typing import Any, Callable

from .invalidation import CacheFingerprint
from .manager import FeatureStoreManager

_NAMESPACE = "tabular_cache"


class TabularFeatureCache:
    """Caches normalized features, categorical encodings, continuous
    features, missing-value masks and scaler outputs for one observation.

    Stored as a Parquet row-group per observation (via a single-row
    DataFrame) so it stays inspectable outside of Python.
    """

    def __init__(
        self,
        store: FeatureStoreManager,
        tabular_builder: Callable[[Any], Any],
        *,
        preprocessing_config: dict[str, Any],
        feature_version: str = "1",
    ) -> None:
        """
        Args:
            tabular_builder: Callable ``(observation) -> pandas.DataFrame``
                (single row) with normalized/encoded/masked features. Pass
                your existing TabularFeatureBuilder.build(...) unchanged.
            preprocessing_config: The active tabular preprocessing config
                (scaler type, encoding scheme, imputation strategy, ...) —
                hashed into the fingerprint so config edits auto-invalidate.
        """
        self.store = store
        self.tabular_builder = tabular_builder
        self.preprocessing_hash = CacheFingerprint.hash_config(preprocessing_config)
        self.feature_version = feature_version

    def get_features(self, observation_id: str, observation: Any, *, dataset_version: str) -> Any:
        fp = CacheFingerprint(
            dataset_version=dataset_version,
            feature_version=self.feature_version,
            preprocessing_hash=self.preprocessing_hash,
        ).digest()

        if self.store.exists(_NAMESPACE, observation_id, fingerprint=fp):
            return self.store.get(_NAMESPACE, observation_id, backend="parquet")

        df = self.tabular_builder(observation)
        self.store.put(_NAMESPACE, observation_id, df, backend="parquet", fingerprint=fp)
        return df
