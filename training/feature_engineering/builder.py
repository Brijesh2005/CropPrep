"""Feature builder registry + feature-frame assembly.

:class:`FeatureBuilderRegistry` owns the configured per-modality builders and
merges their output into one row per observation. :func:`build_feature_frame`
turns a corpus's accepted observations (or any list of observations) into a
stable :class:`pandas.DataFrame` whose columns are the union of every row's
feature keys — missing values become ``NaN`` so downstream statistics,
balancing and export can rely on a rectangular table.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import pandas as pd

from .config import FeatureEngineeringConfig
from .exceptions import FeatureFrameError
from .image import ImageFeatureBuilder
from .logger import get_logger
from .tabular import TabularFeatureBuilder
from .temporal import TemporalFeatureBuilder

logger = get_logger("builder")


class FeatureBuilderRegistry:
    """Owns the enabled per-modality feature builders."""

    def __init__(self, config: FeatureEngineeringConfig | None = None) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.tabular_builder = TabularFeatureBuilder(self.config.tabular)
        self.image_builder = ImageFeatureBuilder(self.config.image)
        self.temporal_builder = TemporalFeatureBuilder(self.config.temporal)

    # -- Row building --------------------------------------------------------- #

    def build(
        self,
        observation: Any,
        *,
        extractor: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Merge all enabled modality builders into a single feature row.

        Raises:
            MissingExtractorError: When image patch statistics are enabled but
                no extractor was supplied.
        """
        row: dict[str, Any] = {}
        prefix = "tab" if self.config.prefixes else ""
        if self.config.tabular.enabled:
            row.update(self.tabular_builder.build(observation, prefix=prefix))
        if self.config.image.enabled:
            row.update(
                self.image_builder.build(
                    observation, prefix="img" if self.config.prefixes else "",
                    extractor=extractor,
                )
            )
        if self.config.temporal.enabled:
            row.update(
                self.temporal_builder.build(
                    observation, prefix="tmp" if self.config.prefixes else ""
                )
            )
        return row

    def build_many(
        self,
        observations: Sequence[Any],
        *,
        extractor: Callable[..., Any] | None = None,
    ) -> list[dict[str, Any]]:
        return [self.build(obs, extractor=extractor) for obs in observations]

    # -- Frame assembly ------------------------------------------------------- #

    def columns(self, observations: Sequence[Any]) -> list[str]:
        """Union of feature keys across the observations (stable order)."""
        keys: list[str] = []
        seen: set[str] = set()
        for observation in observations:
            for key in self.build(observation):
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        return keys

    def build_frame(
        self,
        observations: Sequence[Any],
        *,
        extractor: Callable[..., Any] | None = None,
    ) -> pd.DataFrame:
        """Build a rectangular feature :class:`pandas.DataFrame`.

        Raises:
            FeatureFrameError: When no observations are supplied or the rows
                cannot be assembled.
        """
        if not observations:
            raise FeatureFrameError("build_frame requires at least one observation")
        rows = self.build_many(observations, extractor=extractor)
        return pd.DataFrame.from_records(rows)


def build_features(
    observation: Any,
    config: FeatureEngineeringConfig | None = None,
    *,
    extractor: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """One-off convenience wrapper around :class:`FeatureBuilderRegistry`."""
    return FeatureBuilderRegistry(config).build(observation, extractor=extractor)


def build_feature_frame(
    observations: Sequence[Any],
    config: FeatureEngineeringConfig | None = None,
    *,
    extractor: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Build a feature :class:`pandas.DataFrame` for observations.

    Accepts either :class:`AgriculturalObservation` objects or an
    :class:`~training.stam.observation_resolver.ObservationCorpus` (accepted
    observations are used automatically).
    """
    from .utils import observations_from_corpus

    items = observations_from_corpus(observations)
    return FeatureBuilderRegistry(config).build_frame(items, extractor=extractor)
