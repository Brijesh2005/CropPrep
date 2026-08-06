"""CropFusion feature engineering — R2.3 training-sample feature builders.

Converts the resolved
:class:`~training.stam.observation_resolver.ObservationCorpus` (produced by
the STAM :class:`~training.stam.observation_resolver.ObservationResolver`)
into flat, rectangular feature tables plus corpus statistics and class-balance
analysis. These tables feed the R2.3 exporters
(:mod:`training.export`) and the quality-control reports
(:mod:`training.quality.samples`), and the corpus itself bridges into the
existing preprocessing stack
(:func:`build_cropfusion_datasets`).

Public surface:

* :class:`FeatureBuilderRegistry`, :func:`build_features`,
  :func:`build_feature_frame` — per-modality feature rows / DataFrames.
* Per-modality builders — :class:`TabularFeatureBuilder`,
  :class:`ImageFeatureBuilder`, :class:`TemporalFeatureBuilder`.
* :class:`CorpusStatistics` — dataset statistics over the corpus.
* :class:`BalancingReport` — crop-class balance + weights.
* :func:`build_cropfusion_datasets` — corpus -> train/val/test PyTorch
  datasets (CropFusionDataset).

Example::

    from training.stam import ObservationResolver
    from training.feature_engineering import (
        build_feature_frame,
        CorpusStatistics,
        BalancingReport,
    )

    corpus = ObservationResolver(stam).resolve()       # R2.3 corpus
    frame = build_feature_frame(corpus)                # feature DataFrame
    stats = CorpusStatistics.summarize(corpus)
    balance = BalancingReport.summarize(corpus)
"""

from __future__ import annotations

from .balancing import BalancingReport
from .builder import (
    FeatureBuilderRegistry,
    build_feature_frame,
    build_features,
)
from .config import (
    FeatureEngineeringConfig,
    ImageFeatureConfig,
    TabularFeatureConfig,
    TemporalFeatureConfig,
    load_feature_engineering_config,
    save_feature_engineering_template,
)
from .dataset import build_cropfusion_datasets
from .exceptions import (
    FeatureBuilderError,
    FeatureConfigError,
    FeatureEngineeringError,
    FeatureFrameError,
    MissingExtractorError,
)
from .image import ImageFeatureBuilder
from .statistics import CorpusStatistics
from .tabular import TabularFeatureBuilder
from .temporal import TemporalFeatureBuilder

__version__ = "0.1.0"

__all__ = [
    "FeatureBuilderRegistry",
    "build_features",
    "build_feature_frame",
    "build_cropfusion_datasets",
    "TabularFeatureBuilder",
    "ImageFeatureBuilder",
    "TemporalFeatureBuilder",
    "CorpusStatistics",
    "BalancingReport",
    "FeatureEngineeringConfig",
    "TabularFeatureConfig",
    "ImageFeatureConfig",
    "TemporalFeatureConfig",
    "load_feature_engineering_config",
    "save_feature_engineering_template",
    # Exceptions
    "FeatureEngineeringError",
    "FeatureConfigError",
    "FeatureBuilderError",
    "MissingExtractorError",
    "FeatureFrameError",
]
