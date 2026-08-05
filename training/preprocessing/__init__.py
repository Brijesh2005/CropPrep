"""CropFusion preprocessing / feature-engineering pipeline.

Converts :class:`~training.stam.observation.AgriculturalObservation`
samples (produced by STAM) into **AI-ready tensors** consumed directly by
PyTorch DataLoaders.

Public surface:

* :class:`Preprocessor` — master pipeline (``fit`` / ``transform`` /
  ``fit_transform`` / ``validate`` / ``summary`` / ``save`` / ``load``).
* Per-modality pipelines — :class:`TabularPipeline`,
  :class:`ImagePipeline`, :class:`TemporalPipeline`, :class:`LabelPipeline`.
* :class:`CropFusionDataset` + :func:`split_observations` (leakage-free).
* :func:`build_dataloader` + :func:`collate_samples`.
* :class:`DatasetStatistics` / :class:`StatisticsReport`.
* Scalers/encoders — :class:`StandardScaler`, :class:`MinMaxScaler`,
  :class:`RobustScaler`, :class:`OrdinalEncoder`, :class:`OneHotEncoder`,
  :class:`LabelEncoder`.

Example::

    from training.preprocessing import Preprocessor, CropFusionDataset, build_dataloader, split_observations

    preprocessor = Preprocessor.from_config()
    train, val, test = split_observations(accepted, config.split)
    preprocessor.fit(train, extractor=stam.get_patch)

    train_loader = build_dataloader(
        CropFusionDataset.build(preprocessor, train, split="train", extractor=stam.get_patch),
        config, split="train",
    )
    for batch in train_loader:
        # batch["tabular"], batch["ndvi"], batch["evi"], batch["temporal_mask"],
        # batch["crop_label"], batch["yield_label"]
        ...
"""

from __future__ import annotations

from .augmentations import ImageAugmentation
from .config import (
    AugmentationConfig,
    DataloaderConfig,
    ImageConfig,
    LabelConfig,
    PreprocessingConfig,
    QualityConfig,
    SplitConfig,
    TabularConfig,
    TemporalConfig,
    load_preprocessing_config,
    save_preprocessing_template,
)
from .dataloader import build_dataloader, collate_samples
from .dataset import CropFusionDataset, split_observations
from .exceptions import (
    ArtifactError,
    ConfigurationError,
    FitError,
    MissingDependencyError,
    PreprocessingError,
    SampleRejectedError,
    ShapeMismatchError,
)
from .image_pipeline import ImagePipeline
from .label_pipeline import LabelPipeline
from .master_pipeline import Preprocessor
from .statistics import DatasetStatistics, StatisticsReport
from .tabular_pipeline import TabularPipeline
from .temporal_pipeline import TemporalPipeline
from .transforms import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from .validators import FilterDecision, filter_observation, filter_observations

__version__ = "0.1.0"

__all__ = [
    "Preprocessor",
    "PreprocessingConfig",
    "load_preprocessing_config",
    "save_preprocessing_template",
    "TabularPipeline",
    "ImagePipeline",
    "TemporalPipeline",
    "LabelPipeline",
    "CropFusionDataset",
    "split_observations",
    "build_dataloader",
    "collate_samples",
    "DatasetStatistics",
    "StatisticsReport",
    "ImageAugmentation",
    "filter_observation",
    "filter_observations",
    "FilterDecision",
    # Configs
    "TabularConfig",
    "ImageConfig",
    "TemporalConfig",
    "LabelConfig",
    "SplitConfig",
    "AugmentationConfig",
    "DataloaderConfig",
    "QualityConfig",
    # Transforms
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "OrdinalEncoder",
    "OneHotEncoder",
    "LabelEncoder",
    # Exceptions
    "PreprocessingError",
    "ConfigurationError",
    "SampleRejectedError",
    "MissingDependencyError",
    "FitError",
    "ArtifactError",
    "ShapeMismatchError",
]
