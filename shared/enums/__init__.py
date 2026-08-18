"""Canonical enumerations shared across the CropFusion platforms.

These enums form the vocabulary used by both the Training and Application
platforms.  They were extracted from ``training.dataset_manager.models`` and
``training.stam`` so that neither platform needs to import from the other.
"""

from __future__ import annotations

import enum


class IndexType(str, enum.Enum):
    """Vegetation index encoded in a raster file (detected from paths/names)."""

    NDVI = "NDVI"
    EVI = "EVI"
    NONE = "NONE"


class Resolution(str, enum.Enum):
    """Spatial resolution band (Sentinel-2 10m / 20m products)."""

    R10M = "R10m"
    R20M = "R20m"
    UNKNOWN = "UNKNOWN"


class FileCategory(str, enum.Enum):
    """High-level file classification produced by the scanner."""

    CSV = "csv"
    GEOTIFF = "geotiff"
    OTHER = "other"


class Severity(str, enum.Enum):
    """Severity of a validation issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DatasetStatus(str, enum.Enum):
    """Lifecycle status of a registered dataset."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    VALIDATING = "validating"
    VALIDATED = "validated"
    READY = "ready"
    FAILED = "failed"
    STALE = "stale"


class ValidationStatus(str, enum.Enum):
    """Overall outcome of a validation run."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


class CropType(str, enum.Enum):
    """Canonical crop types used by prediction and training metadata.

    Class IDs are assigned by the :class:`~training.preprocessing.transforms.LabelEncoder`
    at data-fit time (first-seen order), **not** by enum member order.  New
    members may be appended without shifting existing integer class IDs or
    invalidating checkpoints / label_encoder.pkl artifacts.
    """

    RICE = "rice"
    PADDY = "paddy"
    RAGI = "ragi"
    MAIZE = "maize"
    COCONUT = "coconut"
    ARECANUT = "arecanut"
    PEPPER = "pepper"
    COFFEE = "coffee"
    CARDAMOM = "cardamom"
    BLACKGRAM = "blackgram"
    OTHER = "other"
    UNKNOWN = "unknown"


class Season(str, enum.Enum):
    """Canonical cropping-season vocabulary.

    The STAM :class:`~training.stam.temporal_index.SeasonCalendar` remains
    configuration-driven (``seasons.yaml``); this enum provides the canonical
    names shared across platforms for metadata and reporting.
    """

    KHARIF = "kharif"
    RABI = "rabi"
    SUMMER = "summer"
    UNKNOWN = "unknown"


class ModelStatus(str, enum.Enum):
    """Lifecycle status of a trained model artifact."""

    PENDING = "pending"
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class TrainingStage(str, enum.Enum):
    """Stage of a training run."""

    INIT = "init"
    DATA = "data"
    TRAIN = "train"
    VALIDATE = "validate"
    SAVE = "save"
    FINISHED = "finished"


class ReleaseStatus(str, enum.Enum):
    """Status of a versioned release (dataset, model or application)."""

    DRAFT = "draft"
    CANDIDATE = "candidate"
    RELEASED = "released"
    DEPRECATED = "deprecated"
    RETRACTED = "retracted"


class ProviderType(str, enum.Enum):
    """Kind of data provider."""

    TABULAR = "tabular"
    IMAGE = "image"
    BOUNDARY = "boundary"
    UNKNOWN = "unknown"


class EnvironmentType(str, enum.Enum):
    """Runtime environment name."""

    DEV = "dev"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


# Issues of at least this weight cause a validation report to fail.
FAILING_SEVERITY = frozenset({Severity.ERROR, Severity.CRITICAL})

__all__ = [
    "CropType",
    "DatasetStatus",
    "EnvironmentType",
    "FAILING_SEVERITY",
    "FileCategory",
    "IndexType",
    "ModelStatus",
    "ProviderType",
    "ReleaseStatus",
    "Resolution",
    "Season",
    "Severity",
    "TrainingStage",
    "ValidationStatus",
]
