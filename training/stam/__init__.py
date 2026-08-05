"""CropFusion Spatial-Temporal Alignment Module (STAM).

STAM is the research contribution that fuses **tabular data** with
**multi-temporal Sentinel-2 vegetation indices** into one unified multimodal
agricultural observation. It is the only entry point between raw datasets
(accessed strictly through the Dataset Manager) and the AI pipeline:

* Every training sample passes through :meth:`STAM.build_observation`.
* No AI model, GIS service or backend reads CSVs / GeoTIFFs directly.

Public surface:

* :class:`STAM` — the facade (``initialize``, ``build_observation``,
  ``find_nearest``, ``build_sequence``, ``get_patch``, ``validate``,
  ``summary``).
* :class:`AgriculturalObservation` — the strongly-typed output sample.
* :class:`SeasonResolver` — resolves the season from the calendar date
  (YAML-configurable), powering the location-only farmer workflow.
* :class:`HistoricalContext` — multi-year "same location + season" context
  attached to every observation before inference.
* Configuration (:class:`StamConfig`, :func:`load_stam_config`).
* Exceptions, indexes, matchers, patch generator, validators and cache.

Example::

    from training.dataset_manager import DatasetManager
    from training.stam import STAM

    manager = DatasetManager.from_config()
    stam = STAM(manager)
    stam.initialize()
    obs = stam.build_observation(lon=74.87, lat=13.09)  # farmer: location only
    print(obs.crop, obs.yield_value, obs.historical_context.years)
"""

from __future__ import annotations

from .cache import DatasetManagerStamCache
from .config import StamConfig, load_stam_config, save_stam_config_template
from .coordinate_transform import (
    geographic_to_raster_index,
    normalise_crs,
    transform_point,
)
from .exceptions import (
    BoundaryNotFoundError,
    CRSMismatchError,
    InvalidCoordinatesError,
    LocationNotFoundError,
    NoImageRecordError,
    NoTabularRecordError,
    NotInitializedError,
    PairingError,
    PatchOutOfBoundsError,
    ResolutionMismatchError,
    StamConfigurationError,
    StamError,
    TemporalGapError,
)
from .historical_context import HistoricalContextBuilder
from .matcher import SpatialTemporalMatcher
from .observation import (
    AdminLocation,
    AgriculturalObservation,
    GeographicPoint,
    HistoricalContext,
    ImagePairRef,
    ImageRecordRef,
    LocationInfo,
    QualityIssue,
    QualityReport,
    SequenceInfo,
    TabularFeatures,
    TemporalInfo,
)
from .patch_generator import RasterPatch, SpatialPatchGenerator
from .season_resolver import SeasonResolver
from .sequence_builder import (
    ImagePairBuilder,
    ObservationSequenceBuilder,
    SequenceBuildResult,
)
from .spatial_index import BoundaryIndex, KDTreeSpatialIndex, NearestMatch
from .stam import STAM
from .temporal_index import Season, SeasonCalendar, TemporalIndex

__version__ = "0.1.0"

__all__ = [
    "STAM",
    "StamConfig",
    "load_stam_config",
    "save_stam_config_template",
    "SpatialTemporalMatcher",
    "KDTreeSpatialIndex",
    "BoundaryIndex",
    "SeasonCalendar",
    "TemporalIndex",
    "ObservationSequenceBuilder",
    "ImagePairBuilder",
    "SequenceBuildResult",
    "SpatialPatchGenerator",
    "RasterPatch",
    "DatasetManagerStamCache",
    "SeasonResolver",
    "HistoricalContextBuilder",
    # Observation models
    "AgriculturalObservation",
    "LocationInfo",
    "AdminLocation",
    "GeographicPoint",
    "TemporalInfo",
    "TabularFeatures",
    "SequenceInfo",
    "ImagePairRef",
    "ImageRecordRef",
    "QualityIssue",
    "QualityReport",
    "HistoricalContext",
    "Season",
    "NearestMatch",
    # Utilities
    "normalise_crs",
    "transform_point",
    "geographic_to_raster_index",
    # Exceptions
    "StamError",
    "StamConfigurationError",
    "InvalidCoordinatesError",
    "LocationNotFoundError",
    "BoundaryNotFoundError",
    "NoTabularRecordError",
    "NoImageRecordError",
    "PairingError",
    "CRSMismatchError",
    "ResolutionMismatchError",
    "TemporalGapError",
    "PatchOutOfBoundsError",
    "NotInitializedError",
]
