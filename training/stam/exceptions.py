"""Exception hierarchy for the Spatial-Temporal Alignment Module (STAM).

Every STAM failure raises :class:`StamError` (or a subclass) carrying a
stable machine-readable ``code`` (``ST-<AREA>-<NNN>``). Callers can catch the
base type and still discriminate between failure modes.
"""

from __future__ import annotations

from typing import Any

from shared.exceptions import CropFusionError


class StamError(CropFusionError):
    """Base class for all STAM errors."""

    code: str = "ST-ERROR"


class StamConfigurationError(StamError):
    """Raised when STAM configuration is invalid or inconsistent."""

    code = "ST-CONFIG-001"


class InvalidCoordinatesError(StamError):
    """Raised when a location has out-of-range latitude/longitude."""

    code = "ST-COORD-001"


class LocationNotFoundError(StamError):
    """Raised when no dataset location matches within the search radius."""

    code = "ST-SPATIAL-001"


class BoundaryNotFoundError(StamError):
    """Raised when no administrative boundary contains a point."""

    code = "ST-ADMIN-001"


class NoTabularRecordError(StamError):
    """Raised when no tabular agricultural record matches a query."""

    code = "ST-TABULAR-001"


class NoImageRecordError(StamError):
    """Raised when no image record matches a query (year/season/index)."""

    code = "ST-IMAGE-001"


class PairingError(StamError):
    """Raised when NDVI/EVI pairing fails for an observation date."""

    code = "ST-PAIR-001"


class CRSMismatchError(StamError):
    """Raised when two rasters (or a raster and a point) use different CRS."""

    code = "ST-CRS-001"


class ResolutionMismatchError(StamError):
    """Raised when paired rasters use different spatial resolutions."""

    code = "ST-RES-001"


class TemporalGapError(StamError):
    """Raised when a temporal sequence exceeds the allowed gap."""

    code = "ST-TEMP-001"


class PatchOutOfBoundsError(StamError):
    """Raised when a requested patch cannot be produced (raster too small)."""

    code = "ST-PATCH-001"


class NotInitializedError(StamError):
    """Raised when an operation requires :meth:`STAM.initialize` first."""

    code = "ST-INIT-001"


class SampleResolutionError(StamError):
    """Raised when a training-sample plan cannot be built or resolved."""

    code = "ST-RESOLVE-001"


class SampleCellError(StamError):
    """Raised when a single sampling cell cannot be resolved to an observation.

    Per-cell failures during bulk resolution are recorded on the
    :class:`~training.stam.observation_resolver.ResolvedSample` instead of
    raised; this error only surfaces when a cell is resolved directly.
    """

    code = "ST-RESOLVE-002"
