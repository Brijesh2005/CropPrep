"""GIS layer of the Prediction Platform (architecture contract, R1.4).

Resolves a raw point (lon, lat) into everything the inference engine needs:

    GeoPoint
      -> ReverseGeocoder        (reverse_geocoding)   place / nearest known point
      -> SpatialResolver        (spatial_resolver)    village / taluk / district
      -> HistoricalContextResolver (historical_context) season + climatology
      -> GeoContext             (consumed by the prediction service)

The raw boundary data (District / Taluk / Dakshina_Kannada shapefiles, KML)
already ships in this directory. R1.4 adds the ports only — no spatial
algorithm is implemented and no geometry library is imported eagerly.
"""

from __future__ import annotations

from .models import AdminContext, GeoContext, GeoPoint, HistoricalContext, ResolvedPlace

__version__ = "0.1.0"

__all__ = [
    "AdminContext",
    "GeoContext",
    "GeoPoint",
    "HistoricalContext",
    "ResolvedPlace",
    "__version__",
]
