# GIS data + architecture (`application/gis/`)

This directory holds the boundary data used by the Prediction Platform's
location-resolution chain, plus the R1.4 architecture ports.

## Data

- `District/` — district boundary shapefile (`.shp`/`.dbf`/`.prj`/`.shx`)
- `Taluk/` — taluk boundary shapefile
- `Dakshina_Kannada/` — regional boundary set
- `kml/` — KML/KMZ exports of the same boundaries (visualisation)

CRS is the native shapefile CRS; spatial queries use EPSG:32643 (UTM 43N) via
`shared.constants.CRS_UTM_43N` (see `shared/constants`).

## Architecture (R1.4 ports only)

```
GeoPoint
  -> reverse_geocoding.ReverseGeocoder       location_index.parquet
  -> spatial_resolver.SpatialResolver        District/Taluk/Dakshina_Kannada
  -> historical_context.HistoricalContextResolver  historical_context.parquet
  -> GeoContext
```

Each step is an abstract port; implementations are deferred to a later phase.
No geometry library is imported eagerly. The facade in `resolver.py`
(`LocationResolver`) documents the composition order.
