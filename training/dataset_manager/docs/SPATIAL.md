# Spatial Index & Location Resolution

The **spatial index** turns tabular location data (village names, latitudes,
longitudes, optional parent districts) into a queryable in-memory index used by
location resolution, spatial validation and the spatial report.

## How it is built

The manager auto-builds the index on startup (`_auto_build_spatial_index`):

1. Discover tabular datasets through the tabular provider.
2. For each dataset, find the location-name column and latitude / longitude
   columns by name heuristics.
3. Read up to 50 000 rows and index every usable row as a `SpatialRecord`.
4. Records without usable coordinates are skipped; datasets without the
   required columns are skipped.
5. The result is persisted to `spatial_metadata` in `metadata.db`.

Rows are also exposed programmatically:

```python
from training.dataset_manager.models import SpatialRecord
from training.dataset_manager.spatial_index import build_records_from_frame

records = build_records_from_frame(
    frame,
    name_col="village", lat_col="latitude", lon_col="longitude",
    kind="village", district_col="district",
)
index.build(records)
```

## Lookups

| Method | Matches |
| --- | --- |
| `lookup_village(name)` | Records whose name matches (case-insensitive, prefix-tolerant) |
| `lookup_district(name)` | District-kind records whose name matches |
| `nearest(lat, lon, k)` | `k` nearest records by coordinate distance (KD-tree when scipy is present, linear fallback) |
| `search_coordinates(lat, lon, tol)` | Records within `tolerance` degrees |
| `within_bbox(min_lon, min_lat, max_lon, max_lat)` | Records inside the box |
| `within_radius(lat, lon, radius_km)` | Records within a haversine radius |

`metadata()` returns aggregate counts (`count`, `villages`, `districts`,
`bounds`).

## Location resolution via the manager

`DatasetManager.get_location` resolves a named location or a coordinate point:

```python
manager.get_location(name="Moodabidri", kind="village")
manager.get_location(latitude=13.08, longitude=74.89, k=3)
manager.get_location(latitude=13.08, longitude=74.89, radius_km=10)
manager.get_location(latitude=13.08, longitude=74.89, tolerance=0.01)
```

`HistoricalContextBuilder` and patch extraction use the index to resolve
village / district names to coordinates before gathering satellite records.

## CLI

```bash
python -m training.dataset_manager spatial          # index metadata + locations
python -m training.dataset_manager location --name Moodabidri --kind village
python -m training.dataset_manager location --lat 13.08 --lon 74.89 --k 3
```

## Persistence

Each indexed location is upserted into the `spatial_metadata` table (primary
key `(name, kind)`) — see [METADATA_REPOSITORY.md](METADATA_REPOSITORY.md).
