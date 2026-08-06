# Patch Extraction & Historical Context

Two R2.2 capabilities turn raw data into model-ready inputs: **patch
extraction** (a geographic window of imagery) and the **historical context
builder** (everything the models know about a location across all available
years).

## Patch extraction

`PatchExtractor` turns a farmer location + patch size into a raw NumPy array
window. It composes the image provider — no raster is ever fully loaded.

```python
manager.get_patch(
    latitude=13.08, longitude=74.89, size=8,
    index_type="NDVI", resolution="R10m", year=2019,
)                    # -> np.ndarray (8, 8)

array, meta = manager.patch_extractor.extract_with_metadata(
    latitude=13.08, longitude=74.89, size=8,
    index_type="NDVI", padding=True,
)
meta.path        # chosen raster
meta.center      # point in the raster CRS
meta.crs         # raster CRS
meta.padded      # True when edge-padded to size
```

Extraction steps:

1. **Locate** — pick the raster best matching index / resolution / year
   (nearest raster center to the requested point, bounded candidate scan).
2. **Convert** — transform the WGS84 point into the raster CRS (pyproj;
   identity for EPSG:4326).
3. **Extract** — windowed read through `ImageProvider.patch`.
4. **Pad** — edge-pad undersized patches to exactly `size` (optional).
5. **Record** — persist `PatchMetadata` in `patch_metadata` (when a
   metadata repository is wired).

`padding=False` keeps the raw window shape; `padding=True` (default) pads edge
patches to the exact requested size. Points entirely outside every raster raise
`DatasetNotFoundError`.

## Historical context builder

`HistoricalContextBuilder.build` answers *"everything we know about this
location across every available year"* — the tabular record plus NDVI / EVI
satellite records, observation dates and resolution bands. **STAM is not
executed here**; this layer gathers raw context for later inference.

```python
ctx = manager.historical_context_builder.build(
    district="Dakshina Kannada",
    index_type="NDVI",
    resolution="R10m",
    # years=[2018, 2019, 2020],
)
ctx.location              # resolved location label
ctx.latitude / ctx.longitude
ctx.years                 # years actually observed
ctx.missing_years         # expected but not observed
for obs in ctx.observations:
    obs.year
    obs.ndvi               # [record, ...]
    obs.evi
    obs.observation_dates
    obs.quality            # records / ndvi / evi / has_tabular / resolutions
ctx.quality                # overall availability summary
```

Per-year temporal availability is persisted to `temporal_metadata` when a
metadata repository is wired.

## CLI

```bash
python -m training.dataset_manager extract-patch \
    --lat 13.08 --lon 74.89 --size 8 --index NDVI --year 2019
python -m training.dataset_manager historical-context \
    --district "Dakshina Kannada" --index NDVI --resolution R10m
```
