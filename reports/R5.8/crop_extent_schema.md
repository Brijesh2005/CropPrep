# R5.8 Phase 1 — crop_extent schema

**R5.8 STATUS = BLOCKED_BY_CROP_EXTENT_SCHEMA**

## What is `Crop_Extent`?

- `Crop_Extent` is a **scalar land-area measure** in a compound `A-B-C` notation (e.g. `1-76-0.00` = acres + fractional subdivision), recorded once per crop survey row.
- **100.0%** of 837,069 rows match the `X-Y-Z` area pattern.
- The three parts have ranges A∈[0,4266.0], B∈[0,99.0], C∈[0,666.0].
- It is **crop-specific** (same Survey_id yields different extents for different crop rows in 73528 surveys).

## Does it contain spatial geometry?  NO.

- No polygon, bounding box, centroid, or coordinate placement.
- Rows with coordinate-like extent content: 0.
- Geometry-hint values found in extent strings: 0.
- No crop geometry source exists in the repository. The only GIS files are administrative boundaries:

```
- 24_Dakshina_Kannada.dbf
- 24_Dakshina_Kannada.prj
- 24_Dakshina_Kannada.shp
- 24_Dakshina_Kannada.shx
- District.cpg
- District.dbf
- District.kmz
- District.prj
- District.sbn
- District.sbx
- District.shp
- District.shx
- README.md
- Taluk.cpg
- Taluk.dbf
- Taluk.kmz
- Taluk.prj
- Taluk.sbn
- Taluk.sbx
- Taluk.shp
- Taluk.shx
- __init__.py
- __init__.py
- __init__.py
- __init__.py
- models.py
- resolver.py
```

## Why R5.8 is blocked

The sub-field hypothesis requires a **physical sub-field region** to map crop-specific satellite pixels to. `crop_extent` is a scalar area with no spatial placement, and no polygon/bbox geometry exists in the available data. Producing 'sub-field' pixels would require inventing geometry (forbidden by R5.8 rules 4/6/7), so the experiment cannot be honestly run on this dataset.

## Unblock contract

R5.8's sub-field phases will run the moment a real per-crop geometry source is provided (e.g. cadastral parcel polygons with crop attribution, or a village/field polygon layer). The interface is defined in `reports/R5.8/crop_extent_schema.json` (`unblock_contract`).
