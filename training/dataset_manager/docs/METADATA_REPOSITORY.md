# Extended Metadata Repository

R2.2 extends the **same** `metadata.db` used by the R1.2 metadata store with
four auditable tables. The repository reuses the shared SQLite connection
discipline from `_db.py` (one short-lived connection per call, WAL journaling).

## Tables

| Table | Contents | Key |
| --- | --- | --- |
| `provider_metadata` | Registered providers: kind, status, availability, priority, capabilities, manifest | `name` |
| `spatial_metadata` | Named locations (villages / districts) + coordinates + parent district | `(name, kind)` |
| `temporal_metadata` | Index x year x resolution availability counts + observation dates | `(index_type, year, resolution)` |
| `patch_metadata` | Every patch extraction request / result (path, center, size, band, CRS, padded) | `patch_id` |

All tables are created idempotently (`CREATE TABLE IF NOT EXISTS`) next to the
existing `metadata_records` table.

## Python API

```python
repo = manager.metadata_repository     # MetadataRepository(metadata.db)

# Providers
repo.save_provider(registration)
repo.list_providers()                  # ordered by priority desc, name
repo.provider_count()

# Spatial
repo.save_spatial(record)
repo.save_spatial_many(records)        # bulk upsert
repo.list_spatial(kind="village")
repo.spatial_count()

# Temporal
repo.save_temporal(TemporalRecord(...))
repo.list_temporal(index_type="NDVI", year=2019, resolution="R10m")

# Patches
repo.save_patch(patch_metadata)        # returns patch_id
repo.list_patches(limit=100)
repo.patch_count()
```

## Writers

* Providers are recorded by the manager when the registry is wired.
* Spatial records come from the auto-built spatial index (see
  [SPATIAL.md](SPATIAL.md)).
* Temporal records come from the historical context builder (see
  [EXTRACTION.md](EXTRACTION.md)).
* Patch records come from every `PatchExtractor` extraction (see
  [EXTRACTION.md](EXTRACTION.md)).

## Reports

The persisted tables feed the `provider`, `spatial` and `temporal` reports —
see [REPORTS.md](REPORTS.md).
