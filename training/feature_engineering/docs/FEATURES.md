# Feature Engineering — R2.3 Feature Builders

The feature-engineering package turns accepted
[`AgriculturalObservation`](../../../training/stam/observation.py) objects
into rectangular feature rows, one per sample, covering three modalities.

Package: `training/feature_engineering/`.

## Modality builders

| Builder | Prefix | Responsibility |
| --- | --- | --- |
| `TabularFeatureBuilder` | `tab.*` | Location features (lon/lat/distance/admin), resolved crop + yield labels, and generic table fields — **labels** (`crop`, `yield_value`, …) are excluded from fields via `label_columns` so they are never leaked into features |
| `ImageFeatureBuilder` | `img.*` | Sequence stats from the NDVI/EVI record pairs (counts, dates, gap lengths, coverage, per-date patch statistics when an extractor is supplied) |
| `TemporalFeatureBuilder` | `tmp.*` | Year, season, days-in-season, observation-date range (dates joined with `;` for flat rows) |

## Row assembly — `FeatureBuilderRegistry`

```python
from training.feature_engineering import (
    FeatureEngineeringConfig,
    FeatureBuilderRegistry,
    build_feature_frame,
)

row  = FeatureBuilderRegistry(config).build(observation)   # one dict
frame = build_feature_frame(corpus, config)                 # rectangular DataFrame
```

- `build_feature_frame` accepts a corpus (uses its **accepted** observations)
  or a plain list of observations.
- Missing keys across rows become `NaN`, so statistics / balancing / export
  always see a rectangular table.
- `prefixes=False` emits bare column names (`crop`, `pair_count`, …).

## Configuration

`FeatureEngineeringConfig` (pydantic) is loaded by
`load_feature_engineering_config` with **env > YAML > defaults** precedence:

| Env | Purpose |
| --- | --- |
| `FE_TABULAR__ENABLED` | toggle the tabular builder |
| `FE_IMAGE__EXTRACT_PATCH_STATS` | also extract per-date patch stats |
| `FE_TEMPORAL__INCLUDE_DATES` | include iso date columns |
| `FE_PREFIXES` | prefix every feature with its modality |

`save_feature_engineering_template(path)` writes an annotated YAML template.

## Error codes

| Code | Class | Meaning |
| --- | --- | --- |
| `FE-CONFIG-001` | `FeatureConfigError` | Invalid feature-engineering config |
| `FE-BUILD-001` | `FeatureBuilderError` | A feature builder failed |
| `FE-BUILD-002` | `MissingExtractorError` | Patch stats requested without an extractor |
| `FE-FRAME-001` | `FeatureFrameError` | Frame assembly failed (e.g. empty input) |

## Label safety

`tabular.label_columns` (default `["crop", "yield", "yield_value", "yield_kg"]`)
names raw table columns that are training labels. The tabular builder **skips**
them when emitting generic `fields` — only the STAM-resolved `crop` /
`yield_value` labels are emitted, and those follow `include_labels`.

See `training/feature_engineering/tests/` (38 tests) for the contract.
