# Shared Framework — Configuration

Every CropFusion platform resolves its settings with the same precedence and the
same helpers. `shared/config/loader.py` is the single source of truth for the
pattern.

## Precedence (highest wins)

```text
1. Environment variables   <PREFIX><SECTION>__<KEY>
2. YAML configuration file
3. Built-in defaults       (each platform's pydantic settings)
```

## Environment variable format

- Prefix per platform: `DM_` (dataset manager), `ST_` (STAM), `TD_` (training),
  `PPT_` (preprocessing), `MOD_` (models), `EXP_` (explainability), `ML_`
  (mlops), `BACKEND_` (application), `CF_` (generic).
- Nested settings use `__` as the separator.
- Values are auto-parsed: `true`/`false` → bool, JSON (`[...]`, `{...}`, numbers)
  → parsed, anything else stays a string.

```ini
DM_DATASET_ROOT=/data/cropfusion/datasets
DM_DOWNLOAD__FORCE_DOWNLOAD=true
DM_SCAN__WORKERS=16
DM_LOG__LEVEL=DEBUG
DM_VALIDATE__EXPECTED_YEARS="[2018, 2025]"
```

## API

### `shared.config.parse_env(env, prefix)`

Convert `<PREFIX><SECTION>__<FIELD>` env vars into a nested dict.

```python
from shared.config import parse_env

parse_env({"DM_SCAN__WORKERS": "16", "DM_DATASET_ROOT": "/data"}, prefix="DM_")
# {'scan': {'workers': 16}, 'dataset_root': '/data'}
```

### `shared.config.deep_merge(base, override)`

Recursively merge `override` into a **copy** of `base` (nested dicts merge,
scalars replace). `base` is never mutated.

### `shared.config.apply_case_insensitive(data, schema)`

Match config keys to pydantic field names case-insensitively (`Dataset_Root` →
`dataset_root`) using the pydantic `model_fields` of the target settings class.

### `shared.config.normalise_key(key)`

Lower-case a field path segment and replace `-` with `_`.

### `shared.config.load_yaml_config(path, env=..., prefix=...)`

Load a YAML file, merge environment overrides on top, and return the resulting
mapping ready for pydantic validation. Raises `shared.exceptions.ConfigurationError`
when the file is missing, malformed, or the root is not a mapping.

## Reference implementations

| Consumer | Loader |
| --- | --- |
| `training/dataset_manager/config.py` | `load_settings()` → `shared.config.{parse_env, deep_merge, apply_case_insensitive}` |
| `training/training/config.py` | `load_training_config()` |
| `training/stam/config.py` | `load_stam_config()` |
| `training/preprocessing/config.py` | `load_preprocessing_config()` |
| `training/explainability/config.py` | `load_explainability_config()` |
| `application/backend/app/core/config.py` | `Settings` bootstrap |

All six consumers use the same shared primitives; the per-platform files only
add their own pydantic settings models and validation.

## Backward-compatible aliases

`shared.config` still exports the legacy private names
`_parse_env`, `_apply_case_insensitive`, `_normalise_key` for any code that
referenced them during the transition. New code should use the public names.

## YAML safety

`shared.utils.yaml_safe(data)` converts objects that `yaml.safe_dump` cannot
serialize (e.g. `pathlib.Path`, torch tensors, numpy scalars — including a
`.item()` path for tensors/scalars) into plain Python values. It replaces the
four private `_yaml_safe` copies that previously lived in the platforms.

## Related

- [Configuration validation](../shared/EXTENSION_GUIDE.md#validation)
- [R1.3 configuration diagram](../diagrams/r1-3-config-resolution.md)
