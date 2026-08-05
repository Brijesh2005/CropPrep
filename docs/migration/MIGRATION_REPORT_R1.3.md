# Migration Report — R1.2 → R1.3 (Shared Framework)

Detailed record of the R1.3 refactor: extracting the reusable, duplicated
configuration / exceptions / logging / enums / versioning / validation /
serialization utilities into a platform-agnostic `shared/` package so the
Training Platform (`training/`) and Prediction Platform (`application/`)
depend on `shared` — never on each other's internals. Companion to
[MIGRATION_REPORT_R1.2](MIGRATION_REPORT_R1.2.md) and
[MIGRATION_REPORT](MIGRATION_REPORT.md).

- **Status**: complete
- **Date**: 2026-08-05
- **Branch**: `main` (not yet committed)

## 1. Scope and principles

- **Shared utilities only**: R1.3 moved *infrastructure* (config loading,
  exceptions, logging, enums, versioning, validation, serialization,
  interfaces, schemas, pure utils). Training algorithms, models, STAM
  algorithms, FastAPI APIs, React, Docker, database and prediction logic were
  **not** touched.
- **Dependency rules** (enforced by layout, verified by grep + tests):
  `training → shared`, `application → shared`, `shared → stdlib + third-party
  only`. Neither platform imports the other's private helpers.
- **Backward compatible**: platform public APIs and test suites are unchanged;
  every suite stays green.
- **No deletions of platform behaviour**: local helper *definitions* were
  removed, but only after the shared replacements were verified identical.

## 2. New components — `shared/`

`shared/` grew from a 100%-placeholder package into the contract layer
(version `0.1.0`):

| Package | Modules | Responsibility |
| --- | --- | --- |
| `shared/config/` | `loader.py` | `deep_merge`, `parse_env`, `apply_case_insensitive`, `load_yaml_config`, `normalise_key` (+ legacy aliases `_parse_env`, `_apply_case_insensitive`, `_normalise_key`). |
| `shared/constants/` | `__init__.py` | directories, extensions, `CRS_UTM_43N="EPSG:32643"`, GeoTIFF/CSV formats, env prefixes (`DM_` … `BACKEND_`, `CF_`), provider names, metadata keys, chunk size. |
| `shared/enums/` | `__init__.py` | `IndexType`, `Resolution`, `FileCategory`, `Severity`, `DatasetStatus`, `ValidationStatus`, `CropType`, `Season`, `ModelStatus`, `TrainingStage`, `ReleaseStatus`, `ProviderType`, `EnvironmentType`, `FAILING_SEVERITY`. |
| `shared/exceptions/` | `base.py` + 9 domain modules | `CropFusionError` + 30+ errors with stable `CF-*` codes, `detail` and `suggested_resolution`. |
| `shared/interfaces/` | `providers.py`, `data.py`, `platform.py` | Ports: `Provider`/`DatasetProvider`/`TabularProvider`/`ImageProvider`, `Repository`/`Cache`/`Storage`, `ModelExporter`/`Logger`/`ConfigurationProvider`/`Serializer`/`VersionProvider`. |
| `shared/logging/` | `formatters.py`, `setup.py`, `audit.py` | `setup_logging` (profiles default/training/application/audit, JSON/compact/colored), `get_logger`, `log_dict`, `audit()`, idempotent re-configuration. |
| `shared/schemas/` | `dataset.py`, `image.py`, `prediction.py`, `validation.py`, `meta.py` | Metadata dataclasses (`DatasetInventorySchema`, `RasterMetadataSchema`, `PredictionResultSchema`, `ValidationReportSchema`, `TrainingRunSchema`, `ReleaseMetadataSchema`, ...). |
| `shared/serialization/` | `registry.py`, `formats.py` | Pluggable registry + 7 built-ins: JSON, YAML, pickle, parquet, CSV, NumPy (`.npy`/`.npz`), torch. |
| `shared/types/` | placeholder | Shared type aliases. |
| `shared/utils/` | hash/io/path/env/time/naming/parallel/yaml/json_util | `sha256_file`, `count_lines_fast`, `run_parallel`, `is_geotiff_bytes`/`is_geotiff_path`, `classify_index_type`/`classify_resolution`, `extract_year_from_path`, `parse_observation_date`, `yaml_safe`, `write_json`/`read_json`, `human_size`, `walk_files`, `env_map_of`, ... |
| `shared/validation/` | `base.py`, `validators.py`, `registry.py` | `Validator` port, `ValidationResult`/`ValidationIssue`, registry with 5 built-ins (`config`, `csv`, `image`, `metadata`, `version`). |
| `shared/versioning/` | `semver.py`, `versions.py`, `provider.py` | `SemanticVersion`, `VersionInfo` + `DatasetVersion`/`ModelVersion`/`InferenceVersion`/`ApplicationVersion`, `VersionProvider` port. |
| `shared/tests/` | 9 test modules | 101 platform-agnostic tests. |

## 3. Extracted and moved

| Item | From (old) | To (new) |
| --- | --- | --- |
| `deep_merge` | `training/dataset_manager/config.py`, `training/training/{ablation,experiment}.py` | `shared.config.deep_merge` |
| `parse_env` | `dataset_manager/config.py`, `training/config.py`, `stam/config.py`, `preprocessing/config.py`, `explainability/config.py`, `application/backend/app/core/config.py` | `shared.config.parse_env` |
| `apply_case_insensitive` | same six files | `shared.config.apply_case_insensitive` |
| `normalise_key` | same six files | `shared.config.normalise_key` |
| `yaml_safe` (4 `_yaml_safe` copies) | `training/config.py`, `explainability/config.py`, `training/logger.py`, `application/backend/app/core/config.py` | `shared.utils.yaml_safe` (superset handling `.item()` for torch/numpy) |
| JSON/compact log formatters | `training/dataset_manager/logger.py`, `training/stam/logger.py`, `training/preprocessing/logger.py` | `shared.logging.formatters.{JsonFormatter, CompactFormatter, ColoredFormatter, RESERVED}` |
| Enums | `training/dataset_manager/models.py` | `shared.enums` (re-exported by the old module) |
| Exception bases | 6 platform `exceptions.py` | `shared.exceptions.CropFusionError` (+ domain bases) |

## 4. Removed duplicates

- `_yaml_safe` deleted from `training/training/logger.py`,
  `training/explainability/config.py` and `application/backend/app/core/config.py`.
- `_parse_env` / `_apply_case_insensitive` / `_normalise_key` / local
  `deep_merge` deleted from `training/dataset_manager/config.py` (the last
  remaining copies; now importing `shared.config`).
- Local `JsonFormatter`/`CompactFormatter` class definitions deleted from
  `training/dataset_manager/logger.py`.
- `training/stam/logger.py` and `training/preprocessing/logger.py` dropped
  their dependency on `training.dataset_manager` (now import `shared.logging`).
- `training/dataset_manager/models.py` no longer *defines* the enums; they are
  imported from `shared.enums` and re-exported for compatibility.

## 5. Exceptions hierarchy

| Platform base | Subclasses | Shared parent |
| --- | --- | --- |
| `DatasetManagerError` | `DM-*` | `CropFusionError` |
| `TrainingError` | `TD-*` | `CropFusionError` |
| `StamError` | `ST-*` | `CropFusionError` |
| `PreprocessingError` | `PPT-*` | `CropFusionError` |
| `ModelError` | `MOD-*` | `CropFusionError` |
| `ExplainabilityError` | `EXP-*` | `CropFusionError` |

Each platform keeps its own code prefix; `shared.exceptions` contributes the
base contract (`code`/`message`/`detail`/`suggested_resolution`) plus shared
domain errors (`ConfigurationError`, `NotFoundError`, `IntegrityError`,
`SerializationError`, `ValidationFailedError`, `InvalidVersionError`,
`PredictionError`, `AuthenticationError`, ...).

## 6. Interfaces

`shared.interfaces` formalises the contracts the platforms already used
informally:

- **Data access**: `Repository` (`save`/`save_many`/`get`/`query`/`count`/`close`),
  `Cache` (`get`/`set`/`delete`/`delete_prefix`/`clear`/`prune`),
  `Storage` (`exists`/`read_bytes`/`write_bytes`/`delete`/`list`).
- **Providers**: `Provider`, `DatasetProvider`, `TabularProvider`,
  `ImageProvider` (health/describe/discover/load/preview/catalog/read_window/...).
- **Platform**: `ModelExporter`, `Logger`, `ConfigurationProvider`, `Serializer`,
  `VersionProvider`.

Concrete implementations remain in the platforms; the ports live in `shared`.

## 7. Configuration

Six consumers were re-pointed onto `shared.config`:

- `training/dataset_manager/config.py` — `parse_env(env, ENV_PREFIX)` +
  `deep_merge` + `apply_case_insensitive`; local copies deleted.
- `training/training/config.py` — `parse_env`/`apply_case_insensitive` +
  `shared.utils.yaml_safe`; the R1.2 regression (`to_yaml` still calling the
  deleted `_yaml_safe`) was fixed to call `yaml_safe`.
- `training/stam/config.py`, `training/preprocessing/config.py` — `parse_env`/
  `apply_case_insensitive`.
- `training/explainability/config.py` — `parse_env`/`apply_case_insensitive` +
  `yaml_safe`.
- `application/backend/app/core/config.py` — imports `apply_case_insensitive`,
  `deep_merge`, `parse_env` from `shared.config` and `yaml_safe` from
  `shared.utils`; private `_yaml_safe` deleted; `save_settings_template`
  re-verified.

## 8. Imports

- Cross-platform utility imports removed (grep-verified):
  `application/backend` no longer imports private `training` helpers.
- `training/*` never imports `application/*` (grep-verified).
- `shared/*` never imports `training/*` or `application/*` (grep-verified).

## 9. Dependency improvements

- **One source of truth**: config/env/yaml and logging fixes apply once.
- **Direction is explicit**: the backend consumes `shared.config`/`shared.utils`
  instead of reaching into `training` internals.
- **Cheap import cost**: heavy optional deps (numpy/torch/pandas) are imported
  lazily inside serializers.
- **Shared vocabulary**: `CropType`, `Season`, `Severity`, `DatasetStatus`,
  etc. are single definitions in `shared.enums`.

## 10. Verification

| Check | Result |
| ----- | ------ |
| `pytest shared/tests` | **101 passed** in ~1s |
| `pytest training/...` (all 7 suites) | **599 passed** (700 total with shared) |
| `pytest application/backend/app/tests` | **80 passed** |
| `DM_SCAN__WORKERS=16` → `scan.workers == 16` | env override works post-refactor |
| Backend `Settings` import (`app_name == "CropFusion Backend"`) | OK |
| Grep: `from/import training` in `application` (utility layer) | only execution delegation remains (see §12) |
| Grep: `from/import application` in `training` | none |
| Grep: `from/import training`/`application` in `shared` | none |
| Grep: leftover `def _yaml_safe`/`def _parse_env`/`def _apply_case_insensitive` | none |

Also fixed along the way:

- `shared.serialization.NumpySerializer`: `dump` now writes real `.npy` files
  for `.npy` paths (previously always produced `.npz`), and `load` handles both
  `.npy` (returns `ndarray`) and `.npz` archives.

## 11. Future extensions

- Migrate the remaining `application → training` execution delegation behind
  the `shared.interfaces` ports (`ModelExporter`, `PredictionService`) so the
  backend can swap models without touching algorithm code.
- Move the duplicated dataset/STAM/preprocessing YAML templates under
  `training/config/` onto `load_yaml_config` for consistent validation.
- Fill the `shared/dto/` and `shared/types/` placeholders with the canonical
  request/response DTOs shared by both platforms.
- Add a schema-based `SchemaValidator` integration into the platform scanners.

## 12. Known limitations

- **Deliberate execution coupling**: `application/backend` still calls Training
  Platform modules at runtime (`training.models.ModelFactory`,
  `training.stam.STAM`, `training.preprocessing`, `training.explainability`,
  `training.dataset_manager`) to perform real predictions. This is composition
  of algorithms, not utility reuse; relocating it would move training
  algorithms into `shared`, which R1.3 explicitly did not do.
- `application/backend/app/core/paths.py` still injects `training` into
  `sys.path` to support that delegation (unchanged).
- Pydantic `protected_namespaces` warnings for `model_version` /
  `model_ready` fields in backend schemas are pre-existing and unrelated to
  this refactor.
- `shared/types` and `shared/dto` remain placeholders (future work).

## 13. Rollback / safety

All changes are additive to platform behaviour. Removing the `shared` imports
and restoring the local helper copies in the six config files + three loggers +
`dataset_manager/models.py` reverts to the R1.2 state; every public API is
unchanged, so the full test suites are the safety net.
