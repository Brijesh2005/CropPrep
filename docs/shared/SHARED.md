# Shared Framework — Overview

The `shared/` package is the **platform-agnostic contract layer** used by both
the Training Platform (`training/`) and the Prediction Platform
(`application/`). It holds the reusable primitives that both platforms need
(configuration loading, exceptions, logging, validation, serialization,
versioning, enums, interfaces and metadata schemas) so that **neither platform
imports from the other**.

- **Package**: `shared`
- **Version**: `0.1.0`
- **Status**: complete (R1.3)
- **Dependencies**: Python standard library + third-party only (pydantic,
  PyYAML, numpy, torch, pandas are optional/used lazily). It never imports
  `training` or `application`.

## Subpackages

| Package | Responsibility |
| --- | --- |
| `shared/config/` | Configuration loading primitives: `deep_merge`, `parse_env`, `apply_case_insensitive`, `load_yaml_config`, `normalise_key`. |
| `shared/constants/` | Canonical constants: directories, file extensions, CRS (`EPSG:32643`), env prefixes, provider names, metadata keys. |
| `shared/enums/` | Canonical vocabulary: `IndexType`, `Resolution`, `FileCategory`, `Severity`, `DatasetStatus`, `ValidationStatus`, `CropType`, `Season`, `ModelStatus`, `TrainingStage`, `ReleaseStatus`, `ProviderType`, `EnvironmentType`. |
| `shared/exceptions/` | `CropFusionError` base plus 30+ domain errors with stable `CF-*` codes and `suggested_resolution`. |
| `shared/interfaces/` | Ports/ABCs: `Provider`, `DatasetProvider`, `TabularProvider`, `ImageProvider`, `Repository`, `Cache`, `Storage`, `ModelExporter`, `Logger`, `ConfigurationProvider`, `Serializer`, `VersionProvider`. |
| `shared/logging/` | `setup_logging` (profiles `default` / `training` / `application` / `audit`), JSON/compact/colored formatters, `get_logger`, `audit()`. |
| `shared/schemas/` | Metadata dataclasses (`DatasetInventorySchema`, `RasterMetadataSchema`, `PredictionResultSchema`, `ValidationReportSchema`, ...). |
| `shared/serialization/` | Pluggable serializer registry with 7 built-ins: JSON, YAML, pickle, parquet, CSV, NumPy, torch. |
| `shared/types/` | Shared type aliases (e.g. path types). |
| `shared/utils/` | Pure helpers: `sha256_file`, `count_lines_fast`, `run_parallel`, `is_geotiff_bytes`/`is_geotiff_path`, `classify_index_type`, `classify_resolution`, `extract_year_from_path`, `parse_observation_date`, `yaml_safe`, `write_json`/`read_json`, `human_size`, ... |
| `shared/validation/` | `Validator` port, `ValidationResult`/`ValidationIssue`, registry with 5 built-ins (`config`, `csv`, `image`, `metadata`, `version`). |
| `shared/versioning/` | `SemanticVersion` (MAJOR.MINOR.PATCH), `VersionInfo` + `DatasetVersion`/`ModelVersion`/`InferenceVersion`/`ApplicationVersion`, `VersionProvider` port. |
| `shared/dto/` | Data-transfer-object placeholder (future shared DTOs). |

## What was extracted (duplicates removed)

| Helper | Old homes (duplicated) | New home |
| --- | --- | --- |
| `deep_merge` | `dataset_manager.config`, `training.config`, `ablation`, `experiment` | `shared.config.deep_merge` |
| `parse_env` (`_parse_env`) | `dataset_manager.config`, `training.config`, `stam.config`, `preprocessing.config`, `explainability.config`, backend `core/config.py` | `shared.config.parse_env` |
| `apply_case_insensitive` (`_apply_case_insensitive`) | same six files | `shared.config.apply_case_insensitive` |
| `normalise_key` (`_normalise_key`) | config modules | `shared.config.normalise_key` |
| `yaml_safe` (`_yaml_safe`, 4 copies) | `training.config`, `explainability.config`, `backend core/config.py`, `training/logger.py` | `shared.utils.yaml_safe` |
| JSON/compact log formatters | `dataset_manager.logger`, `stam.logger`, `preprocessing.logger` | `shared.logging.formatters` |
| Enums (`IndexType`, `Resolution`, `FileCategory`, `Severity`, `DatasetStatus`, `FAILING_SEVERITY`) | `training/dataset_manager/models.py` | `shared.enums` |
| Exception bases | 6 platform `exceptions.py` modules | `shared.exceptions.CropFusionError` |

## Dependency rules (enforced by layout, verified by tests)

```text
training   → shared   (never application)
application → shared  (never training at the utility layer)
shared     → stdlib + third-party only (never training, never application)
```

## Quick start

```python
from shared.config import deep_merge, parse_env, load_yaml_config
from shared.enums import IndexType, Season
from shared.exceptions import CropFusionError
from shared.serialization import dump, load
from shared.validation import validate
from shared.versioning import SemanticVersion

merged = deep_merge({"scan": {"workers": 4}}, {"scan": {"workers": 8}})
parsed = parse_env({"DM_SCAN__WORKERS": "8"}, prefix="DM_")
dump({"x": 1}, "artifact.json")
result = validate("some.csv", "csv")
version = SemanticVersion.from_string("1.2.3").bump("minor")
```

## Tests

```powershell
$env:PYTHONPATH = "D:\CropPrep"
& "D:\CropPrep\.venv\Scripts\python.exe" -m pytest "D:\CropPrep\shared\tests" -q
```

101 tests cover the config loader, utils, serialization, validation, logging,
versioning, exceptions, schemas and interfaces.
