# Shared Framework — Extension Guide

How to extend the pluggable parts of `shared/` without changing the platforms'
internals: serializers, validators, versioning and providers.

## Serialization

`shared.serialization` dispatches on file extension via a
`SerializerRegistry`. Built-ins (name → extensions):

| Serializer | Name | Extensions | Requires |
| --- | --- | --- | --- |
| JSON | `json` | `.json` | stdlib |
| YAML | `yaml` | `.yaml`, `.yml` | PyYAML |
| pickle | `pickle` | `.pkl`, `.pickle` | stdlib |
| Parquet | `parquet` | `.parquet`, `.pq` | pandas + pyarrow |
| CSV | `csv` | `.csv` | pandas |
| NumPy | `numpy` | `.npy`, `.npz` | numpy |
| PyTorch | `torch` | `.pt`, `.pth`, `.ckpt` | torch |

### Adding a new format

```python
from shared.serialization import Serializer, default_registry

class AvroSerializer(Serializer):
    name = "avro"
    extensions = (".avro",)

    def dump(self, data, path): ...   # return Path
    def load(self, path): ...

default_registry.register(AvroSerializer())
```

After registering, `shared.serialization.dump(data, "x.avro")` and
`shared.serialization.load("x.avro")` work automatically. Add a test in
`shared/tests/test_serialization.py`.

Optional-backend formats raise `shared.exceptions.SerializationError` with a
`suggested_resolution` when their dependency is missing.

## Validation

`shared.validation` uses a `ValidatorRegistry` keyed by a stable name.
Built-ins:

| Validator | Name | Validates |
| --- | --- | --- |
| `CsvValidator` | `csv` | file exists, non-empty, header, data rows |
| `ImageValidator` | `image` | exists, non-empty, GeoTIFF magic bytes |
| `MetadataValidator` | `metadata` | mapping with required keys |
| `ConfigValidator` | `config` | mapping root |
| `VersionValidator` | `version` | semantic-version string |

### Adding a new validator

```python
from shared.validation import Validator, default_registry

class ManifestValidator(Validator):
    name = "manifest"

    def validate(self, target, **context):
        return ValidationResult(passed=True, issues=[], target="manifest")

default_registry.register(ManifestValidator())
```

Then dispatch through the public API:

```python
from shared.validation import validate

result = validate("manifest.json", "manifest")
```

### Issue semantics

`ValidationResult.passed` is `False` only when at least one issue has
`ERROR` or `CRITICAL` severity (`shared.enums.FAILING_SEVERITY`). Warnings and
infos do not fail a run. Each issue carries `code`, `severity`, `message`,
optional `path` and `detail`.

## Versioning

`shared.versioning.SemanticVersion` is strict `MAJOR.MINOR.PATCH` with
comparison operators and a `bump(part)` helper. Domain artifacts add a `kind`
tag:

```python
from shared.versioning import DatasetVersion, ModelVersion

DatasetVersion("crop-ds", "1.0.0").kind      # 'dataset'
ModelVersion("lstm-fusion", "1.0.0").kind     # 'model'
```

`VersionProvider` is the port for read/bump of artifact versions; implement
`current(name)`, `list(name)` and `bump(name, part, message=...)` for a concrete
store (filesystem, database, S3). `SemanticVersion` rejects anything that is
not strict semver (`v1.2.3`, `1.2`, pre-release/build metadata) with
`InvalidVersionError` (`CF-VERSION-002`).

## Providers and ports

`shared.interfaces` defines the ports both platforms program against:

- `Provider` → `health()`, `describe()`
- `DatasetProvider` → `fetch()`, `exists()`, `version()`
- `TabularProvider` → `discover()`, `load()`, `preview()`
- `ImageProvider` → `catalog()`, `read_metadata()`, `read_window()`, `iterate()`
- `Repository` → `save()`, `save_many()`, `get()`, `query()`, `count()`, `close()`
- `Cache` → `get()`, `set()`, `delete()`, `delete_prefix()`, `clear()`, `prune()`
- `Storage` → `exists()`, `read_bytes()`, `write_bytes()`, `delete()`, `list()`
- `ModelExporter` → `export()`, `export_report()`
- `Logger` → `get_logger()`, `setup()`
- `ConfigurationProvider` → `get()`, `load()`
- `Serializer` → `dump()`, `load()`
- `VersionProvider` → `current()`, `bump()`

Concrete implementations live in the platforms and implement these ports; the
ports never import from the platforms.

## Logging

`shared.logging.setup_logging` accepts a `profile` (`default`, `training`,
`application`, `audit`), an output format (`json`, `compact`, `colored`), a
`log_dir`, rotation settings and an optional `logger_name`. Adding a new
profile is a one-line change to `PROFILES` in `shared/logging/setup.py` plus an
entry in `PROFILE_FILE_NAMES`.

`shared.logging.audit.audit()` emits a structured audit event; `log_dict`
emits structured fields without colliding with the reserved LogRecord
attributes.

## Coding rules for extensions

- Register globals at import time through the public registry objects.
- Never import `training.*` or `application.*` from `shared/`.
- Heavy optional dependencies are imported lazily and raise a
  `SerializationError`-style error with a `suggested_resolution`.
- Every new extension ships with a test in `shared/tests/`.
