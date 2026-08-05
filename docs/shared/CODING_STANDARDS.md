# Shared Framework — Coding Standards

Conventions that keep `shared/` platform-agnostic, testable and easy to reason
about. They apply to every file under `shared/`.

## Dependency rules

```text
shared → stdlib + third-party only
shared → never training
shared → never application
```

- Importing the platforms from `shared/` is a build break, not just a style
  issue. New PRs that add such an import must be rejected.
- Third-party imports that are heavy or optional (torch, pandas, numpy for
  serialization) go **inside** the function that needs them, so importing
  `shared` stays cheap and works on a minimal environment.

## Public API surface

- Each subpackage's `__init__.py` is the public facade and re-exports the
  names consumers should use (mirror `shared/config/__init__.py`).
- Private helpers are prefixed with `_` and are not part of the API.
- Keep backward-compatible aliases for names that existed during the R1.3
  transition (e.g. `_parse_env`) only where a consumer still references them;
  prefer updating the consumer.

## Naming

- Functions and methods: `snake_case`.
- Classes: `PascalCase`; ports end in `Provider`, `Repository`, `Cache`,
  `Storage`, `Exporter` or end with a role noun (e.g. `SemanticVersion`).
- Enum members: `UPPER_SNAKE` (`IndexType.NDVI`); enum values are lowercase
  strings (`"ndvi"` for `FileCategory`) except where existing vocabulary is
  naturally uppercase (`"NDVI"`, `"R10m"`).
- Module names: lowercase, single word (`loader.py`, `formatters.py`).
- Constants: `UPPER_SNAKE` (`FAILING_SEVERITY`, `CRS_UTM_43N`).

## Typing and Python version

- Target Python 3.12. Use `from __future__ import annotations` in every module.
- Annotate public signatures (`def load(path: str | Path) -> Any`). Use `Any`
  for the generic payloads that serializers/validators transport.
- Prefer `dataclasses` for the lightweight metadata/value objects in
  `shared/schemas` and `shared/versioning`; use pydantic only for
  configuration models inside the platforms.

## Exceptions

- Every shared error subclasses `shared.exceptions.CropFusionError`.
- Stable `code` constants (e.g. `CF-VERSION-002`), a human `message`,
  optional `detail`, and `suggested_resolution` for recoverable errors.
- Do not re-raise raw third-party exceptions from `shared`; wrap them in the
  nearest shared domain error with the useful detail attached.

## Logging

- Consumers use `shared.logging.get_logger(name)` (loggers live under
  `cropfusion.<namespace>`).
- Structured fields go through `log_dict` / the JSON formatter's `extra=`
  support; keep reserved LogRecord keys (`name`, `msg`, `levelname`, ...) out
  of `extra=`.

## Testing

- Every public behaviour has a test in `shared/tests/test_<area>.py`.
- Tests must be platform-agnostic: no imports from `training` or `application`,
  no dependence on local datasets, no network access.
- Use `tmp_path` fixtures for file-based tests; write real small files
  (CSVs, TIFF magic bytes) rather than mocking the module under test.
- Run with:

```powershell
$env:PYTHONPATH = "D:\CropPrep"
& "D:\CropPrep\.venv\Scripts\python.exe" -m pytest "D:\CropPrep\shared\tests" -q
```

## Documentation

- Add a docstring to every module and public function describing what it does
  and (for public entry points) the arguments and raised errors.
- Update `docs/shared/SHARED.md` when the package surface changes, and
  `docs/diagrams/r1-3-shared-packages.md` when subpackages are added/removed.

## Verification gates

Before finishing a change under `shared/`:

1. `pytest shared/tests` → 101 passed.
2. `pytest training/...` all suites → 599 passed (700 with shared).
3. `pytest application/backend/app/tests` → 80 passed.
4. No `from training`/`from application` imports anywhere in `shared/`.
