# Shared Framework — Developer Guide

How to work with `shared/` from either platform, and how to keep the dependency
rules intact.

## Setting up a working interpreter

```powershell
$env:PYTHONPATH = "D:\CropPrep"                 # both platforms see `shared`
& "D:\CropPrep\.venv\Scripts\python.exe" -m pytest "D:\CropPrep\shared\tests" -q
```

For the Prediction Platform add the backend root:

```powershell
$env:PYTHONPATH = "D:\CropPrep;D:\CropPrep\application\backend"
```

## Consuming `shared` from `training/`

Import the public entry points only. Do **not** import private submodules'
internals unless it is a deliberate port.

```python
# good
from shared.config import deep_merge, parse_env
from shared.enums import IndexType, Season
from shared.exceptions import CropFusionError

# good — public facade
from shared.utils import yaml_safe, sha256_file

# bad — reach into internals
from shared.config.loader import _parse_env  # noqa
```

## Consuming `shared` from `application/`

The backend should rely on `shared` for the same utilities it used to copy.
`application/backend/app/core/config.py` is the reference example: it imports
`apply_case_insensitive`, `deep_merge`, `parse_env` from `shared.config` and
`yaml_safe` from `shared.utils`, and deleted its private `_yaml_safe` copy.

## Exception handling

Every platform error inherits from `shared.exceptions.CropFusionError`, so a
single `except CropFusionError` in shared infrastructure catches every domain
error from both platforms. Each error carries:

- `code` — stable machine-readable identifier (e.g. `CF-VERSION-002`).
- `message` — human readable summary (appears in `str(error)`).
- `detail` — optional structured payload.
- `suggested_resolution` — optional actionable hint.

Platforms keep their own prefixes (`DM-`, `TD-`, `ST-`, `PPT-`, `MOD-`, `EXP-`,
`ML-`, `API-`) by subclassing the shared domain bases.

## Adding a new shared module

1. Create the subpackage under `shared/<name>/` with an `__init__.py` that
   re-exports the public API (same style as `shared/config/__init__.py`).
2. Keep it **stdlib + third-party only**; if a dependency is heavy or optional
   (torch, pandas), import it lazily inside the function that uses it.
3. Add tests under `shared/tests/test_<name>.py`.
4. Update `docs/shared/SHARED.md` and the package diagram
   `docs/diagrams/r1-3-shared-packages.md`.

## Logging

Use `shared.logging.get_logger(name)` so loggers live under
`cropfusion.<namespace>` and keep working with the existing platform loggers
(`cropfusion.dataset_manager`, `cropfusion.spatial_alignment`, ...).

```python
from shared.logging import get_logger

logger = get_logger("training.experiment")
logger.info("starting run %s", run_id)
```

## Running the shared test suite

```powershell
& "D:\CropPrep\.venv\Scripts\python.exe" -m pytest "D:\CropPrep\shared\tests" -q
```

Expected: **101 passed**.
