# Developer Guide

## Repository layout

```
services/dataset_manager/
├── __init__.py          # public API surface
├── manager.py           # facade (the only public entry point)
├── interfaces.py        # abstract ports (ABCs)
├── models.py            # shared dataclasses + enums
├── exceptions.py        # typed errors with stable codes
├── config.py            # Settings (YAML + env + defaults, pydantic)
├── logger.py            # structured JSON + rotating logging
├── utils.py             # hashing, parallel map, classifiers, streaming
├── _db.py               # shared SQLite connection helper
├── manager_paths.py     # filesystem layout bootstrap
├── downloader.py        # KaggleDownloader
├── scanner.py           # DatasetScanner
├── validator.py         # DatasetValidator
├── metadata.py          # MetadataGeneratorImpl + SQLiteMetadataStore
├── csv_loader.py        # PandasCSVLoader
├── image_loader.py      # RasterioImageLoader (+ light TIFF parser)
├── cache_manager.py     # CacheManager
├── dataset_registry.py  # SQLiteRegistry
├── version_manager.py   # SQLiteVersionManager
├── cli.py               # argparse CLI
├── manage_dataset.py    # executable entry point
├── __main__.py          # python -m services.dataset_manager
├── pyproject.toml       # pytest / ruff config
├── requirements.txt
├── README.md
├── docs/                # architecture / install / usage / developer docs
└── tests/               # pytest suite
```

## Coding standards

* **Python 3.12+**, PEP 8, type hints on all public functions.
* **Docstrings** — Google style, present on every public function/class.
* **Clean architecture** — adapters implement `interfaces.py` ports; the
  facade never imports concrete internals directly.
* **No duplicate code** — shared SQLite logic lives in `_db.py`; shared
  classification/hashing lives in `utils.py`.
* **Logging** — use `get_logger(name)`; avoid the reserved `extra` keys
  (use `log_dict(logger, level, event, **fields)` when in doubt).
* **No TODOs / placeholders** — every function is implemented and covered.

## Conventions

| Thing | Convention |
|-------|------------|
| Modules | `snake_case.py` |
| Classes | `PascalCase` |
| Functions / variables | `snake_case` |
| Constants | `UPPER_SNAKE_CASE` |
| Error codes | `DM-<AREA>-<NNN>` |
| Cache keys | `namespace:key` (e.g. `scan:/root/path`) |
| Env vars | `DM_<SECTION>__<FIELD>` |
| Validation issue codes | `V-<AREA>-<NNN>` |

## Testing

```bash
cd services/dataset_manager
pytest -v
```

* **Unit tests** — each adapter in isolation (`test_*` per module).
* **Integration tests** — `test_manager.py` exercises the full pipeline on a
  synthetic dataset; `test_cli.py` runs the CLI end-to-end.
* **Mocks** — downloads are mocked (`fake_kaggle` fixture); GeoTIFFs are
  generated in-memory; the real Kaggle cache is never touched.

## Adding a capability

1. Declare the port in `interfaces.py` (if new).
2. Implement the adapter in its own module.
3. Wire it into `DatasetManager.__init__` as an injectable dependency with a
   sensible default factory.
4. Add tests under `tests/`.
5. Add a CLI subcommand (if user-facing) and document it in `README.md` /
   `docs/USAGE.md`.

## Extending the validator

Add a check as a `_check_*` method returning `list[ValidationIssue]`, then
call it from `DatasetValidator.validate`. Reuse existing codes or add a new
`V-<AREA>-<NNN>` code in the docstring of the method.
