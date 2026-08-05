# Development

Guidelines for working in the CropFusion repository.

## Environment

Follow [INSTALLATION.md](INSTALLATION.md). The Makefile encodes common tasks:

```bash
make install          # pip install everything
make install-dev      # + dev tooling and frontend deps
make test             # full Python suite
make lint             # ruff on Python packages
make format           # ruff format
make typecheck        # mypy
make security         # bandit + safety
make compose          # validate compose files
make docs             # build the docs site
```

## Repository conventions

- **Python 3.12**, type hints throughout, pydantic v2 for configuration.
- **12-factor configuration**: env (`PREFIX_...`) > YAML file > defaults.
- **Modular monolith**: new domain features go in `backend/app/modules/<name>/`
  with `router.py`, `service.py`, `schemas/` and tests.
- **First-party packages** (`ai/*`, `services/*`, `quality/*`) each have their
  own `pyproject.toml`, docs/ and tests/.
- No emojis in code; ASCII diagrams allowed; docstrings required on public APIs.

## Code style

- `ruff` (E, F, I, B, UP, S; line length 88) - configured in each package's
  `pyproject.toml` and at the root.
- `mypy` for type checking.
- Frontend: ESLint + Prettier (see `frontend/`).

## Testing

See [TESTING.md](TESTING.md). Summary:

```bash
pytest                          # all Python suites from repo root
pytest backend/app/tests -q     # backend only
pytest quality -q               # quality package
cd frontend && npm test         # frontend unit tests
```

## Branching & PRs

- PRs target `main`; CI runs lint, tests, coverage, security scans and compose
  validation (`.github/workflows/`).
- Use the [PR template](../.github/PULL_REQUEST_TEMPLATE.md).
- No credentials or `.env` files in commits (see `.gitignore`).

## Adding a backend module

1. Create `backend/app/modules/<name>/` with `router.py`, `service.py`,
   `schemas/__init__.py`.
2. Add ORM models in `backend/database/models/` and a migration.
3. Register the router in `app/main.py`.
4. Add `backend/app/tests/test_<name>.py` using the existing fixtures.
5. Run `pytest backend/app/tests -q` and `make lint`.

## Documentation

Keep `docs/` current: update the relevant guide when behaviour changes. Build
the docs site with `make docs` and preview it via the `docs` compose service.
