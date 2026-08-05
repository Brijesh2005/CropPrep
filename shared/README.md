# Shared Contracts (`shared/`)

The **Shared** layer holds platform-agnostic contracts used by both the
Training Platform (`training/`) and the Prediction Platform (`application/`).
Neither platform may import the other; everything they share must live here.

## Layout

| Directory | Responsibility |
| --------- | -------------- |
| `schemas/` | JSON / DB / API schema definitions |
| `dto/` | Data-transfer objects used across platforms |
| `enums/` | Shared enumerations |
| `interfaces/` | Abstract interfaces / protocols |
| `validation/` | Cross-platform validation rules |
| `utils/` | Shared utilities (formatting, geo helpers, ...) |
| `exceptions/` | Shared exception hierarchy |
| `config/` | Shared configuration models |
| `constants/` | Shared constants |
| `serialization/` | Shared (de)serialization helpers |
| `tests/` | Tests for the shared layer |

## Import rules

- `training.*` may import `shared.*`.
- `application.*` may import `shared.*`.
- `training.*` and `application.*` must never import each other.

Add a package root to your `sys.path` (or `PYTHONPATH`) to import it, e.g.
`import shared.schemas` once the repo root is on the path.
