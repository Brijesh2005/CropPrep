"""JSON serialization helpers."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    """Default JSON encoder for enums, paths and datetimes."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_dumps(data: Any, *, indent: int | None = None, **kwargs: Any) -> str:
    """JSON-serialize ``data`` using the shared default encoder."""
    return json.dumps(data, default=json_default, indent=indent, **kwargs)


def json_loads(raw: str | bytes) -> Any:
    """Deserialize a JSON string or bytes."""
    return json.loads(raw)


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    """Write ``data`` as pretty JSON and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json_dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    return out


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
