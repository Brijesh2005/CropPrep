"""Common type aliases shared across the CropFusion platforms."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeAlias

#: Any path-like value accepted by file helpers.
PathLike: TypeAlias = str | os.PathLike[str]

#: A JSON-serialisable value.
JSONValue: TypeAlias = (
    dict[str, "JSONValue"]
    | list["JSONValue"]
    | tuple["JSONValue", ...]
    | str
    | int
    | float
    | bool
    | None
)

#: A JSON object (mapping).
JSONObject: TypeAlias = dict[str, JSONValue]

#: A loaded configuration mapping (pre-validation).
SettingsMapping: TypeAlias = dict[str, Any]

#: A mapping of environment variables.
EnvMapping: TypeAlias = dict[str, str]

#: Bounds tuple ``(left, bottom, right, top)``.
Bounds: TypeAlias = tuple[float, float, float, float]

#: Pixel size tuple ``(x_res, y_res)``.
PixelSize: TypeAlias = tuple[float, float]

#: Image window ``(row_off, col_off, height, width)``.
ImageWindow: TypeAlias = tuple[int, int, int, int]

#: Result of a version comparison (negative / zero / positive).
VersionComparison: TypeAlias = int

__all__ = [
    "Bounds",
    "EnvMapping",
    "ImageWindow",
    "JSONObject",
    "JSONValue",
    "PathLike",
    "PixelSize",
    "SettingsMapping",
    "VersionComparison",
]
