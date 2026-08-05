"""Naming-convention classifiers shared across platforms.

These helpers detect vegetation index, resolution band, year and observation
date from raster file names / paths.  They were ported verbatim from the
Dataset Manager utilities so the Application platform can use them too.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from ..enums import IndexType, Resolution

_YEAR_RE = re.compile(r"(?<![\d])(20\d{2}|19\d{2})(?![\d])")
_OBS_DATE_RE = re.compile(
    r"(?<![\d])(20\d{2})[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])(?![\d])"
)
_R10_RE = re.compile(r"R\s?10\s?m", re.IGNORECASE)
_R20_RE = re.compile(r"R\s?20\s?m", re.IGNORECASE)


def classify_index_type(name: str) -> IndexType:
    """Detect the vegetation index from a path segment or file name."""
    upper = name.upper()
    if "NDVI" in upper:
        return IndexType.NDVI
    if "EVI" in upper:
        return IndexType.EVI
    return IndexType.NONE


def classify_index_type_from_path(path: str | Path) -> IndexType:
    """Detect the index by inspecting every path segment of ``path``."""
    parts = Path(path).parts
    for part in reversed(parts):
        detected = classify_index_type(part)
        if detected is not IndexType.NONE:
            return detected
    return IndexType.NONE


def classify_resolution(name: str) -> Resolution:
    """Detect the resolution band (R10m/R20m) from a path segment or name."""
    if _R10_RE.search(name):
        return Resolution.R10M
    if _R20_RE.search(name):
        return Resolution.R20M
    # Sentinel-2 naming convention: S2A_..._10m_... / _20m_
    if re.search(r"(?<![Rr])\b10m\b", name, re.IGNORECASE):
        return Resolution.R10M
    if re.search(r"(?<![Rr])\b20m\b", name, re.IGNORECASE):
        return Resolution.R20M
    return Resolution.UNKNOWN


def classify_resolution_from_path(path: str | Path) -> Resolution:
    """Detect the resolution by inspecting every path segment of ``path``."""
    parts = Path(path).parts
    for part in parts:
        detected = classify_resolution(part)
        if detected is not Resolution.UNKNOWN:
            return detected
    return Resolution.UNKNOWN


def extract_year_from_path(path: str | Path) -> int | None:
    """Extract the first 4-digit year (19xx/20xx) found in the path."""
    match = _YEAR_RE.search(str(path))
    if not match:
        return None
    year = int(match.group(1))
    return year if 1950 <= year <= 2100 else None


def parse_observation_date(path: str | Path) -> date | None:
    """Parse an ``YYYY[_-]?MM[_-]?DD`` date from a file name, if present."""
    match = _OBS_DATE_RE.search(Path(path).name)
    if not match:
        return None
    year, month, day = (int(g) for g in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None
