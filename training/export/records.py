"""Feature-frame normalisation for export.

Turns a rectangular feature :class:`pandas.DataFrame` (as produced by
:func:`~training.feature_engineering.builder.build_feature_frame`) into
JSON-safe records and attaches corpus metadata (sample id, year, season,
quality score) so downstream consumers do not need the corpus object.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable

import pandas as pd

from .logger import get_logger

logger = get_logger("records")

_META_COLUMNS = ("sample_id", "year", "season", "location_id", "quality_score")


def attach_meta(frame: pd.DataFrame, corpus: Any | None = None) -> pd.DataFrame:
    """Add corpus metadata columns to a feature frame.

    When ``corpus`` is supplied its accepted observations (or, for a plain
    sample list, resolved samples) are aligned to frame rows in order. Rows
    beyond the provided metadata keep ``None`` placeholders.
    """
    out = frame.copy()
    if corpus is None or len(corpus.samples if hasattr(corpus, "samples") else []) == 0:
        return out

    samples = _corpus_samples(corpus)
    meta: list[dict[str, Any]] = []
    for index in range(len(out)):
        entry: dict[str, Any] = {}
        if index < len(samples):
            sample = samples[index]
            entry["sample_id"] = sample.sample_id if hasattr(sample, "sample_id") else None
            entry["quality_score"] = getattr(sample, "quality_score", None)
            obs = getattr(sample, "observation", None)
            if obs is not None:
                temporal = getattr(obs, "temporal", None)
                entry["year"] = getattr(temporal, "year", None)
                entry["season"] = getattr(temporal, "season", None)
            entry["location_id"] = getattr(sample, "location_id", None)
        meta.append(entry)

    for column in _META_COLUMNS:
        if column not in out.columns:
            out[column] = [entry.get(column) for entry in meta]
    return out


def frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records (``NaN`` -> ``None``)."""
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        records.append({key: _json_safe(value) for key, value in record.items()})
    return records


def write_json_records(records: Iterable[dict[str, Any]], path: Any) -> None:
    """Write records as a JSON array to ``path``."""
    import io

    buffer = io.StringIO()
    json.dump(list(records), buffer, ensure_ascii=False, indent=2)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def write_jsonl_records(records: Iterable[dict[str, Any]], path: Any) -> None:
    """Write records as newline-delimited JSON to ``path``."""
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Feature columns that can be fed to a tensor stack."""
    selected = frame.select_dtypes(include=["number", "bool"])
    return [col for col in selected.columns if col not in _META_COLUMNS]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _corpus_samples(corpus: Any) -> list[Any]:
    if hasattr(corpus, "samples"):
        samples = list(corpus.samples)
        return [s for s in samples if s.status == "accepted"] or samples
    return list(corpus)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    try:
        return str(value)
    except Exception:
        return None
