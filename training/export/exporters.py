"""Per-format exporters for generated training datasets.

Each exporter writes one artifact type from the same normalised
:class:`pandas.DataFrame`:

* :class:`JsonExporter`  -> ``*.json`` (array) and ``*.jsonl`` (NDJSON)
* :class:`ParquetExporter` -> ``*.parquet`` (pyarrow / pandas engine)
* :class:`TorchExporter`  -> ``*.pt`` dict with a numeric feature tensor

All exporters return the path they wrote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .exceptions import ExportWriteError
from .logger import get_logger
from .records import frame_to_records, numeric_columns, write_jsonl_records, write_json_records

logger = get_logger("exporters")


class JsonExporter:
    """Write records as a JSON array and/or NDJSON stream."""

    def export(self, frame: pd.DataFrame, path: str | Path) -> Path:
        target = Path(path)
        try:
            records = frame_to_records(frame)
            write_json_records(records, target)
        except OSError as exc:
            raise ExportWriteError(f"Failed to write JSON export: {exc}", detail=str(target)) from exc
        logger.info("Wrote JSON export", extra={"path": str(target), "rows": len(records)})
        return target

    def export_jsonl(self, frame: pd.DataFrame, path: str | Path) -> Path:
        target = Path(path)
        try:
            records = frame_to_records(frame)
            write_jsonl_records(records, target)
        except OSError as exc:
            raise ExportWriteError(f"Failed to write JSONL export: {exc}", detail=str(target)) from exc
        logger.info("Wrote JSONL export", extra={"path": str(target), "rows": len(records)})
        return target


class ParquetExporter:
    """Write the feature frame to a Parquet file."""

    def export(self, frame: pd.DataFrame, path: str | Path) -> Path:
        target = Path(path)
        try:
            frame.to_parquet(target, index=False, engine="pyarrow")
        except ImportError as exc:
            raise ExportWriteError(
                "Parquet export requires pyarrow; install it or drop the format",
                detail=str(exc),
            ) from exc
        except OSError as exc:
            raise ExportWriteError(f"Failed to write Parquet export: {exc}", detail=str(target)) from exc
        logger.info("Wrote Parquet export", extra={"path": str(target), "rows": len(frame)})
        return target


class TorchExporter:
    """Write a :class:`dict` of numeric features + sample ids to ``.pt``.

    The payload is ``{"sample_id": [...], "features": tensor,
    "feature_names": [...], "n_samples": int}`` so a PyTorch ``DataLoader``
    can wrap it directly.
    """

    def __init__(self) -> None:
        try:
            import torch  # noqa: F401

            self.torch = torch
        except ImportError:
            self.torch = None

    def export(self, frame: pd.DataFrame, path: str | Path) -> Path:
        if self.torch is None:
            raise ExportWriteError(
                "Torch export requires PyTorch; install it or drop the format"
            )
        columns = numeric_columns(frame)
        if not columns:
            raise ExportWriteError("Torch export requires at least one numeric feature column")

        target = Path(path)
        try:
            features = self.torch.tensor(frame[columns].to_numpy(dtype="float32"))
            payload: dict[str, Any] = {
                "sample_id": [
                    (str(value) if value is not None else None)
                    for value in frame.get("sample_id", pd.Series([None] * len(frame)))
                ],
                "features": features,
                "feature_names": columns,
                "n_samples": len(frame),
            }
            self.torch.save(payload, target)
        except OSError as exc:
            raise ExportWriteError(f"Failed to write Torch export: {exc}", detail=str(target)) from exc
        logger.info(
            "Wrote Torch export",
            extra={"path": str(target), "rows": len(frame), "features": len(columns)},
        )
        return target
