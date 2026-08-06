"""Dataset export package (R2.3).

Writes generated training datasets to portable artifacts. A feature frame from
:func:`~training.feature_engineering.builder.build_feature_frame` (plus an
optional corpus for metadata) is normalised once and emitted as JSON / NDJSON /
Parquet / PyTorch tensor payloads, with a ``manifest.json`` describing every
artifact.

Typical usage::

    from training.feature_engineering import build_feature_frame
    from training.export import ExportConfig, export_dataset

    frame = build_feature_frame(corpus)
    artifacts = export_dataset(frame, corpus=corpus, config=ExportConfig(
        output_dir="data/out/datasets", formats=["json", "parquet", "torch"],
    ))
"""

from __future__ import annotations

from .builder import export_dataset
from .config import (
    ExportConfig,
    SUPPORTED_FORMATS,
    load_export_config,
    save_export_template,
)
from .exceptions import (
    ExportConfigError,
    ExportError,
    ExportFormatError,
    ExportWriteError,
)
from .exporters import JsonExporter, ParquetExporter, TorchExporter
from .records import attach_meta, frame_to_records

__version__ = "0.1.0"

__all__ = [
    "ExportConfig",
    "ExportConfigError",
    "ExportError",
    "ExportFormatError",
    "ExportWriteError",
    "JsonExporter",
    "ParquetExporter",
    "SUPPORTED_FORMATS",
    "TorchExporter",
    "attach_meta",
    "export_dataset",
    "frame_to_records",
    "load_export_config",
    "save_export_template",
]
