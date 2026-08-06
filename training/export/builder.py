"""Dataset-export orchestration.

:func:`export_dataset` takes the feature frame (optionally an
:class:`~training.stam.observation_resolver.ObservationCorpus` for metadata)
plus an :class:`ExportConfig`, writes every requested format into
``output_dir`` and returns a manifest mapping format -> artifact path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExportConfig
from .exceptions import ExportFormatError
from .exporters import JsonExporter, ParquetExporter, TorchExporter
from .logger import get_logger
from .records import attach_meta

logger = get_logger("builder")

_FORMAT_EXPORTERS = {
    "json": lambda e: e.export,
    "jsonl": lambda e: e.export_jsonl,
    "parquet": lambda e: e.export,
    "torch": lambda e: e.export,
}


def export_dataset(
    frame: pd.DataFrame,
    corpus: Any | None = None,
    config: ExportConfig | None = None,
) -> dict[str, str]:
    """Write a feature frame to the configured formats.

    Args:
        frame: Rectangular feature frame (see
            :func:`~training.feature_engineering.builder.build_feature_frame`).
        corpus: Optional corpus (or sample list) used to attach ``sample_id`` /
            ``year`` / ``season`` / ``quality_score`` metadata.
        config: Export settings (defaults to :class:`ExportConfig`).

    Returns:
        A mapping ``{format: artifact_path}``.

    Raises:
        ExportFormatError: When a configured format is unknown.
    """
    config = config or ExportConfig()
    if config.include_meta:
        frame = attach_meta(frame, corpus)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_exporter = JsonExporter()
    parquet_exporter = ParquetExporter()
    torch_exporter = TorchExporter()

    written: dict[str, str] = {}
    for fmt in config.formats:
        if fmt not in _FORMAT_EXPORTERS:
            raise ExportFormatError(f"Unsupported export format: {fmt}")
        suffix = "pt" if fmt == "torch" else fmt
        target = output_dir / f"{config.prefix}.{suffix}"
        if fmt == "json":
            exporter = json_exporter
        elif fmt == "jsonl":
            exporter = json_exporter
        elif fmt == "parquet":
            exporter = parquet_exporter
        else:
            exporter = torch_exporter
        written[fmt] = str(_FORMAT_EXPORTERS[fmt](exporter)(frame, target))

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prefix": config.prefix,
        "rows": len(frame),
        "columns": list(frame.columns),
        "formats": written,
    }
    if config.write_manifest:
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    logger.info(
        "Dataset exported",
        extra={"rows": len(frame), "formats": sorted(written), "dir": str(output_dir)},
    )
    return written
