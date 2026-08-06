"""Report builders for the Dataset Manager.

Seven report families (R2.2):

* **inventory**   — every scanned file, categorised.
* **csv**         — per tabular dataset: rows, columns, size, missing values.
* **image**       — NDVI / EVI counts by year and resolution + location.
* **provider**    — registered providers, health, capabilities, priority.
* **spatial**     — spatial index metadata and locations.
* **temporal**    — index x year x resolution availability.
* **validation**  — full validation report (structure/integrity/metadata).

:func:`generate_reports` writes each report as JSON under ``report_dir`` and
returns the written paths. All data flows through the Dataset Manager — no
direct file access.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .logger import get_logger

logger = get_logger("reports")


def build_inventory_report(manager: Any) -> dict[str, Any]:
    """Report of every scanned file, categorised."""
    inventory = manager.inventory()
    return {
        "kind": "inventory",
        "root": str(inventory.root),
        "scanned_at": inventory.scanned_at.isoformat(),
        "source": inventory.source,
        "duration_s": round(inventory.duration_s, 3),
        "counts": inventory.counts(),
        "files": [e.to_dict() for e in inventory.entries],
    }


def build_csv_report(manager: Any) -> dict[str, Any]:
    """Per-tabular-dataset report: rows, columns, size, missing values."""
    datasets: list[dict[str, Any]] = []
    for name in manager.tabular_names():
        try:
            schema = manager.tabular_schema(name)
            missing = manager.tabular_missing(name)
            metadata = manager.tabular_metadata(name)
        except Exception as exc:  # noqa: BLE001 - best effort
            datasets.append({"name": name, "error": str(exc)})
            continue
        datasets.append(
            {
                "name": name,
                "rows": schema.get("row_count"),
                "columns": schema.get("columns", []),
                "column_count": schema.get("column_count", 0),
                "size_bytes": metadata.get("size_bytes", 0),
                "missing_values": missing,
                "total_missing": sum(missing.values()),
            }
        )
    return {
        "kind": "csv",
        "count": len(datasets),
        "datasets": datasets,
    }


def build_image_report(manager: Any) -> dict[str, Any]:
    """NDVI / EVI counts by year and resolution + dataset location."""
    catalog = manager.image_catalog()
    by_year_index: dict[str, dict[str, int]] = {}
    for entry in catalog.ndvi + catalog.evi:
        key = str(entry.year) if entry.year is not None else "unknown"
        bucket = by_year_index.setdefault(key, {"ndvi": 0, "evi": 0})
        bucket[entry.index_type.value.lower()] += 1
    return {
        "kind": "image",
        "location": catalog.location.to_dict(),
        "years": catalog.years,
        "resolutions": catalog.resolutions,
        "ndvi_count": len(catalog.ndvi),
        "evi_count": len(catalog.evi),
        "counts": catalog.counts,
        "by_year_index": {
            k: v for k, v in sorted(by_year_index.items())
        },
    }


def build_provider_report(manager: Any) -> dict[str, Any]:
    """Registered providers: registration, health, capabilities."""
    registry = manager.provider_registry
    return {
        "kind": "provider",
        "registered": registry.discovery(),
        "availability": registry.availability(),
        "health": registry.health(),
        "capabilities": registry.capabilities(),
    }


def build_spatial_report(manager: Any) -> dict[str, Any]:
    """Spatial index metadata and registered locations."""
    spatial_index = manager.spatial_index
    if spatial_index is None:
        return {"kind": "spatial", "count": 0, "metadata": None, "locations": []}
    metadata = spatial_index.metadata()
    return {
        "kind": "spatial",
        "metadata": metadata.to_dict(),
        "count": metadata.count,
        "locations": [r.to_dict() for r in spatial_index.records()],
    }


def build_temporal_report(manager: Any) -> dict[str, Any]:
    """Index x year x resolution availability (persisted or derived)."""
    repository = getattr(manager, "metadata_repository", None)
    if repository is not None:
        records = repository.list_temporal()
        if records:
            return {"kind": "temporal", "records": records}

    catalog = manager.image_catalog()
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for entry in catalog.ndvi + catalog.evi:
        year = entry.year
        if year is None:
            continue
        key = (entry.index_type.value, year, entry.resolution.value)
        bucket = rows.setdefault(
            key, {"index_type": key[0], "year": key[1], "resolution": key[2], "count": 0}
        )
        bucket["count"] += 1
    return {
        "kind": "temporal",
        "records": [
            {"index_type": k[0], "year": k[1], "resolution": k[2], "count": v["count"]}
            for k, v in sorted(rows.items())
        ],
    }


def build_validation_report(manager: Any) -> dict[str, Any]:
    """Full validation report (structure / integrity / metadata / providers)."""
    report = manager.validate()
    return {
        "kind": "validation",
        "passed": report.passed,
        "root": str(report.root),
        "files_scanned": report.files_scanned,
        "validated_at": report.validated_at.isoformat(),
        "by_severity": report.by_severity(),
        "by_category": report.by_category(),
        "issues": [i.to_dict() for i in report.issues],
    }


#: Ordered report builders (name -> function).
REPORT_BUILDERS: dict[str, Any] = {
    "inventory": build_inventory_report,
    "csv": build_csv_report,
    "image": build_image_report,
    "provider": build_provider_report,
    "spatial": build_spatial_report,
    "temporal": build_temporal_report,
    "validation": build_validation_report,
}


def generate_reports(
    manager: Any, report_dir: str | Path | None = None
) -> list[Path]:
    """Write every report as JSON under ``report_dir``.

    Args:
        manager: A wired :class:`DatasetManager`.
        report_dir: Output directory (defaults to
            ``<dataset_root>/.cropfusion/reports``).

    Returns:
        The list of written JSON paths.
    """
    out_dir = Path(report_dir) if report_dir else manager.settings.state_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, builder in REPORT_BUILDERS.items():
        try:
            report = builder(manager)
        except Exception as exc:  # noqa: BLE001 - one bad report shouldn't block the rest
            report = {"kind": name, "error": str(exc)}
        report["generated_at"] = datetime.now().isoformat()
        path = out_dir / f"{name}_report.json"
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        written.append(path)
        logger.info("Wrote report", extra={"report_name": name, "path": str(path)})
    return written
