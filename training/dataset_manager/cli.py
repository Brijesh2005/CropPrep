"""Command-line interface for the Dataset Manager.

The CLI is a thin adapter over :class:`DatasetManager`. Every operation that
the facade exposes has a corresponding subcommand, and every subcommand
supports ``--json`` for machine-readable output.

Example::

    python services/dataset_manager/manage_dataset.py download
    python services/dataset_manager/manage_dataset.py validate --json
    python services/dataset_manager/manage_dataset.py scan
    python services/dataset_manager/manage_dataset.py summary
    python services/dataset_manager/manage_dataset.py metadata
    python services/dataset_manager/manage_dataset.py inventory --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .logger import get_logger
from .manager import DatasetManager

logger = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with all subcommands."""
    # A parent parser shared by the top level and every subcommand so that
    # ``--json`` / ``--config`` work before OR after the subcommand name.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", default=None, help="Path to a YAML configuration file"
    )
    common.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )

    parser = argparse.ArgumentParser(
        prog="manage_dataset",
        parents=[common],
        description="CropFusion Dataset Management System — CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _add(sub, "download", "Download the primary Kaggle dataset", _cmd_download, common)
    _add(sub, "scan", "Scan the dataset and build an inventory", _cmd_scan, common)
    _add(sub, "validate", "Validate structure, integrity and metadata", _cmd_validate, common)
    _add(sub, "metadata", "Generate metadata records for all files", _cmd_metadata, common)
    _add(sub, "inventory", "Print the full file inventory", _cmd_inventory, common)
    _add(sub, "summary", "Print a dataset summary", _cmd_summary, common)
    _add(sub, "csvs", "List discovered CSV files", _cmd_csvs, common)
    _add(sub, "images", "List GeoTIFF files (NDVI/EVI, R10m/R20m)", _cmd_images, common)
    _add(sub, "tabulars", "List Git-versioned tabular datasets", _cmd_tabulars, common)
    _add(sub, "tabular-schema", "Show the schema of a tabular dataset", _cmd_tabular_schema, common)
    _add(sub, "tabular-statistics", "Show numeric statistics of a tabular dataset", _cmd_tabular_stats, common)
    _add(sub, "image-ensure", "Download (or reuse) the imagery dataset", _cmd_image_ensure, common)
    _add(sub, "image-catalog", "Show the classified imagery catalog", _cmd_image_catalog, common)
    _add(sub, "image-patch", "Retrieve a raster patch around a center point", _cmd_image_patch, common)
    _add(sub, "providers", "Show provider manifests", _cmd_providers, common)
    _add(sub, "register", "Register the dataset in the registry", _cmd_register, common)
    _add(sub, "versions", "List dataset version history", _cmd_versions, common)
    _add(sub, "bump-version", "Bump the dataset version (major/minor/patch)", _cmd_bump, common)
    _add(sub, "rollback", "Roll back to a previously snapshotted version", _cmd_rollback, common)
    _add(sub, "cache-stats", "Show cache statistics", _cmd_cache, common)
    _add(sub, "info", "Show environment and configuration info", _cmd_info, common)
    _add(sub, "config-template", "Write an annotated YAML config template", _cmd_template, common)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level error boundary
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _emit(result, args)
    return 0


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #


def _cmd_download(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    path = manager.download(force=getattr(args, "force", False))
    return {"ok": True, "path": str(path), "status": "downloaded"}


def _cmd_scan(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    inventory = manager.scan(use_cache=not getattr(args, "refresh", False))
    return {
        "ok": True,
        "root": str(inventory.root),
        "source": inventory.source,
        "duration_s": round(inventory.duration_s, 3),
        "counts": inventory.counts(),
    }


def _cmd_validate(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    report = manager.validate(report_dir=getattr(args, "report_dir", None))
    return {
        "ok": True,
        "passed": report.passed,
        "root": str(report.root),
        "files_scanned": report.files_scanned,
        "by_severity": report.by_severity(),
        "by_category": report.by_category(),
        "issues": [i.to_dict() for i in report.issues],
    }


def _cmd_metadata(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    written = manager.generate_metadata(force=getattr(args, "force", False))
    return {"ok": True, "records_written": written, "total_records": manager.metadata_count()}


def _cmd_inventory(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    return {"ok": True, "inventory": manager.inventory().to_dict()}


def _cmd_summary(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    return {"ok": True, "summary": manager.summary().to_dict()}


def _cmd_csvs(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    files = [str(p) for p in manager.list_csvs()]
    return {"ok": True, "files": files, "count": len(files)}


def _cmd_images(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    files = [
        str(p)
        for p in manager.list_images(
            index_type=getattr(args, "index", None),
            resolution=getattr(args, "resolution", None),
            year=getattr(args, "year", None),
        )
    ]
    return {"ok": True, "files": files, "count": len(files)}


def _cmd_tabulars(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    catalog = manager.tabular_catalog()
    return {
        "ok": True,
        "root": str(catalog.root),
        "datasets": [d.to_dict() for d in catalog.datasets],
        "count": len(catalog.datasets),
    }


def _cmd_tabular_schema(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    return {"ok": True, "name": args.name, "schema": manager.tabular_schema(args.name)}


def _cmd_tabular_stats(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    return {
        "ok": True,
        "name": args.name,
        "statistics": manager.tabular_statistics(args.name),
    }


def _cmd_image_ensure(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    path = manager.ensure_image(force=getattr(args, "force", False))
    return {"ok": True, "path": str(path), "status": "downloaded"}


def _cmd_image_catalog(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    catalog = manager.image_catalog()
    return {
        "ok": True,
        "location": catalog.location.to_dict(),
        "ndvi_count": len(catalog.ndvi),
        "evi_count": len(catalog.evi),
        "years": catalog.years,
        "resolutions": catalog.resolutions,
        "counts": catalog.counts,
    }


def _cmd_image_patch(args: argparse.Namespace) -> dict[str, Any]:
    from .providers.models import PatchRequest

    manager = _manager(args)
    patch = manager.patch_image(
        PatchRequest(
            path=args.path,
            center=(args.x, args.y),
            size=args.size,
            band=getattr(args, "band", 1),
        )
    )
    return {
        "ok": True,
        "path": args.path,
        "shape": list(patch.shape),
        "dtype": str(patch.dtype),
    }


def _cmd_providers(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    return {"ok": True, "providers": manager.provider_manifests()}


def _cmd_register(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    dataset_id = manager.register()
    return {"ok": True, "dataset_id": dataset_id}


def _cmd_versions(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    versions = [v.to_dict() for v in manager.list_versions()]
    return {
        "ok": True,
        "current": manager.current_version(),
        "versions": versions,
    }


def _cmd_bump(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    entry = manager.bump_version(args.part, message=args.message)
    return {"ok": True, "version": entry.to_dict()}


def _cmd_rollback(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    entry = manager.rollback_version(args.version)
    return {"ok": True, "version": entry.to_dict()}


def _cmd_cache(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    return {"ok": True, "cache": manager.cache_stats()}


def _cmd_info(args: argparse.Namespace) -> dict[str, Any]:
    manager = _manager(args)
    return {"ok": True, "info": manager.info()}


def _cmd_template(args: argparse.Namespace) -> dict[str, Any]:
    from .config import save_settings_template

    path = Path(args.path)
    save_settings_template(path)
    return {"ok": True, "path": str(path)}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _add(
    parser: argparse._SubParsersAction,
    name: str,
    help_text: str,
    handler: Callable[[argparse.Namespace], dict[str, Any]],
    common: argparse.ArgumentParser,
) -> None:
    cmd = parser.add_parser(name, help=help_text, parents=[common])
    cmd.set_defaults(handler=handler)
    if name == "download":
        cmd.add_argument("--force", action="store_true", help="Re-download even if cached")
    elif name == "scan":
        cmd.add_argument("--refresh", action="store_true", help="Force a re-scan")
    elif name == "validate":
        cmd.add_argument("--report-dir", default=None, help="Where to write the JSON report")
    elif name == "metadata":
        cmd.add_argument("--force", action="store_true", help="Regenerate existing records")
    elif name == "images":
        cmd.add_argument("--index", default=None, help="Filter by index (NDVI/EVI)")
        cmd.add_argument("--resolution", default=None, help="Filter by resolution (R10m/R20m)")
        cmd.add_argument("--year", type=int, default=None, help="Filter by year")
    elif name == "tabular-schema":
        cmd.add_argument("name", help="Dataset name (file stem)")
    elif name == "tabular-statistics":
        cmd.add_argument("name", help="Dataset name (file stem)")
    elif name == "image-ensure":
        cmd.add_argument("--force", action="store_true", help="Re-download even if cached")
    elif name == "image-patch":
        cmd.add_argument("path", help="Raster file path")
        cmd.add_argument("--x", type=float, required=True, help="Center longitude/X (CRS units)")
        cmd.add_argument("--y", type=float, required=True, help="Center latitude/Y (CRS units)")
        cmd.add_argument("--size", type=int, default=64, help="Square patch edge (pixels)")
        cmd.add_argument("--band", type=int, default=1, help="Band index (1-based)")
    elif name == "bump-version":
        cmd.add_argument("part", choices=["major", "minor", "patch"], default="patch", nargs="?")
        cmd.add_argument("--message", default="")
    elif name == "rollback":
        cmd.add_argument("version", help="Version to restore (e.g. 1.2.0)")
    elif name == "config-template":
        cmd.add_argument("path", help="Destination for the template file")


def _manager(args: argparse.Namespace) -> DatasetManager:
    return DatasetManager.from_config(getattr(args, "config", None))


def _emit(result: dict[str, Any], args: argparse.Namespace) -> None:
    """Print a result as JSON or as a compact human readable summary."""
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))
        return
    print(_render(result))


def _render(result: dict[str, Any]) -> str:
    """Render a result dict as human readable text."""
    lines: list[str] = []

    if "path" in result:
        lines.append(f"Dataset: {result['path']}")
    if "status" in result:
        lines.append(f"Status:  {result['status']}")
    if "root" in result:
        lines.append(f"Root:    {result['root']}")
    if "source" in result:
        lines.append(f"Source:  {result['source']}")
    if "duration_s" in result:
        lines.append(f"Scan duration: {result['duration_s']} s")

    if "counts" in result:
        lines.append("Counts:")
        for key, value in result["counts"].items():
            lines.append(f"  {key:<10} {value}")

    if "passed" in result:
        lines.append(f"Validation passed: {result['passed']}")
    if "by_severity" in result:
        lines.append("Issues by severity: " + ", ".join(
            f"{k}={v}" for k, v in sorted(result["by_severity"].items())
        ))
    if "by_category" in result:
        lines.append("Issues by category: " + ", ".join(
            f"{k}={v}" for k, v in sorted(result["by_category"].items())
        ))
    if "issues" in result and result["issues"]:
        lines.append("Issues:")
        for issue in result["issues"]:
            lines.append(
                f"  [{issue['severity']}] {issue['code']} {issue['message']}"
            )

    if "records_written" in result:
        lines.append(f"Metadata records written: {result['records_written']}")
    if "total_records" in result:
        lines.append(f"Metadata records total:   {result['total_records']}")

    if "files" in result:
        lines.append(f"Files ({result['count']}):")
        for path in result["files"][:50]:
            lines.append(f"  {path}")
        if result["count"] > 50:
            lines.append(f"  ... and {result['count'] - 50} more")

    if "datasets" in result:
        lines.append(f"Tabular datasets ({result['count']}):")
        for dataset in result["datasets"]:
            lines.append(f"  {dataset['name']:<32} {dataset['size_bytes']} bytes  {dataset['relative_path']}")

    if "schema" in result:
        schema = result["schema"]
        lines.append(f"Schema of {result['name']}: {schema['column_count']} columns, "
                     f"{schema['row_count']} rows, dtypes={schema['dtypes']}")

    if "statistics" in result and "name" in result:
        lines.append(f"Statistics of {result['name']}:")
        for column, stats in result["statistics"].items():
            lines.append(f"  {column:<24} {stats}")

    if "location" in result:
        loc = result["location"]
        lines.append(f"Imagery: {loc['handle']}  materialized={loc['materialized']} "
                     f"downloaded={loc['downloaded']} files={loc['files']}")
    if "ndvi_count" in result:
        lines.append(f"NDVI: {result['ndvi_count']}  EVI: {result['evi_count']}")
        lines.append(f"Years: {result['years']}  Resolutions: {result['resolutions']}")

    if "shape" in result:
        lines.append(f"Patch at {result['path']}: shape={result['shape']} dtype={result['dtype']}")

    if "providers" in result:
        lines.append("Providers:")
        for name, manifest in result["providers"].items():
            lines.append(
                f"  {name:<24} status={manifest['status']} available={manifest['available']}"
            )

    if "current" in result:
        lines.append(f"Current version: {result['current']}")
    if "versions" in result and isinstance(result["versions"], list):
        lines.append("Version history:")
        for version in result["versions"]:
            marker = " *" if version.get("is_current") else ""
            lines.append(
                f"  {version['version']}{marker}  {version['created_at']}  {version['message']}"
            )

    if "version" in result and isinstance(result["version"], dict):
        lines.append(
            f"Version: {result['version'].get('version')} "
            f"({result['version'].get('message')})"
        )

    if "summary" in result:
        summary = result["summary"]
        lines.append(
            f"Summary: {summary['total_files']} files, "
            f"{summary['csv_count']} CSV, {summary['geotiff_count']} GeoTIFF, "
            f"{summary['ndvi_count']} NDVI, {summary['evi_count']} EVI"
        )
        lines.append(f"Years covered: {summary['years_covered']}")

    if "info" in result:
        info = result["info"]
        lines.append("Environment:")
        for key, value in info.items():
            lines.append(f"  {key}: {value}")

    if "cache" in result:
        lines.append(f"Cache: {result['cache']}")

    if "dataset_id" in result:
        lines.append(f"Dataset id: {result['dataset_id']}")

    return "\n".join(lines) if lines else json.dumps(result, default=str)


if __name__ == "__main__":  # pragma: no cover - invoked via manage_dataset.py
    raise SystemExit(main())
