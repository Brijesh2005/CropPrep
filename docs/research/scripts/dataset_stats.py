"""Compute dataset statistics for the CropFusion research assets.

Usage::

    python research/scripts/dataset_stats.py [--out research/DATASETS.md]

Scans ``Tabular_Datasets/*.csv`` and emits a JSON snapshot plus an optional
Markdown report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def stats_for(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include="number")
    return {
        "file": path.name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "numeric_columns": int(numeric.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "columns_sample": [str(c) for c in df.columns[:12]],
        "min_year": int(df[df.columns[0]].min()) if len(df) and pd.api.types.is_numeric_dtype(df[df.columns[0]]) else None,
        "max_year": int(df[df.columns[0]].max()) if len(df) and pd.api.types.is_numeric_dtype(df[df.columns[0]]) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Tabular_Datasets")
    parser.add_argument("--out", default=None, help="markdown report path")
    args = parser.parse_args()

    source = Path(args.source)
    results = []
    for path in sorted(source.glob("*.csv")):
        results.append(stats_for(path))

    snapshot = {
        "datasets": results,
        "total_rows": sum(int(r["rows"]) for r in results),
        "total_columns": sum(int(r["columns"]) for r in results),
    }
    Path("research/dataset_stats.json").write_text(
        json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(snapshot, indent=2, default=str))

    if args.out:
        lines = ["# Dataset statistics\n", "| Dataset | Rows | Columns | Missing cells | Memory (MB) |"]
        lines.append("|---|---:|---:|---:|---:|")
        for r in results:
            lines.append(
                f"| {r['file']} | {r['rows']:,} | {r['columns']} | {r['missing_cells']:,} "
                f"| {r['memory_mb']} |"
            )
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
