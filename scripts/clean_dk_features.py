"""Clean the Earth-Engine ``DK_Features_*.csv`` exports for mapping.

Fixes the three problems found in the raw exports so the gridded points can be
joined to the KGIS admin boundaries safely:

1. ``District`` is re-assigned by point-in-polygon against the KGIS district
   shapefile instead of trusting the (incorrect) source label. The original
   label is kept in ``District_Source`` and a ``In_Admin_Boundary`` flag
   marks points that fall outside every district polygon (coastal sea).
2. ``.geo`` (an empty ``MultiPoint`` in 2018-2023) is replaced by a real
   ``geometry`` column of WKT ``POINT``s (EPSG:4326) rebuilt from
   ``Longitude``/``Latitude``.
3. ``Year`` is filled from the filename for years that omitted it
   (``DK_Features_2024 (1).csv``).

Usage::

    python scripts/clean_dk_features.py
    python scripts/clean_dk_features.py --source Tabular_Datasets --output Tabular_Datasets/cleaned
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = _REPO_ROOT / "Tabular_Datasets"
DEFAULT_OUTPUT = _REPO_ROOT / "Tabular_Datasets" / "cleaned"
DEFAULT_DISTRICT_SHP = _REPO_ROOT / "application" / "gis" / "District" / "District.shp"
DISTRICT_NAME_COL = "KGISDist_1"

#: ``DK_Features_2024 (1).csv`` -> ``2024``
_YEAR_RE = re.compile(r"20\d\d")


def _year_from_name(name: str) -> int | None:
    match = _YEAR_RE.search(name)
    return int(match.group()) if match else None


def clean_one(
    source: Path,
    districts: gpd.GeoDataFrame,
    output_dir: Path,
    year: int | None,
) -> dict:
    """Clean one DK_Features CSV and write the mapped-ready output."""
    frame = pd.read_csv(source)
    original_rows = len(frame)

    if "Latitude" not in frame.columns or "Longitude" not in frame.columns:
        raise ValueError(f"{source.name}: no Latitude/Longitude columns")

    if year is None:
        raise ValueError(f"{source.name}: cannot infer year from filename")

    if "Year" in frame.columns:
        year_values = set(frame["Year"].dropna().astype(int).unique())
        if year_values and year_values != {year}:
            raise ValueError(
                f"{source.name}: Year column values {sorted(year_values)} "
                f"disagree with filename year {year}"
            )
    frame["Year"] = year

    geometry = gpd.points_from_xy(frame["Longitude"], frame["Latitude"])
    points = gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")

    boundary = districts[["KGISDist_1", "geometry"]].copy()
    joined = gpd.sjoin(points, boundary, how="left", predicate="within")
    joined = (
        joined.reset_index()
        .drop_duplicates(subset="index", keep="first")
        .set_index("index")
    )
    joined.index.name = None
    joined = joined.sort_index()

    assigned = joined["KGISDist_1"]
    in_admin = assigned.notna()

    original_district = frame["District"] if "District" in frame.columns else None
    frame["District"] = assigned.where(in_admin)
    frame["District_Source"] = original_district
    frame["In_Admin_Boundary"] = in_admin
    frame["geometry"] = [geom.wkt for geom in geometry]

    frame = frame.drop(columns=[".geo"] if ".geo" in frame.columns else [])

    column_order = [c for c in frame.columns if c not in
                    ("District", "District_Source", "In_Admin_Boundary", "geometry")]
    trailing = ["District", "District_Source", "In_Admin_Boundary", "geometry"]
    frame = frame[column_order + trailing]

    out_path = output_dir / f"{source.stem.replace(' (1)', '')}_cleaned.csv"
    frame.to_csv(out_path, index=False, encoding="utf-8")

    district_counts = frame["District"].value_counts()
    dk_count = int((frame["District"] == "Dakshina Kannada").sum())
    other_count = int((in_admin & (frame["District"] != "Dakshina Kannada")).sum())
    outside_count = int((~in_admin).sum())
    duplicate_grids = int(frame.duplicated(subset=["Latitude", "Longitude"]).sum())

    return {
        "source": source.name,
        "output": out_path.name,
        "year": year,
        "rows": original_rows,
        "in_dk": dk_count,
        "in_other_district": other_count,
        "outside_admin": outside_count,
        "pct_in_admin": round(float(in_admin.mean()), 4),
        "pct_in_dk": round(dk_count / original_rows, 4),
        "districts_present": " | ".join(district_counts.dropna().index.astype(str)),
        "duplicate_gridpoints": duplicate_grids,
        "has_yield_column": "Yield_Proxy_NPP" in frame.columns,
        "has_season_column": "Season" in frame.columns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="dir with DK_Features CSVs")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="cleaned output dir")
    parser.add_argument("--districts", default=str(DEFAULT_DISTRICT_SHP),
                        help="KGIS District shapefile")
    parser.add_argument("--files", nargs="*", default=None,
                        help="only clean these filenames (default: all DK_Features_*.csv)")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    if not source_dir.is_dir():
        print(f"source dir not found: {source_dir}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    district_shp = Path(args.districts)
    if not district_shp.is_file():
        print(f"district shapefile not found: {district_shp}", file=sys.stderr)
        return 1
    districts = gpd.read_file(str(district_shp))
    if DISTRICT_NAME_COL not in districts.columns:
        raise KeyError(f"{district_shp.name} has no {DISTRICT_NAME_COL!r} column")
    districts = districts.to_crs("EPSG:4326")

    files = sorted(
        source_dir.glob("DK_Features_*.csv")
        if not args.files
        else [source_dir / name for name in args.files]
    )
    if not files:
        print(f"no DK_Features CSVs found under {source_dir}")
        return 0

    summaries: list[dict] = []
    for path in files:
        year = _year_from_name(path.name)
        print(f"cleaning {path.name} ... ", end="", flush=True)
        try:
            summary = clean_one(path, districts, output_dir, year)
        except Exception as exc:
            print(f"ERROR {type(exc).__name__}: {exc}")
            continue
        summaries.append(summary)
        print(
            f"OK -> {summary['output']} "
            f"({summary['rows']} rows; {summary['pct_in_dk']:.0%} in DK, "
            f"{summary['pct_in_admin']:.0%} inside any district)"
        )

    if summaries:
        summary_frame = pd.DataFrame(summaries)
        summary_frame.to_csv(output_dir / "_summary.csv", index=False, encoding="utf-8")
        print(f"\nsummary written -> {output_dir / '_summary.csv'}")
        print(summary_frame[["year", "rows", "in_dk", "in_other_district",
                             "outside_admin", "duplicate_gridpoints",
                             "has_yield_column"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
