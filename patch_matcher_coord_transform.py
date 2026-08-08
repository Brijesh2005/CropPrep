"""Patch training/stam/matcher.py to fix ST-COORD-001 "Out-of-range
coordinates" errors.

Root cause: when building image-record location points,
`DatasetManagerLocationCatalog.points()` took the raster bounds centroid
directly as `(lon, lat)` without reprojecting it out of the raster's native
CRS (e.g. UTM zone 43N / EPSG:32643, with coordinates in metres such as
504870.0, 1495110.0). Every downstream coordinate-range check then rejected
these as out-of-range for WGS-84 degrees, so 100% of resolved cells came
back with status="error".

This patches the image-centroid branch to reproject through
`coordinate_transform.transform_point()` (already present in the codebase)
before constructing the `LocationPoint`.

Run this from the repository root (the folder that contains `training/`):

    python patch_matcher_coord_transform.py

Or point it at a specific repo:

    python patch_matcher_coord_transform.py --repo-root /path/to/CropPrep

Idempotent — safe to run more than once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET_REL_PATH = "training/stam/matcher.py"

OLD_IMPORT = "from .config import AdminConfig, StamConfig, TabularConfig"
NEW_IMPORT = (
    "from .config import AdminConfig, StamConfig, TabularConfig\n"
    "from .coordinate_transform import WGS84, transform_point"
)

OLD_BLOCK = '''        for record in images:
            if not record.bounds:
                continue
            left, bottom, right, top = record.bounds
            lon, lat = (left + right) / 2.0, (bottom + top) / 2.0
            points.append(
                LocationPoint(
                    id=f"image:{record.relative_path}",
                    name=Path(record.relative_path).stem,
                    lon=float(lon),
                    lat=float(lat),
                    meta={"source": "image", "index_type": record.index_type.value,
                          "year": record.year},'''

NEW_BLOCK = '''        for record in images:
            if not record.bounds:
                continue
            left, bottom, right, top = record.bounds
            x, y = (left + right) / 2.0, (bottom + top) / 2.0
            # `record.bounds` is in the raster's native CRS (e.g. UTM metres
            # such as EPSG:32643), not WGS-84 degrees. STAM's public
            # boundary always deals in (lon, lat) WGS-84 — see
            # coordinate_transform.py — so the centroid must be reprojected
            # before being stored as a LocationPoint, otherwise every
            # downstream coordinate-range check (ST-COORD-001) fails.
            if record.crs:
                lon, lat = transform_point(record.crs, WGS84, x, y)
            else:
                lon, lat = x, y
            points.append(
                LocationPoint(
                    id=f"image:{record.relative_path}",
                    name=Path(record.relative_path).stem,
                    lon=float(lon),
                    lat=float(lat),
                    meta={"source": "image", "index_type": record.index_type.value,
                          "year": record.year},'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the CropPrep repository root (default: current directory)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    target = repo_root / TARGET_REL_PATH

    if not target.is_file():
        print(f"ERROR: could not find {target}", file=sys.stderr)
        print(
            "Pass --repo-root pointing at the CropPrep checkout "
            "(the folder containing 'training/').",
            file=sys.stderr,
        )
        return 1

    text = target.read_text(encoding="utf-8")
    original_text = text

    if "from .coordinate_transform import WGS84, transform_point" in text:
        print(f"[skip] {target} already patched.")
        return 0

    if OLD_IMPORT not in text:
        print(
            "ERROR: could not find the expected import line to patch. "
            "The file may already differ from the expected version.",
            file=sys.stderr,
        )
        return 1
    text = text.replace(OLD_IMPORT, NEW_IMPORT, 1)

    if OLD_BLOCK not in text:
        print(
            "ERROR: could not find the expected image-centroid block to "
            "patch. The file may already differ from the expected version.",
            file=sys.stderr,
        )
        return 1
    text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)

    if text == original_text:
        print("ERROR: no changes were applied.", file=sys.stderr)
        return 1

    backup = target.with_suffix(target.suffix + ".bak")
    backup.write_text(original_text, encoding="utf-8")
    target.write_text(text, encoding="utf-8")

    print(f"[ok] patched {target}")
    print(f"[ok] backup saved to {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
