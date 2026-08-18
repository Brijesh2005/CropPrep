"""Local release-package backfill for the Prediction Platform contract.

The Kaggle release build (``build_release.py``) copies the train-side
``release_sources/`` into ``cropfusion_release/``. When those sources are
missing (e.g. the train kernel uploaded only model + preprocess + configs),
the resulting package fails ``ReleasePackageLoader`` validation because the
``metadata/`` artifacts it requires are absent and ``reports/metrics.json``
is empty.

This script reconstructs those artifacts *locally* from the repo's own
tabular + raster sources, then regenerates the package manifest and checksums
so the loader accepts the package as-is:

    metadata/location_index.parquet      <- real raster grid points (DK_Features)
    metadata/village_metadata.parquet    <- per-village raster aggregates
    metadata/historical_context.parquet  <- tabular climatology (data_season.csv)
    metadata/metadata.db                 <- categorical codes + feature contract
    preprocess/yield_scaler.pkl          <- exact StandardScaler from the corpus
    reports/metrics.json                 <- recorded training/eval metrics
    configs/inference.yaml               <- full input contract (categorical,
                                             image size, temporal obs)
    version/manifest.json + checksum.json <- regenerated for the added files

Run::

    python training/kaggle/scripts/backfill_release.py \
        --release-dir releases/v2.0.0/cropfusion_release-v2.0.0 \
        --tabular "Tabular_Datasets/data_season.csv" \
        --raster "Tabular_Datasets/DK_Features_2024 (1).csv" \
        --corpus kaggle_runs/train-v6/.../corpus.json \
        --metrics kaggle_runs/train-v6/.../checkpoint.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Number of numeric tabular features the scaler was fit on.
NUMERIC_FEATURES = ["Area", "Rainfall", "Temperature", "Humidity", "price"]
#: Categorical (ordinal) tabular features in the exported model, as stored in
#: the release's historical-context artifact.
CATEGORICAL_FEATURES = ["soil_type", "irrigation"]
#: Exported model input contract.
INPUT_DIM = 7  # 5 scaled numeric + 2 ordinal categorical codes
IMAGE_SIZE = 224
TEMPORAL_OBSERVATIONS = 1

#: CSV Location -> district used by the STAM join (subset of the training
#: aliases; kept local so this script never imports training code).
_LOCATION_TO_DISTRICT: dict[str, str] = {
    "Mangalore": "Dakshina Kannada",
    "Bangalore": "Bengaluru",
    "Chikmangaluru": "Chikkamagaluru",
    "Davangere": "Davanagere",
    "Gulbarga": "Kalaburgi",
    "Madikeri": "Madikeri",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------- #
# Location index + village metadata (raster sources)
# ---------------------------------------------------------------------- #

def build_location_index(raster: pd.DataFrame, grid_step: float = 0.02) -> pd.DataFrame:
    """Nearest-neighbour service points from the DK_Features raster grid.

    Points are snapped to a ``grid_step``-degree grid so the package stays
    lean while still covering the coastal Dakshina Kannada service area.
    """
    df = raster.rename(
        columns={"Longitude": "lon", "Latitude": "lat", "District": "district"}
    )
    district = str(df["district"].mode().iloc[0])
    grid = (
        df[["lon", "lat"]]
        .round(int(round(-np.log10(grid_step))))
        .drop_duplicates()
        .reset_index(drop=True)
    )
    grid["village"] = district
    grid["district"] = district
    grid["taluk"] = None
    return grid[["village", "district", "taluk", "lon", "lat"]]


_RASTER_AGG_COLUMNS = [
    "NDVI",
    "EVI",
    "Annual_Rainfall_mm",
    "Elevation",
    "Slope",
    "Soil_Clay_Pct",
    "Soil_Moisture",
    "Soil_Organic_Carbon",
    "Soil_Sand_Pct",
    "Soil_pH",
]


def build_village_metadata(raster: pd.DataFrame) -> pd.DataFrame:
    """Per-village static raster aggregates (used for the image patches)."""
    rows: list[dict[str, Any]] = []
    for (village, district), group in raster.groupby(["District", "District"]):
        row: dict[str, Any] = {"village": village, "district": district}
        for col in _RASTER_AGG_COLUMNS:
            if col in group.columns:
                row[col] = float(group[col].mean())
        if "Is_Cropland" in group.columns:
            row["Is_Cropland"] = float(group["Is_Cropland"].mean())
        if "Land_Cover_Class" in group.columns:
            row["Land_Cover_Class"] = int(group["Land_Cover_Class"].mode().iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# Historical context (tabular sources)
# ---------------------------------------------------------------------- #

def _ordinal_code_map(values: Iterable[Any]) -> dict[Any, int]:
    """Deterministic category -> ordinal code (sorted by label)."""
    present = sorted({v for v in values if pd.notna(v)})
    return {v: i for i, v in enumerate(present)}


def build_historical_context(tabular: pd.DataFrame) -> pd.DataFrame:
    """Per (village, season, year) climatology with ordinal categorical codes.

    ``yeilds`` is carried for provenance only — the model never sees it as an
    input (the release scaler is fit on 5 numeric features).
    """
    soil_map = _ordinal_code_map(tabular["Soil type"])
    irrig_map = _ordinal_code_map(tabular["Irrigation"])

    df = tabular.copy()
    df["village"] = df["Location"].map(_LOCATION_TO_DISTRICT).fillna(df["Location"])
    df["district"] = df["village"]
    df["soil_type"] = df["Soil type"].map(soil_map).fillna(0).astype(int)
    df["irrigation"] = df["Irrigation"].map(irrig_map).fillna(0).astype(int)
    df["soil_type_label"] = df["Soil type"].astype(str)
    df["irrigation_label"] = df["Irrigation"].astype(str)

    grouped = (
        df.groupby(["village", "district", "Season", "Year"], dropna=False)
        .agg(
            Area=("Area", "mean"),
            Rainfall=("Rainfall", "mean"),
            Temperature=("Temperature", "mean"),
            Humidity=("Humidity", "mean"),
            price=("price", "mean"),
            yeilds=("yeilds", "mean"),
            soil_type=("soil_type", "first"),
            irrigation=("irrigation", "first"),
            soil_type_label=("soil_type_label", "first"),
            irrigation_label=("irrigation_label", "first"),
        )
        .reset_index()
    )
    grouped = grouped.rename(columns={"Season": "season", "Year": "year"})
    grouped["location"] = grouped["district"]
    return grouped


# ---------------------------------------------------------------------- #
# Yield scaler (exact, from the accepted training corpus)
# ---------------------------------------------------------------------- #

#: Matched in document order so each ``yield_value`` is attributed to the
#: ``status`` of its own sample (chunked scan: the corpus is hundreds of MB).
_CORPUS_TOKEN = re.compile(
    r'"location_id":|"status":\s*"(\w+)"|"yield_value":\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
)


def _iter_corpus_yields(corpus_path: Path) -> list[float]:
    """Accepted observations' ``yield_value``s, streamed from corpus.json."""
    yields: list[float] = []
    status: str | None = None
    chunk_size = 8 * 1024 * 1024
    tail = 256
    with corpus_path.open("r", encoding="utf-8") as fh:
        prev = ""
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            data = prev + buf
            for match in _CORPUS_TOKEN.finditer(data):
                if match.group(0) == '"location_id":':
                    status = None
                elif match.group(1) is not None:
                    status = match.group(1)
                elif match.group(2) is not None and status == "accepted":
                    yields.append(float(match.group(2)))
            prev = data[-tail:]
    return yields


def fit_yield_scaler(corpus_path: Path | None, tabular: pd.DataFrame) -> Any:
    """StandardScaler on training yield targets (sklearn, for inverse-transform)."""
    from sklearn.preprocessing import StandardScaler

    if corpus_path is not None and corpus_path.exists():
        values = _iter_corpus_yields(corpus_path)
        if values:
            return StandardScaler().fit(np.asarray(values, dtype="float64").reshape(-1, 1))
    values = tabular["yeilds"].dropna().to_numpy(dtype="float64")
    return StandardScaler().fit(values.reshape(-1, 1))


# ---------------------------------------------------------------------- #
# metadata.db
# ---------------------------------------------------------------------- #

def build_metadata_db(path: Path, *, location_index: pd.DataFrame,
                      soil_map: dict[str, int], irrig_map: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    try:
        con.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        con.execute("INSERT INTO schema_version (version) VALUES (1)")

        con.execute(
            "CREATE TABLE categorical_codes ("
            " feature TEXT NOT NULL, category TEXT NOT NULL, code INTEGER NOT NULL)"
        )
        for cat, code in soil_map.items():
            con.execute(
                "INSERT INTO categorical_codes (feature, category, code) VALUES (?, ?, ?)",
                ("soil_type", cat, code),
            )
        for cat, code in irrig_map.items():
            con.execute(
                "INSERT INTO categorical_codes (feature, category, code) VALUES (?, ?, ?)",
                ("irrigation", cat, code),
            )

        con.execute(
            "CREATE TABLE feature_contract (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        contract = {
            "feature_order": json.dumps(NUMERIC_FEATURES),
            "categorical_features": json.dumps(CATEGORICAL_FEATURES),
            "input_dim": str(INPUT_DIM),
            "image_size": str(IMAGE_SIZE),
            "temporal_observations": str(TEMPORAL_OBSERVATIONS),
        }
        for key, value in contract.items():
            con.execute(
                "INSERT INTO feature_contract (key, value) VALUES (?, ?)", (key, value)
            )

        con.execute(
            "CREATE TABLE backfill (generated_at TEXT NOT NULL, "
            " location_points INTEGER NOT NULL, sources TEXT NOT NULL)"
        )
        con.execute(
            "INSERT INTO backfill (generated_at, location_points, sources) VALUES (?, ?, ?)",
            (
                datetime.now(UTC).isoformat(),
                len(location_index),
                json.dumps({"location_index": "DK_Features raster grid",
                            "historical_context": "data_season.csv"}),
            ),
        )
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------- #
# Configs + metrics + version files
# ---------------------------------------------------------------------- #

def _update_inference_config(path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    payload.update(
        {
            "feature_order": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "input_dim": INPUT_DIM,
            "image_size": IMAGE_SIZE,
            "temporal_observations": TEMPORAL_OBSERVATIONS,
        }
    )
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _update_model_config(path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "feature_order" not in payload:
        payload["feature_order"] = list(NUMERIC_FEATURES)
    payload["input_dim"] = INPUT_DIM
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_metrics(path: Path, metrics_source: Path | None, version: str,
                   dataset_version: str) -> None:
    payload: dict[str, Any] = {"model_version": version, "dataset_version": dataset_version}
    if metrics_source is not None and metrics_source.exists():
        raw = json.loads(metrics_source.read_text(encoding="utf-8"))
        registered = raw.get("registered") or {}
        payload["metrics"] = (registered.get("metrics") or {}).get("metrics", {})
        payload["registered"] = {
            k: registered.get(k)
            for k in ("run_name", "stage", "version", "epoch", "created_at")
        }
    else:
        payload["metrics"] = {}
    _write_json(path, payload)


def _regenerate_version_files(release_dir: Path, version: str, dataset_version: str) -> None:
    """Rebuild manifest + checksum for every file currently in the package."""
    files: dict[str, Path] = {}
    for path in sorted(release_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(release_dir).as_posix()
        if rel in ("version/manifest.json", "version/checksum.json"):
            continue
        files[rel] = path

    checksums = {rel: _sha256(p) for rel, p in sorted(files.items())}
    _write_json(release_dir / "version" / "checksum.json", {"files": checksums})

    manifest = {
        "format": "cropfusion_release",
        "schema_version": 1,
        "package_name": "cropfusion",
        "model_version": version,
        "dataset_version": dataset_version,
        "released_at": datetime.now(UTC).isoformat(),
        "files": sorted(files),
    }
    _write_json(release_dir / "version" / "manifest.json", manifest)


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #

def backfill_release(
    *,
    release_dir: Path,
    tabular_path: Path,
    raster_path: Path,
    corpus_path: Path | None = None,
    metrics_path: Path | None = None,
    grid_step: float = 0.02,
) -> dict[str, Any]:
    release_dir = release_dir.resolve()
    if not release_dir.is_dir():
        raise SystemExit(f"release directory not found: {release_dir}")
    if not tabular_path.exists():
        raise SystemExit(f"tabular source not found: {tabular_path}")
    if not raster_path.exists():
        raise SystemExit(f"raster source not found: {raster_path}")

    tabular = pd.read_csv(tabular_path)
    raster = pd.read_csv(raster_path)

    print("[backfill] building location_index ...")
    location_index = build_location_index(raster, grid_step=grid_step)
    print(f"[backfill]   {len(location_index)} service points")

    print("[backfill] building village_metadata ...")
    village_metadata = build_village_metadata(raster)

    print("[backfill] building historical_context ...")
    historical_context = build_historical_context(tabular)
    soil_map = _ordinal_code_map(tabular["Soil type"])
    irrig_map = _ordinal_code_map(tabular["Irrigation"])
    print(f"[backfill]   {len(historical_context)} (village, season, year) rows")

    (release_dir / "metadata").mkdir(parents=True, exist_ok=True)
    location_index.to_parquet(release_dir / "metadata" / "location_index.parquet", index=False)
    village_metadata.to_parquet(release_dir / "metadata" / "village_metadata.parquet", index=False)
    historical_context.to_parquet(release_dir / "metadata" / "historical_context.parquet", index=False)
    build_metadata_db(
        release_dir / "metadata" / "metadata.db",
        location_index=location_index,
        soil_map=soil_map,
        irrig_map=irrig_map,
    )

    print("[backfill] fitting yield_scaler ...")
    yield_scaler = fit_yield_scaler(corpus_path, tabular)
    with (release_dir / "preprocess" / "yield_scaler.pkl").open("wb") as fh:
        pickle.dump(yield_scaler, fh)

    print("[backfill] updating configs ...")
    _update_inference_config(release_dir / "configs" / "inference.yaml")
    _update_model_config(release_dir / "configs" / "model.yaml")

    version = "2.0.0"
    dataset_version = "1.0.0"
    _write_metrics(release_dir / "reports" / "metrics.json", metrics_path, version, dataset_version)

    print("[backfill] regenerating manifest + checksums ...")
    _regenerate_version_files(release_dir, version, dataset_version)

    metrics = {}
    metrics_file = release_dir / "reports" / "metrics.json"
    if metrics_file.exists():
        metrics = (json.loads(metrics_file.read_text(encoding="utf-8")) or {}).get("metrics", {})

    report = {
        "release_dir": str(release_dir),
        "location_points": int(len(location_index)),
        "historical_context_rows": int(len(historical_context)),
        "village_metadata_rows": int(len(village_metadata)),
        "yield_scaler_mean": float(yield_scaler.mean_[0]),
        "yield_scaler_scale": float(yield_scaler.scale_[0]),
        "metrics_keys": sorted(metrics),
    }
    print(json.dumps(report, indent=2, default=str))
    print(f"[backfill] release package complete -> {release_dir}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-backfill-release",
        description="Reconstruct missing metadata/artifacts in a release package.",
    )
    parser.add_argument(
        "--release-dir", required=True,
        default=str(_REPO_ROOT / "releases" / "v2.0.0" / "cropfusion_release-v2.0.0"),
        help="cropfusion_release/ directory to backfill in place",
    )
    parser.add_argument(
        "--tabular",
        default=str(_REPO_ROOT / "Tabular_Datasets" / "data_season.csv"),
    )
    parser.add_argument(
        "--raster",
        default=str(_REPO_ROOT / "Tabular_Datasets" / "DK_Features_2024 (1).csv"),
    )
    parser.add_argument("--corpus", default=None, help="train corpus.json (exact yield scaler)")
    parser.add_argument("--metrics", default=None, help="train checkpoint.json (recorded metrics)")
    parser.add_argument("--grid-step", type=float, default=0.02)
    args = parser.parse_args(argv)

    backfill_release(
        release_dir=Path(args.release_dir),
        tabular_path=Path(args.tabular),
        raster_path=Path(args.raster),
        corpus_path=Path(args.corpus) if args.corpus else None,
        metrics_path=Path(args.metrics) if args.metrics else None,
        grid_step=args.grid_step,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
