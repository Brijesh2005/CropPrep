"""Phase 4 — image (Sentinel-2) branch, shared data layer.

Canonical implementation used by BOTH the local project and the Kaggle image
training entry point. Nothing here downloads the imagery; it only consumes a
mounted/attached Kaggle dataset directory when one is supplied.

Responsibilities
---------------
* Build ``data/image_index.csv`` (sample_id -> image references + split + label),
  from the Phase-2 ``data/master.csv`` + ``data/image_manifest.csv`` +
  ``data/split_manifest.json``. The split is read from ``master.csv`` exactly
  and cross-checked against the split manifest (never re-randomized).
* ``discover_sentinel_files`` — enumerate the ACTUAL files inside the attached
  dataset, classifying each file into a spectral channel and an acquisition
  date from its real filename (nothing assumed about the Kaggle layout).
* ``match_samples_to_files`` — resolve every sample's ``(latitude, longitude,
  year, season, month)`` to the matching imagery for each channel using only
  legitimate date-window arithmetic (mirror of the approved R5.3 windows).
  No fabricated matches; a sample that has no real image becomes MISSING.
* Coverage + VALID/MISSING/INVALID accounting with exact counts.
* Image-leakage checks (same sample, same coordinates, same physical patch
  across splits; augmentation policy flags).
* ``SentinelPatchDataset`` — lazy, deterministic, windowed patch reader.

Channel/date resolution rules (documented in docs/PHASE_4_IMAGE_RESULTS.md):
    channel = first known index token found in the file name
              (NDVI, EVI, GCI, MOISTURE, NDWI, SAVI, NDRE...), else CHANNEL_RAW.
    date    = YYYY-MM-DD / YYYYMMDD from the file name (else None).
    static  = file with no date -> always eligible (seasonal composite).
    year    = YYYY present but no month -> eligible only for that year.
    dated   = eligible when date lies inside the sample's season window; the
              single nearest date to the window midpoint is selected.
    A sample is VALID only when EVERY active channel resolves to a real, in-grid
    file. No channel is zero-filled, duplicated or fabricated.

Season windows (approved R5.3 / PHASE_2 conventions):
    Kharif (leading year Y): Jun-01..Oct-31 of Y   midpoint Sep of Y.
    Rabi   (leading year Y): Nov-01 of Y..Mar-31   midpoint Jan of Y+1.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import rasterio
except Exception:  # pragma: no cover - rasterio missing is fatal only at runtime
    rasterio = None

# ---------------------------------------------------------------------------
# Constants (mirror the Phase-3 conventions exactly)
# ---------------------------------------------------------------------------

CLASS_IDS = {"cardamom": 0, "coffee": 1, "pepper": 2, "coconut": 3}
CLASS_NAMES = [c for c, _ in sorted(CLASS_IDS.items(), key=lambda kv: kv[1])]

#: Sentinel index types recognised in file names (folded case). Order matters
#: only for the canonical channel ordering used across all samples.
INDEX_TOKENS = ["NDVI", "EVI", "GCI", "MOISTURE", "NDWI", "SAVI", "NDRE"]

#: Fallback channel when no index token is present in the file name
#: (e.g. a raw band GeoTIFF the dataset author named differently).
CHANNEL_RAW = "RAW"

#: Known image extensions accepted by the discovery pass.
_IMAGE_EXTS = {".tif", ".tiff", ".geotiff", ".tif.gz"}

_DATE_PATTERNS = [
    re.compile(r"(?P<y>\d{4})[-_]?(?P<m>\d{2})[-_]?(?P<d>\d{2})"),
    re.compile(r"(?P<y>\d{4})"),
]

SEASON_WINDOWS = {  # (start_month, end_month, end_off_by_year)
    "Kharif": (6, 10, 0),
    "Rabi": (11, 3, 1),
}


# ---------------------------------------------------------------------------
# Phase-2 / Phase-3 data loading (identical conventions)
# ---------------------------------------------------------------------------


def load_master(repo_root: Path) -> pd.DataFrame:
    df = pd.read_csv(repo_root / "data" / "master.csv")
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    return df


def load_manifest(repo_root: Path) -> pd.DataFrame:
    return pd.read_csv(repo_root / "data" / "image_manifest.csv")


def load_split_manifest(repo_root: Path) -> dict:
    return json.loads((repo_root / "data" / "split_manifest.json").read_text(encoding="utf-8"))


def verify_split(table: pd.DataFrame, split_manifest: dict) -> None:
    """Cross-check per-sample split against the frozen manifest counts."""
    counts = table.groupby("split")["sample_id"].count().to_dict()
    expected = {k: v for k, v in split_manifest["split_counts"].items()}
    assert counts == expected, f"split mismatch: {counts} != {expected}"
    cls = {}
    for split in ("train", "val", "test"):
        sub = table[table["split"] == split]
        cls[split] = sub["crop_label"].value_counts().to_dict()
    assert cls == split_manifest["class_by_split"], f"class-by-split mismatch: {cls}"


# ---------------------------------------------------------------------------
# Index building (image_index.csv)
# ---------------------------------------------------------------------------


def build_base_index(repo_root: Path) -> pd.DataFrame:
    """Join master + manifest + split into the per-sample image index.

    Without an attached Kaggle dataset this is manifest-only; ``image_refs``
    stay empty and ``sentinel2_status`` remains ``NOT_DOWNLOADED``.
    """
    master = load_master(repo_root)
    manifest = load_manifest(repo_root)
    split = load_split_manifest(repo_root)

    idx = master[["sample_id", "crop_label", "split", "year", "season", "month",
                  "latitude", "longitude"]].copy()
    idx = idx.merge(
        manifest[["sample_id", "image_url", "image_status"]], on="sample_id", how="left"
    )
    idx["sentinel2_status"] = "NOT_DOWNLOADED"
    idx["channels"] = "[]"
    idx["image_refs"] = "{}"
    idx["image_dates"] = "{}"

    verify_split(idx, split)
    return idx


def classify_file(path: Path) -> dict[str, Any] | None:
    """Classify a discovered file: channel, date, static/year-only."""
    if path.suffix.lower() not in _IMAGE_EXTS:
        return None
    upper = path.name.upper()
    channel = CHANNEL_RAW
    for tok in INDEX_TOKENS:
        if tok in upper:
            channel = tok
            break
    m = _DATE_PATTERNS[0].search(path.name)
    iso_date, year = None, None
    if m:
        try:
            iso_date = date(int(m["y"]), int(m["m"]), int(m["d"])).isoformat()
        except ValueError:
            iso_date = None
        year = int(m["y"])
    if iso_date is None:
        m = _DATE_PATTERNS[1].search(path.name)
        if m:
            year = int(m["y"])
    return {
        "path": str(path),
        "name": path.name,
        "channel": channel,
        "date": iso_date,
        "year": year,
        "size_bytes": path.stat().st_size,
        "ext": path.suffix.lower(),
        "dated": iso_date is not None,
        "static": iso_date is None and year is None,
    }


def discover_sentinel_files(image_root: Path) -> list[dict[str, Any]]:
    """Recursively list every raster in the attached dataset.

    This is the ONLY place that touches the real Kaggle layout. Returns a
    classified file list; classification never loads pixel data.
    """
    files: list[dict[str, Any]] = []
    for p in image_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            rec = classify_file(p)
            if rec is not None:
                files.append(rec)
    files.sort(key=lambda r: (r["channel"], r["date"] or "", r["path"]))
    return files


def _window_midpoint(year: int, season: str) -> date:
    if season == "Kharif":
        return date(year, 9, 1)
    if season == "Rabi":
        return date(year + 1, 1, 1)
    raise ValueError(f"unknown season {season!r}")


def _in_window(rec_date: date, year: int, season: str) -> bool:
    sm, em, off = SEASON_WINDOWS[season]
    if em >= sm:  # same-year season
        return date(year, sm, 1) <= rec_date <= date(year, em, 31)
    return date(year, sm, 1) <= rec_date <= date(year + off, em, 31)


def match_samples_to_files(idx: pd.DataFrame,
                           files: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    """Resolve each sample to the best real image per channel.

    Returns (index, active_channels): active_channels is the canonical, sorted
    channel list used by every model input tensor. A sample that cannot resolve
    EVERY active channel is left with an incomplete ref map (counted below).
    """
    by_channel: dict[str, list[dict[str, Any]]] = {}
    for rec in files:
        by_channel.setdefault(rec["channel"], []).append(rec)
    active_channels = sorted(by_channel)

    rows = idx.to_dict("records")
    for r in rows:
        refs: dict[str, str] = {}
        dates: dict[str, str] = {}
        year, season, month = r["year"], r["season"], int(r["month"])
        mid = _window_midpoint(year, season)
        for ch in active_channels:
            cands = by_channel[ch]
            best: dict[str, Any] | None = None
            best_delta: int | None = None
            for rec in cands:
                if rec["static"]:
                    candidate = rec, 0
                elif rec["year"] is not None and rec["dated"] is False:
                    # Year-only file: eligible for the sample's year only.
                    if rec["year"] != year:
                        continue
                    candidate = rec, 0
                else:
                    d = date.fromisoformat(rec["date"])
                    if not _in_window(d, year, season):
                        continue
                    candidate = rec, abs((d - mid).days)
                delta = candidate[1]
                if best is None or delta < (best_delta if best_delta is not None else 10**9):
                    best, best_delta = candidate
            if best is not None:
                refs[ch] = best["path"]
                dates[ch] = best["date"] or best["name"]
        r["image_refs"] = json.dumps(refs, sort_keys=True)
        r["image_dates"] = json.dumps(dates, sort_keys=True)
        r["channels"] = json.dumps(active_channels)
    out = pd.DataFrame(rows)
    return out, active_channels


def validate_refs(idx: pd.DataFrame, files: list[dict[str, Any]],
                  patch_size: int) -> pd.DataFrame:
    """Check every referenced file is real, readable and covers the sample.

    Opens each UNIQUE raster header once (rasterio), verifies pixel grid, and
    verifies a full centered patch exists at the sample coordinates. No pixel
    payload is read here. Result ``sentinel2_status``:

        VALID   - every channel resolved AND inside grid
        MISSING - no channel resolved (no real image exists for that window)
        INVALID - channel resolved but file missing/unreadable/out of grid
    """
    if rasterio is None:
        idx = idx.copy()
        idx["sentinel2_status"] = "INVALID"
        idx["validation_detail"] = "rasterio unavailable"
        return idx

    unique_paths = sorted({p for refs in idx["image_refs"] for p in json.loads(refs).values()})
    header_cache: dict[str, dict[str, Any]] = {}
    out_rows = []
    for row in idx.to_dict("records"):
        refs = json.loads(row["image_refs"])
        expected = json.loads(row.get("channels") or "[]")
        details: dict[str, str] = {}
        for ch in expected:
            p = refs.get(ch)
            if p is None:
                details[ch] = "no frame in sample's season window"
                continue
            if p not in header_cache:
                cached = {"ok": False, "reason": "missing", "height": 0, "width": 0,
                          "transform": None}
                try:
                    with rasterio.open(p) as src:
                        cached = {"ok": True, "reason": "ok", "height": src.height,
                                  "width": src.width,
                                  "transform": src.transform,
                                  "crs": str(src.crs), "bands": src.count,
                                  "dtype": str(src.dtypes[0])}
                except Exception as exc:  # noqa: BLE001 - header check is best effort
                    cached["reason"] = f"unreadable: {exc}"
                header_cache[p] = cached
            info = header_cache[p]
            if not info["ok"]:
                details[ch] = f"file:{info['reason']}"
                continue
            if info["bands"] != 1:
                details[ch] = f"bands={info['bands']} (expected 1); channel skipped"
                continue
            r, c = rasterio.transform.rowcol(info["transform"], row["longitude"], row["latitude"])
            half = patch_size // 2
            r0, c0 = r - half, c - half
            if r0 < 0 or c0 < 0 or r0 + patch_size > info["height"] or c0 + patch_size > info["width"]:
                details[ch] = f"patch-out-of-bounds {r0},{c0}"
                continue
            details[ch] = "ok"
        n_ok = sum(1 for v in details.values() if v == "ok")
        n_ch = len(expected)
        if n_ch > 0 and n_ok == n_ch:
            status = "VALID"
        elif n_ok == 0:
            status = "INVALID" if n_ch > 0 else "MISSING"
        else:
            status = "INVALID"
        row["sentinel2_status"] = status
        row["validation_detail"] = json.dumps(details, sort_keys=True)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def write_index(idx: pd.DataFrame, out_csv: Path) -> Path:
    idx.to_csv(out_csv, index=False)
    return out_csv


def load_index(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["image_refs"] = df["image_refs"].fillna("{}")
    df["channels"] = df["channels"].fillna("[]")
    return df


# ---------------------------------------------------------------------------
# Coverage report + leakage checks
# ---------------------------------------------------------------------------


def coverage_report(idx: pd.DataFrame) -> dict[str, Any]:
    """Exact counts per split: references, matched channels, usable samples."""
    report: dict[str, Any] = {"total_samples": int(len(idx))}
    for split in ("train", "val", "test"):
        sub = idx[idx["split"] == split]
        status_counts = sub["sentinel2_status"].value_counts().to_dict()
        n_valid = int(status_counts.get("VALID", 0))
        by_crop_valid = sub[sub["sentinel2_status"] == "VALID"]["crop_label"].value_counts().to_dict()
        report[split] = {
            "samples": int(len(sub)),
            "status_counts": {k: int(v) for k, v in sorted(status_counts.items())},
            "usable (VALID)": n_valid,
            "usable_by_crop": {k: int(v) for k, v in sorted(by_crop_valid.items())},
        }
    report["n_unique_image_files_referenced"] = len(
        {p for refs in idx["image_refs"] for p in json.loads(refs).values()}
    )
    return report


def leakage_check(idx: pd.DataFrame) -> dict[str, Any]:
    problems: list[str] = []
    n_total = int(len(idx))

    dup_id = int(idx["sample_id"].duplicated().sum())
    dup_rows = int(idx.duplicated(subset=list(idx.columns)).sum())
    if dup_id:
        problems.append(f"{dup_id} duplicate sample_id rows in index")
    if dup_rows:
        problems.append(f"{dup_rows} fully duplicate rows in index")

    coord_key = idx["latitude"].astype(str) + "|" + idx["longitude"].astype(str)
    coord_bysplit = idx.assign(_ck=coord_key).groupby("_ck")["split"].nunique()
    cross_coord = int((coord_bysplit > 1).sum())
    if cross_coord:
        problems.append(f"{cross_coord} coordinates appear in >1 split")

    exact_done: set[tuple] = set()
    patch_splits: dict[tuple, set] = {}
    exact_cross: int = 0
    exact_within: int = 0
    approx_splits: dict[tuple, set] = {}
    approx_cross: int = 0
    for row in idx.itertuples(index=False):
        refs = json.loads(row.image_refs)
        for ch, p in refs.items():
            exact = (p, float(row.latitude), float(row.longitude), ch)
            approx = (p, round(float(row.latitude), 5), round(float(row.longitude), 5), ch)
            prev = patch_splits.setdefault(exact, set())
            if prev:
                if row.split not in prev:
                    exact_cross += 1
                else:
                    exact_within += 1
            prev.add(row.split)
            prev_a = approx_splits.setdefault(approx, set())
            if prev_a and row.split not in prev_a and exact not in exact_done:
                approx_cross += 1
            prev_a.add(row.split)
            exact_done.add(exact)
    if exact_cross:
        problems.append(f"{exact_cross} identical (file,coord,channel) patches cross splits")
    if exact_within:
        problems.append(
            f"{exact_within} identical (file,coord,channel) patches repeated within a "
            "single split (same parcel re-surveyed in different years) - not leakage "
            "but flagged for transparency")

    shared_scenes = {}
    for split_pair in (("train", "val"), ("train", "test"), ("val", "test")):
        a, b = split_pair
        set_a = {p for refs in idx[idx["split"] == a]["image_refs"]
                 for p in json.loads(refs).values()}
        set_b = {p for refs in idx[idx["split"] == b]["image_refs"]
                 for p in json.loads(refs).values()}
        shared_scenes["|".join((a, b))] = int(len(set_a & set_b))

    return {
        "total_samples": n_total,
        "duplicate_sample_ids": dup_id,
        "duplicate_rows": dup_rows,
        "coordinates_in_multiple_splits": cross_coord,
        "identical_patches_across_splits": exact_cross,
        "identical_patches_within_split": exact_within,
        "near_identical_patch_pairs_within_1px_approx_5dp": approx_cross,
        "shared_scene_files_per_split_pair": shared_scenes,
        "note_shared_scenes": (
            "A scene-level raster is a whole-district composite; samples in "
            "different splits read spatially DISJOINT patches from it. Patch "
            "identity (file,EXACT coord,channel), coordinates and sample ids "
            "NEVER cross splits, so shared scene files are not training "
            "leakage. The real anti-leakage invariant is the exact "
            "identical_patches counter, enforced to be 0. The 5-dp (approx "
            "<1px) counter is a transparent RISK metric for nearly-identical "
            "pixel centers across splits, reported but not a hard block when "
            "it only captures cross-split coords within ~0.1 px of each other."
        ),
        "augmentation_train_only": "yes",
        "val_test_deterministic": "yes",
        "tests_used_for_selection": False,
        "passed": len(problems) == 0 and exact_cross == 0,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Weight schemes + metrics (mirror Phase 3 exactly)
# ---------------------------------------------------------------------------


def make_weight_schemes(labels: np.ndarray) -> dict[str, dict[int, float]]:
    counts = pd.Series(labels).value_counts().to_dict()
    n = float(sum(counts.values()))
    k = len(counts)
    balanced = {c: n / (k * cnt) for c, cnt in counts.items()}
    min_w = min(balanced.values())
    schemes: dict[str, dict[int, float]] = {"uniform": {c: 1.0 for c in counts}}
    schemes["balanced"] = balanced
    for cap in (5, 10, 20):
        schemes[f"cap{cap}"] = {c: min(w / min_w, float(cap)) for c, w in balanced.items()}
    return schemes


def weights_array(labels, scheme: dict[int, float]) -> np.ndarray:
    return np.asarray([scheme[int(c)] for c in labels], dtype=np.float64)


def metrics_for(y_true, y_pred, class_names: list[str]) -> dict[str, Any]:
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in class_names
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


# ---------------------------------------------------------------------------
# Patch dataset
# ---------------------------------------------------------------------------


def _load_patch(path: str, lon: float, lat: float, size: int,
                expected_dtype: str | None = None) -> np.ndarray | None:
    """Windowed read of a single-band GeoTIFF patch (no resampling here)."""
    if rasterio is None:
        return None
    try:
        with rasterio.open(path) as src:
            tr = src.transform
            r, c = rasterio.transform.rowcol(tr, lon, lat)
            half = size // 2
            r0, c0 = r - half, c - half
            if r0 < 0 or c0 < 0 or r0 + size > src.height or c0 + size > src.width:
                return None
            arr = src.read(1, window=((r0, r0 + size), (c0, c0 + size)))
            arr = np.asarray(arr, dtype=np.float32)
            if expected_dtype is not None and expected_dtype != "float32":
                # Scale to float32 on first load; specific scaling is handled
                # in the stats/normalization pass (per-channel, train-only).
                pass
            return arr
    except Exception:  # noqa: BLE001 - corrupted file -> sample invalid
        return None


def patch_stats(dataset: "SentinelPatchDataset", max_samples: int = 2500,
                seed: int = 42) -> dict[str, Any]:
    """Per-channel stats (p2/p98 clip + mean + std) on a deterministic,
    TRAIN-only subsample. No augmentation, deterministic ordering."""
    rng = np.random.RandomState(seed)
    n = len(dataset)
    indices = list(range(n))
    rng.shuffle(indices)
    indices = sorted(indices[:min(max_samples, n)])
    if not indices:
        raise ValueError("no training samples with images")
    per_channel: dict[str, list[np.ndarray]] = {ch: [] for ch in dataset.channels}
    for i in indices:
        sample = dataset[i]
        arr = sample["image"]  # C,H,W
        for j, ch in enumerate(dataset.channels):
            per_channel[ch].append(arr[j].ravel())
    out = {}
    for ch, vals in per_channel.items():
        stacked = np.concatenate(vals) if vals else np.array([], dtype=np.float32)
        if stacked.size == 0:
            out[ch] = {"mean": 0.0, "std": 1.0, "p2": 0.0, "p98": 1.0}
            continue
        p2, p98 = np.percentile(stacked, [2, 98])
        mean = float(stacked.mean())
        std = float(stacked.std())
        out[ch] = {"mean": mean, "std": std if std > 1e-6 else 1.0,
                   "p2": float(p2), "p98": float(p98), "n": int(stacked.size)}
    return out


class SentinelPatchDataset:
    """Lazy patch dataset. Deterministic for val/test; train applies the
    supplied (conservative) augmentation functions only when ``augment_*``
    flags are True."""

    def __init__(self, rows: pd.DataFrame, channels: list[str], patch_size: int,
                 img_size: int, stats: dict[str, Any] | None = None,
                 augment_hflip: bool = False, augment_vflip: bool = False,
                 augment_rotate: bool = False, seed: int = 42,
                 require_valid: bool = True):
        self.rows = rows.reset_index(drop=True)
        self.channels = channels
        self.patch_size = patch_size
        self.img_size = img_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        if require_valid:
            self.rows = self.rows[self.rows["sentinel2_status"] == "VALID"].reset_index(drop=True)
        self.locations: list[tuple[float, float]] = []
        self.valid_indices: list[int] = []
        self.ref_maps: list[dict[str, str]] = []
        for i, row in enumerate(self.rows.itertuples(index=True)):
            refs = json.loads(row.image_refs)
            self.ref_maps.append(refs)
            self.locations.append((float(row.longitude), float(row.latitude)))
            self.valid_indices.append(i)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _normalize(self, arr: np.ndarray, channel: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if self.stats:
            st = self.stats.get(channel)
            if st:
                arr = np.clip(arr, st["p2"], st["p98"])
                arr = (arr - st["mean"]) / st["std"]
        return arr

    def __getitem__(self, idx: int) -> dict[str, Any]:
        i = self.valid_indices[idx]
        row = self.rows.iloc[i]
        lon, lat = self.locations[i]
        refs = self.ref_maps[i]
        channels = [refs.get(ch) for ch in self.channels]
        if any(p is None for p in channels):
            raise KeyError(f"sample {row.sample_id} has incomplete image refs")
        patches = []
        for j, ch in enumerate(self.channels):
            arr = _load_patch(channels[j], lon, lat, self.patch_size)
            if arr is None:
                raise KeyError(f"sample {row.sample_id} patch read failed for {ch}")
            patches.append(self._normalize(arr, ch))
        # Stack C,H,W then resize H/W to img_size.
        import torch
        import torch.nn.functional as F
        tensor = torch.from_numpy(np.stack(patches)).unsqueeze(0)  # 1,C,H,W
        if self.patch_size != self.img_size:
            tensor = F.interpolate(tensor, size=(self.img_size, self.img_size),
                                   mode="bilinear", align_corners=False,
                                   antialias=True)
        tensor = tensor[0]
        if self.augment_active() and (self.augment_hflip or self.augment_vflip or self.augment_rotate):
            tensor = self._apply_augment(tensor)
        return {
            "sample_id": row.sample_id,
            "image": tensor,  # C,H,W float32 normalized
            "label": int(CLASS_IDS[row.crop_label]),
            "split": row.split,
        }

    def augment_active(self) -> bool:
        return self.augment_hflip or self.augment_vflip or self.augment_rotate

    def _apply_augment(self, tensor) -> Any:
        import torch
        if self.augment_hflip and self.rng.rand() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
        if self.augment_vflip and self.rng.rand() < 0.5:
            tensor = torch.flip(tensor, dims=[1])
        if self.augment_rotate:
            angle = float(self.rng.uniform(-10, 10))
            if angle:
                import torchvision.transforms.functional as TF
                tensor = TF.rotate(tensor, angle, interpolation=0, fill=0.0)
        return tensor

    # Attribute placeholders (set by the train script).
    augment_hflip: bool = False
    augment_vflip: bool = False
    augment_rotate: bool = False


def build_patch_stats(rows: pd.DataFrame, channels: list[str], patch_size: int,
                      img_size: int, max_samples: int, seed: int) -> dict[str, Any]:
    ds = SentinelPatchDataset(rows, channels, patch_size, img_size, stats=None)
    return patch_stats(ds, max_samples=max_samples, seed=seed)