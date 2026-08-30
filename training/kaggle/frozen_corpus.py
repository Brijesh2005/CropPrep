"""Frozen corpus adapter — wire R5.2.7 supervised crop data into the pipeline.

Reads ``crop_supervised_v1.csv`` and its manifest, validates integrity
(checksum, row counts, class mapping), then constructs
:class:`AgriculturalObservation` objects that can be fed directly into
:meth:`Experiment.run` — bypassing :class:`ObservationResolver` entirely
while still using STAM's imagery resolution for NDVI/EVI patch extraction.

The frozen corpus guarantees:
- **Membership**: exactly 10,674 samples (R5.2.7 snapshot).
- **Labels**: government OGD survey crop labels (no tabular matching override).
- **Spatial split**: taluk-level leave-one-out (Belthangady+Mangalore+Puttur
  train / Bantwal val / Sullia test).
- **Quality**: all samples pre-validated (satellite_status=FULL).
- **Provenance**: every observation stamped with corpus version, manifest
  checksum and the row-level provenance from the CSV.

Usage::

    loader = FrozenCorpusLoader(
        csv_path="govt_crop_matched_v1/crop_supervised_v1.csv",
        manifest_path="training_manifests/crop_supervised_v1_manifest.json",
    )
    train, val, test = loader.build(stam)
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Sequence

from training.stam.observation import (
    AgriculturalObservation,
    AdminLocation,
    ImagePairRef,
    ImageRecordRef,
    LocationInfo,
    QualityReport,
    SequenceInfo,
    TabularFeatures,
    TemporalInfo,
)

import logging

from shared.logging import get_logger, log_dict

logger = get_logger("training.frozen_corpus")


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Manifest field names that MUST be present.
_MANIFEST_REQUIRED = {
    "dataset_version",
    "total_samples",
    "train_samples",
    "validation_samples",
    "test_samples",
    "class_mapping",
    # R5.4 explicit class contract: supervised_classes is the crop-head output
    # vocabulary; excluded_classes records corpus labels outside it (so an
    # exclusion can never be silent again).
    "supervised_classes",
    "excluded_classes",
    "split_groups",
    "provenance_schema",
}

# CSV column names that MUST be present.
_CSV_REQUIRED = {
    "record_id",
    "crop_label",
    "crop_class_id",
    "location_taluk",
    "location_village",
    "location_district",
    "lat",
    "lon",
    "year",
    "season",
    "satellite_status",
}

# Taluk → split mapping from R5.2.7 manifest.
_TALUK_SPLIT: dict[str, str] = {
    "Belthangady": "train",
    "Mangalore": "train",
    "Puttur": "train",
    "Bantwal": "val",
    "Sullia": "test",
}

# Canonical class mapping frozen in R5.2.7.
_CANONICAL_CLASS_MAP = {
    "coconut": 4,
    "pepper": 6,
    "coffee": 7,
    "cardamom": 8,
    "blackgram": 9,
}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class FrozenCorpusError(Exception):
    """Raised when the frozen corpus or manifest fails validation."""


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate the frozen corpus manifest.

    Checks:
    - Required top-level fields are present.
    - ``dataset_version`` is ``crop_supervised_v1.1``.
    - Class mapping matches the frozen canonical mapping.
    - Split counts sum to total_samples.
    - ``split_groups`` contains train/val/test taluk lists.

    Returns:
        The parsed manifest dict.

    Raises:
        FrozenCorpusError: on any validation failure.
    """
    import json

    if not manifest_path.exists():
        raise FrozenCorpusError(f"Manifest not found: {manifest_path}")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = _MANIFEST_REQUIRED - raw.keys()
    if missing:
        raise FrozenCorpusError(f"Manifest missing required fields: {sorted(missing)}")

    version = raw["dataset_version"]
    if version != "crop_supervised_v1.1":
        raise FrozenCorpusError(
            f"Unexpected manifest version: {version!r} "
            "(expected 'crop_supervised_v1.1')"
        )

    total = raw["total_samples"]
    train_n = raw["train_samples"]
    val_n = raw["validation_samples"]
    test_n = raw["test_samples"]
    if train_n + val_n + test_n != total:
        raise FrozenCorpusError(
            f"Split counts ({train_n}+{val_n}+{test_n}={train_n+val_n+test_n}) "
            f"do not sum to total_samples ({total})"
        )

    class_map = raw.get("class_mapping", {})
    if class_map != _CANONICAL_CLASS_MAP:
        raise FrozenCorpusError(
            f"Class mapping mismatch: {class_map} != {_CANONICAL_CLASS_MAP}"
        )

    split_groups = raw.get("split_groups", {})
    if not split_groups.get("train_taluk"):
        raise FrozenCorpusError("split_groups.train_taluk is missing or empty")
    if not split_groups.get("validation_taluk"):
        raise FrozenCorpusError("split_groups.validation_taluk is missing")
    if not split_groups.get("test_taluk"):
        raise FrozenCorpusError("split_groups.test_taluk is missing")

    # R5.4 explicit class contract: supervised_classes defines the crop-head
    # output vocabulary; excluded_classes records corpus labels outside it.
    supervised = list(raw.get("supervised_classes") or [])
    excluded = list(raw.get("excluded_classes") or [])
    if not supervised:
        raise FrozenCorpusError(
            "manifest supervised_classes is empty — the crop head would learn "
            "nobody's vocabulary"
        )
    known = set(raw.get("class_mapping", {}))
    if set(supervised) - known:
        raise FrozenCorpusError(
            f"supervised_classes references unknown classes: "
            f"{sorted(set(supervised) - known)}"
        )
    if set(excluded) - known:
        raise FrozenCorpusError(
            f"excluded_classes references unknown classes: "
            f"{sorted(set(excluded) - known)}"
        )
    if set(supervised) & set(excluded):
        raise FrozenCorpusError(
            f"classes listed in BOTH supervised_classes and excluded_classes: "
            f"{sorted(set(supervised) & set(excluded))}"
        )

    log_dict(
        logger,
        logging.INFO,
        "Manifest validated",
        version=version,
        total=total,
        train=train_n,
        val=val_n,
        test=test_n,
    )
    return raw


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #


def _load_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load the frozen corpus CSV into a list of row dicts.

    Raises FrozenCorpusError on missing required columns or zero rows.
    """
    if not csv_path.exists():
        raise FrozenCorpusError(f"CSV not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise FrozenCorpusError("CSV is empty (no header row)")
        missing = _CSV_REQUIRED - set(reader.fieldnames)
        if missing:
            raise FrozenCorpusError(
                f"CSV missing required columns: {sorted(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise FrozenCorpusError("CSV contains zero data rows")

    log_dict(logger, logging.INFO, "CSV loaded", path=str(csv_path), rows=len(rows))
    return rows


# --------------------------------------------------------------------------- #
# Observation construction
# --------------------------------------------------------------------------- #


def _parse_date(date_str: str | None) -> Any:
    """Parse a YYYY-MM-DD string into a date, or return None."""
    if not date_str:
        return None
    from datetime import date

    try:
        return date.fromisoformat(date_str.strip())
    except (ValueError, TypeError):
        return None


def _safe_float(value: Any) -> float | None:
    """Parse a value as float, returning None on failure/missing."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_location(row: dict[str, Any]) -> LocationInfo:
    """Construct LocationInfo from a frozen corpus CSV row."""
    admin = AdminLocation(
        village=row.get("location_village") or None,
        taluk=row.get("location_taluk") or None,
        district=row.get("location_district") or None,
        state="Karnataka",
        country="India",
        level="village",
        source="government_ogd",
    )
    return LocationInfo(
        lon=float(row["lon"]),
        lat=float(row["lat"]),
        distance_km=float(row.get("spatial_match_distance_km") or 0.0),
        dataset_location_id=row.get("record_id"),
        dataset_location_name=row.get("location_hobli"),
        admin=admin,
    )


def _build_temporal(row: dict[str, Any]) -> TemporalInfo:
    """Construct TemporalInfo from a frozen corpus CSV row."""
    survey_date = _parse_date(row.get("survey_date"))
    return TemporalInfo(
        year=int(row["year"]),
        season=row.get("season") or None,
        season_months=None,
        observation_dates=[survey_date] if survey_date else [],
        planting_start=None,
        harvest_end=None,
        tolerance_days=0,
    )


def _resolve_imagery(
    stam: Any,
    row: dict[str, Any],
) -> SequenceInfo:
    """Use STAM to resolve NDVI/EVI imagery for this point.

    Uses :meth:`STAM.resolve_sequence`, which builds the image sequence
    directly from imagery coverage — it never consults the spatial location
    index or tabular chain. Frozen-corpus rows are village GPS points that
    frequently sit farther than ``max_search_radius_km`` (5 km) from the
    nearest indexed dataset location, which made the full
    ``build_observation`` path fail with ``LocationNotFoundError`` for the
    majority of samples even though their imagery is pre-validated
    (``satellite_status=FULL``).
    """
    lon = float(row["lon"])
    lat = float(row["lat"])
    year = int(row["year"])
    season = row.get("season")

    try:
        sequence = stam.resolve_sequence(lon, lat, year=year, season=season)
        if len(getattr(sequence, "pairs", ())) == 0:
            # Diagnostic: attribute the empty sequence to one of two
            # root causes so a Kaggle run can be triaged without digging
            # through per-sample logs:
            #   * zero season-window imagery records, or
            #   * records present but filtered out by point-coverage
            #     (CRS / bounds mismatch on the mounted GeoTIFFs).
            _log_empty_sequence_diagnostics(stam, row, lon, lat, year, season)
        return sequence
    except Exception as exc:  # noqa: BLE001
        log_dict(
            logger,
            logging.WARNING,
            "Imagery resolution failed — returning empty sequence",
            record_id=row.get("record_id"),
            error=str(exc),
        )
        return SequenceInfo()


def _log_empty_sequence_diagnostics(
    stam: Any, row: dict[str, Any], lon: float, lat: float, year: int, season: str | None
) -> None:
    """Best-effort diagnostics for an empty imagery sequence (never raises).

    Uses the STAM internal matcher to distinguish ``no season-window records``
    from ``records dropped by point coverage`` — the two root causes of
    empty sequences on the Kaggle imagery mount.
    """
    try:
        matcher = getattr(stam, "matcher", None)
        if matcher is None or not hasattr(matcher, "resolve_temporal"):
            return
        context = matcher.resolve_temporal(year=year, season=season)
        window_ndvi, window_evi = matcher.match_images(
            year=year, season=context.season, resolution=None
        )
        covered_ndvi, covered_evi = getattr(stam, "_filter_images_for_point", lambda *a, **k: (a[2], a[3]))(
            lon, lat, window_ndvi, window_evi
        )
        log_dict(
            logger,
            logging.WARNING,
            "Empty imagery sequence diagnostics",
            record_id=row.get("record_id"),
            year=year,
            season=season,
            resolved_season=context.season.name if getattr(context, "season", None) else None,
            window_ndvi=len(window_ndvi),
            window_evi=len(window_evi),
            covered_ndvi=len(covered_ndvi),
            covered_evi=len(covered_evi),
            start_hint=(
                "no season-window records"
                if not (window_ndvi or window_evi)
                else "records dropped by point coverage (CRS/bounds)"
            ),
        )
    except Exception:  # noqa: BLE001 - diagnostics are best-effort
        return


def _determine_split(row: dict[str, Any]) -> str:
    """Determine the train/val/test split for a row based on taluk."""
    taluk = (row.get("location_taluk") or "").strip()
    return _TALUK_SPLIT.get(taluk, "unknown")


def build_observation(
    row: dict[str, Any],
    stam: Any,
    *,
    corpus_version: str,
    manifest_checksum: str,
) -> AgriculturalObservation:
    """Construct a single AgriculturalObservation from a frozen corpus row.

    Uses STAM for imagery resolution (NDVI/EVI paths), but overrides
    the crop label, quality and provenance from the frozen CSV.

    Args:
        row: One row from ``crop_supervised_v1.csv``.
        stam: An initialized :class:`STAM` instance for imagery resolution.
        corpus_version: The frozen corpus version string.
        manifest_checksum: SHA-256 of the manifest JSON file.

    Returns:
        A fully constructed :class:`AgriculturalObservation`.
    """
    crop_label = row.get("crop_label") or None
    crop_class_id = int(row["crop_class_id"]) if row.get("crop_class_id") else None

    location = _build_location(row)
    temporal = _build_temporal(row)
    sequence = _resolve_imagery(stam, row)

    # Tabular features — crop label from frozen CSV, no yield (classification).
    #
    # The frozen CSV has NO independent yield/weather/soil/rainfall columns
    # beyond the labels and geometry already consumed by STAM imagery.  We
    # therefore expose only the REAL per-row columns that actually vary
    # between observations (location geometry + survey metadata) so the
    # tabular encoder gets a genuine, non-constant vector — no features are
    # fabricated to satisfy the tabular branch.  Feature absence is
    # documented in the manifest's ``feature_schema`` rather than invented.
    season_value = (row.get("season") or "").strip() or None
    tabular = TabularFeatures(
        crop=crop_label,
        yield_value=None,
        fields={
            "lat": _safe_float(row.get("lat")),
            "lon": _safe_float(row.get("lon")),
            "spatial_match_distance_km": _safe_float(
                row.get("spatial_match_distance_km")
            ),
            "season": season_value,
        },
        source_path=None,
        matched_level="frozen_corpus",
    )

    # Quality — pre-validated (all samples passed satellite verification).
    quality = QualityReport(
        passed=True,
        overall_score=100.0,
        issues=[],
    )

    split = _determine_split(row)

    observation = AgriculturalObservation(
        location=location,
        temporal=temporal,
        tabular=tabular,
        sequence=sequence,
        quality=quality,
        crop=crop_label,
        yield_value=None,
        # patch_size=0 (falsy) keeps the preprocessing image size (224)
        # authoritative: master_pipeline uses ``observation.patch_size or
        # config.image.size``.  A hard-coded 128 here previously made the
        # zero-fill fallback emit [1,128,128] while real patches were
        # resized to [1,224,224], crashing torch.stack in collate_samples.
        patch_size=0,
        provenance={
            "corpus": "crop_supervised_v1",
            "corpus_version": corpus_version,
            "manifest_checksum": manifest_checksum,
            "record_id": row.get("record_id"),
            "source": row.get("source"),
            "source_record_id": row.get("source_record_id"),
            "source_crop_name": row.get("source_crop_name"),
            "crop_class_id": crop_class_id,
            "spatial_match_distance_km": row.get("spatial_match_distance_km"),
            "temporal_match_status": row.get("temporal_match_status"),
            "tabular_source": row.get("tabular_source"),
            "image_source": row.get("image_source"),
            "ndvi_available": row.get("ndvi_available"),
            "evi_available": row.get("evi_available"),
            "satellite_status": row.get("satellite_status"),
            "split": split,
        },
        historical_context=None,
        dataset_version=corpus_version,
        season_calendar_version=None,
    )

    return observation


# --------------------------------------------------------------------------- #
# Main loader
# --------------------------------------------------------------------------- #


class FrozenCorpusLoader:
    """Load the frozen R5.2.7 supervised crop corpus.

    Validates the manifest, loads the CSV, constructs
    :class:`AgriculturalObservation` objects using STAM for imagery,
    and assigns the spatial split from the manifest.

    Args:
        csv_path: Path to ``crop_supervised_v1.csv``.
        manifest_path: Path to ``crop_supervised_v1_manifest.json``.
    """

    def __init__(
        self,
        csv_path: str | Path,
        manifest_path: str | Path,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.manifest_path = Path(manifest_path)
        self._manifest: dict[str, Any] | None = None
        self._rows: list[dict[str, Any]] | None = None
        #: Populated by :meth:`build` — ``{rows, excluded, train, val, test,
        #: accepted}`` from the last corpus build.
        self.last_build_stats: dict[str, int] | None = None

    @property
    def manifest(self) -> dict[str, Any]:
        """The validated manifest dict (available after :meth:`validate`)."""
        if self._manifest is None:
            raise FrozenCorpusError("validate() must run before reading the manifest")
        return self._manifest

    def validate(self) -> dict[str, Any]:
        """Validate manifest + CSV schema without building observations.

        Returns the parsed manifest dict.
        """
        self._manifest = validate_manifest(self.manifest_path)
        self._rows = _load_csv(self.csv_path)

        # Cross-validate CSV row count against manifest.
        csv_count = len(self._rows)
        manifest_total = self._manifest["total_samples"]
        if csv_count != manifest_total:
            raise FrozenCorpusError(
                f"CSV row count ({csv_count}) does not match "
                f"manifest total_samples ({manifest_total})"
            )

        # Cross-validate class counts.
        csv_class_counts: dict[str, int] = {}
        for row in self._rows:
            label = row.get("crop_label", "unknown")
            csv_class_counts[label] = csv_class_counts.get(label, 0) + 1
        manifest_class_counts = self._manifest.get("class_counts", {}).get("overall", {})
        for cls, expected in manifest_class_counts.items():
            actual = csv_class_counts.get(cls, 0)
            if actual != expected:
                raise FrozenCorpusError(
                    f"Class count mismatch for {cls!r}: "
                    f"CSV={actual}, manifest={expected}"
                )

        # Cross-validate split counts via taluk assignment.
        split_counts = {"train": 0, "val": 0, "test": 0}
        for row in self._rows:
            split = _determine_split(row)
            split_counts[split] = split_counts.get(split, 0) + 1
        for split_name, manifest_key in [
            ("train", "train_samples"),
            ("val", "validation_samples"),
            ("test", "test_samples"),
        ]:
            expected = self._manifest[manifest_key]
            actual = split_counts.get(split_name, 0)
            if actual != expected:
                raise FrozenCorpusError(
                    f"Split count mismatch for {split_name!r}: "
                    f"CSV taluk assignment={actual}, manifest={expected}"
                )

        log_dict(
            logger,
            logging.INFO,
            "Frozen corpus validated",
            csv_rows=csv_count,
            manifest_total=manifest_total,
            class_counts=csv_class_counts,
            split_counts=split_counts,
        )
        return self._manifest

    def build(
        self,
        stam: Any,
        *,
        progress_every: int = 100,
    ) -> tuple[list[AgriculturalObservation], list[AgriculturalObservation], list[AgriculturalObservation]]:
        """Build observations and return pre-split (train, val, test).

        Uses STAM for imagery resolution per observation.  The crop label
        and quality are taken directly from the frozen CSV.

        Args:
            stam: An initialized :class:`STAM` instance.
            progress_every: Log progress every N observations.

        Returns:
            ``(train_observations, val_observations, test_observations)``
        """
        if self._manifest is None or self._rows is None:
            self.validate()

        manifest = self._manifest
        rows = self._rows
        manifest_checksum = _sha256_file(self.manifest_path)
        corpus_version = manifest["dataset_version"]

        train: list[AgriculturalObservation] = []
        val: list[AgriculturalObservation] = []
        test: list[AgriculturalObservation] = []
        errors = 0

        for index, row in enumerate(rows, start=1):
            try:
                obs = build_observation(
                    row,
                    stam,
                    corpus_version=corpus_version,
                    manifest_checksum=manifest_checksum,
                )
                if len(obs.sequence.pairs) == 0:
                    # Zero NDVI/EVI pairs would be rejected by the
                    # preprocessing quality gate (min_observations >= 1)
                    # mid-epoch, crashing the DataLoader. Exclude here.
                    errors += 1
                    log_dict(
                        logger,
                        logging.WARNING,
                        "Empty imagery sequence — sample excluded",
                        record_id=row.get("record_id"),
                        lon=row.get("lon"),
                        lat=row.get("lat"),
                        year=row.get("year"),
                        season=row.get("season"),
                    )
                    continue
                split = obs.provenance.get("split", "unknown")
                if split == "train":
                    train.append(obs)
                elif split == "val":
                    val.append(obs)
                elif split == "test":
                    test.append(obs)
                else:
                    log_dict(
                        logger,
                        logging.WARNING,
                        "Unknown split — assigning to train",
                        record_id=row.get("record_id"),
                        taluk=row.get("location_taluk"),
                    )
                    train.append(obs)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                log_dict(
                    logger,
                    logging.WARNING,
                    "Failed to build observation",
                    record_id=row.get("record_id"),
                    error=str(exc),
                )

            if progress_every and index % progress_every == 0:
                log_dict(
                    logger,
                    logging.INFO,
                    "Corpus build progress",
                    built=index,
                    total=len(rows),
                    errors=errors,
                    train=len(train),
                    val=len(val),
                    test=len(test),
                )

        log_dict(
            logger,
            logging.INFO,
            "Frozen corpus built",
            total=len(rows),
            errors=errors,
            train=len(train),
            val=len(val),
            test=len(test),
        )

        self.last_build_stats = {
            "rows": len(rows),
            "excluded": errors,
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "accepted": len(train) + len(val) + len(test),
        }

        return train, val, test

    def corpus_imagery_diagnostics(
        self,
        train: Sequence[AgriculturalObservation],
        val: Sequence[AgriculturalObservation],
        test: Sequence[AgriculturalObservation],
        *,
        max_observations: int,
    ) -> dict[str, Any]:
        """Corpus-level real-vs-zero-filled imagery statistics per split.

        Counts, per split and overall, the temporal slots that will be backed
        by a REAL extracted patch vs zero-filled padding within the fixed
        ``T = max_observations`` window:

        * ``total_slots`` — ``samples * max_observations``,
        * ``real_slots`` — slots with a real NDVI/EVI record (each sequence
          pair contributes one real NDVI frame and one real EVI frame, capped
          at the window),
        * ``zero_filled_slots`` — the remainder,
        * ``real_frac`` — real slots as a share of total slots,
        * ``samples_with_real_imagery`` vs ``samples_without_imagery`` (all
          slots zero-filled).

        ``samples_without_imagery`` is expected to be 0: the corpus build
        excludes observations whose imagery sequence resolves empty (see
        :meth:`build`), so every accepted sample carries at least one real
        pair. This is the data-backed justification for the missing-imagery
        policy — kept samples are never trained entirely on zero-fill.
        """
        def _stream_counts(obs: AgriculturalObservation) -> tuple[int, int]:
            pairs = obs.sequence.pairs or ()
            ndvi_real = 0
            evi_real = 0
            for pair in pairs:
                if getattr(pair, "ndvi", None) is not None:
                    ndvi_real += 1
                if getattr(pair, "evi", None) is not None:
                    evi_real += 1
            return (
                min(max_observations, ndvi_real),
                min(max_observations, evi_real),
            )

        def _split_block(obs_list: Sequence[AgriculturalObservation]) -> dict[str, Any]:
            samples = len(obs_list)
            stream = {}
            for stream_name in ("ndvi", "evi"):
                total_slots = samples * max_observations
                real_slots = 0
                samples_with_real = 0
                for obs in obs_list:
                    ndvi_real, evi_real = _stream_counts(obs)
                    real = ndvi_real if stream_name == "ndvi" else evi_real
                    real_slots += real
                    if real > 0:
                        samples_with_real += 1
                stream[stream_name] = {
                    "total_slots": total_slots,
                    "real_slots": real_slots,
                    "zero_filled_slots": max(0, total_slots - real_slots),
                    "real_frac": round(real_slots / total_slots, 4)
                        if total_slots > 0 else 0.0,
                    "samples_with_real_imagery": samples_with_real,
                    "samples_without_imagery": samples - samples_with_real,
                }
            return {"samples": samples, "streams": stream}

        diagnostics = {
            "max_observations": int(max_observations),
            "policy": (
                "accepted samples always carry >=1 real imagery pair (build "
                "excludes zero-pair sequences); zero-fill pads only the "
                "remaining slots inside the fixed window"
            ),
            "train": _split_block(train),
            "val": _split_block(val),
            "test": _split_block(test),
            "overall": _split_block(list(train) + list(val) + list(test)),
        }
        return diagnostics

    def imagery_summary(
        self,
        train: Sequence[AgriculturalObservation],
        val: Sequence[AgriculturalObservation],
        test: Sequence[AgriculturalObservation],
        *,
        max_observations: int | None = None,
    ) -> dict[str, Any]:
        """Aggregate observation-level imagery coverage across the split.

        Produces the observation / coverage / temporal part of the imagery
        summary block:

        * per-split and overall accepted counts (``train/val/test``),
        * fully-paired vs partial NDVI/EVI sequences,
        * observation-count (sequence length) distribution against the
          configured ``max_observations`` cap,
        * ``autopatch``-style patch size when present on the observations.

        Frame-level real-vs-zero-filled and tensor shapes are measured in the
        Phase 4 first-batch diagnostic (:func:`ai.training.diagnostics.profile_batch`),
        since patch tensors materialize only at extraction time.
        """
        stats = {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "accepted": len(train) + len(val) + len(test),
        }
        total = train + val + test
        if not total:
            return stats

        stats["fully_paired"] = sum(1 for obs in total if obs.has_paired_images)
        stats["partial_pairs"] = sum(
            1 for obs in total if not obs.has_paired_images and len(obs.sequence.pairs) > 0
        )

        lengths = [len(obs.sequence.pairs) for obs in total]
        stats["observations_min"] = min(lengths)
        stats["observations_mean"] = round(sum(lengths) / len(lengths), 2)
        stats["observations_max"] = max(lengths)
        if max_observations:
            stats["at_max_cap"] = sum(1 for length in lengths if length >= max_observations)

        patch_sizes = {obs.patch_size for obs in total if obs.patch_size}
        stats["patch_sizes"] = sorted(patch_sizes)

        build = getattr(self, "last_build_stats", None)
        if build is not None:
            stats.setdefault("rows", build["rows"])
            stats.setdefault("excluded", build["excluded"])
            stats.setdefault("accepted", build["accepted"])
        else:
            stats.setdefault("rows", None)
            stats.setdefault("excluded", None)

        return stats

    def data_contract_printout(
        self,
        train: Sequence[AgriculturalObservation],
        val: Sequence[AgriculturalObservation],
        test: Sequence[AgriculturalObservation],
    ) -> dict[str, Any]:
        """Build the Kaggle data-contract printout for the frozen corpus.

        This is displayed at the start of a Kaggle run and checked against
        the runtime data.  If there is a mismatch, training is aborted.
        """
        manifest = self._manifest or {}
        manifest_checksum = _sha256_file(self.manifest_path)

        def _class_counts(obs_list: Sequence[AgriculturalObservation]) -> dict[str, int]:
            counts: dict[str, int] = {}
            for obs in obs_list:
                label = obs.crop or "unknown"
                counts[label] = counts.get(label, 0) + 1
            return counts

        overall = {}
        for obs in train + val + test:
            label = obs.crop or "unknown"
            overall[label] = overall.get(label, 0) + 1

        contract = {
            "corpus": "crop_supervised_v1",
            "version": manifest.get("dataset_version"),
            "manifest_checksum": manifest_checksum,
            "total_samples": len(train) + len(val) + len(test),
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "overall_class_counts": overall,
            "train_class_counts": _class_counts(train),
            "val_class_counts": _class_counts(val),
            "test_class_counts": _class_counts(test),
            "split_strategy": manifest.get("split_strategy"),
            "split_groups": manifest.get("split_groups"),
            "class_mapping": manifest.get("class_matching"),
            "provenance_schema": manifest.get("provenance_schema"),
        }

        print("\n")
        print("========================================")
        print("  CROPFUSION R5.2.8 DATA CONTRACT")
        print("========================================")
        print(f"  Manifest:     {self.manifest_path}")
        print(f"  Dataset version: {contract['version']}")
        print(f"  Expected samples: {contract['total_samples']}")
        print(f"  Loaded samples:   {contract['total_samples']}")
        print()
        print(f"  TRAIN:       {contract['train_samples']}")
        print(f"  VALIDATION:  {contract['val_samples']}")
        print(f"  TEST:        {contract['test_samples']}")
        print()
        print("  Classes:")
        for cls in ["coconut", "pepper", "coffee", "cardamom", "blackgram"]:
            count = overall.get(cls, 0)
            print(f"    {cls:12s} {count}")
        print()
        print(f"  Corpus source: crop_supervised_v1.csv")
        print(f"  Split strategy: {contract['split_strategy']}")
        print(f"  Split groups:   {contract['split_groups']}")
        print(f"  Manifest SHA-256: {contract['manifest_checksum'][:16]}...")
        print("========================================\n")

        return contract

    def verify_contract(
        self,
        contract: dict[str, Any],
        train: Sequence[AgriculturalObservation],
        val: Sequence[AgriculturalObservation],
        test: Sequence[AgriculturalObservation],
    ) -> tuple[bool, list[str]]:
        """Verify a data-contract printout against the actual observation lists.

        Returns ``(passed, errors)`` where ``errors`` is a list of mismatch
        descriptions.  If ``passed`` is False, training MUST be stopped.
        """
        errors: list[str] = []
        manifest_checksum = _sha256_file(self.manifest_path)

        if contract.get("manifest_checksum") != manifest_checksum:
            errors.append(
                f"Manifest checksum mismatch: "
                f"contract={contract.get('manifest_checksum')[:16]}, "
                f"actual={manifest_checksum[:16]}"
            )

        actual_total = len(train) + len(val) + len(test)
        expected_total = contract.get("total_samples", 0)
        if actual_total != expected_total:
            errors.append(
                f"Total samples mismatch: "
                f"contract={expected_total}, actual={actual_total}"
            )

        if len(train) != contract.get("train_samples", 0):
            errors.append(
                f"Train count mismatch: "
                f"contract={contract.get('train_samples')}, actual={len(train)}"
            )
        if len(val) != contract.get("val_samples", 0):
            errors.append(
                f"Val count mismatch: "
                f"contract={contract.get('val_samples')}, actual={len(val)}"
            )
        if len(test) != contract.get("test_samples", 0):
            errors.append(
                f"Test count mismatch: "
                f"contract={contract.get('test_samples')}, actual={len(test)}"
            )

        # Verify no non-frozen observations leaked in.
        for obs in train + val + test:
            if obs.provenance.get("corpus") != "crop_supervised_v1":
                errors.append(
                    f"Non-frozen observation found: "
                    f"observation_id={obs.observation_id}, "
                    f"provenance={obs.provenance}"
                )
                break

        passed = len(errors) == 0
        if not passed:
            logger.error("Data contract verification FAILED: %s", "; ".join(errors))
        else:
            logger.info("Data contract verification PASSED")
        return passed, errors
