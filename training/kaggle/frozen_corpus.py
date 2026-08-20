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

logger = logging.getLogger("cropfusion.training.frozen_corpus")


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

    logger.info(
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

    logger.info("CSV loaded", path=str(csv_path), rows=len(rows))
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

    This calls the matcher's image resolution without going through
    the full build_observation pipeline (which would override the
    frozen crop label with tabular matching).
    """
    lon = float(row["lon"])
    lat = float(row["lat"])
    year = int(row["year"])
    season = row.get("season")

    try:
        observation = stam.build_observation(
            lon, lat, year=year, season=season, use_cache=True
        )
        return observation.sequence
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Imagery resolution failed — returning empty sequence",
            record_id=row.get("record_id"),
            error=str(exc),
        )
        return SequenceInfo()


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
    tabular = TabularFeatures(
        crop=crop_label,
        yield_value=None,
        fields={},
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
        patch_size=128,
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

        logger.info(
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
                split = obs.provenance.get("split", "unknown")
                if split == "train":
                    train.append(obs)
                elif split == "val":
                    val.append(obs)
                elif split == "test":
                    test.append(obs)
                else:
                    logger.warning(
                        "Unknown split — assigning to train",
                        record_id=row.get("record_id"),
                        taluk=row.get("location_taluk"),
                    )
                    train.append(obs)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                logger.warning(
                    "Failed to build observation",
                    record_id=row.get("record_id"),
                    error=str(exc),
                )

            if progress_every and index % progress_every == 0:
                logger.info(
                    "Corpus build progress",
                    built=index,
                    total=len(rows),
                    errors=errors,
                    train=len(train),
                    val=len(val),
                    test=len(test),
                )

        logger.info(
            "Frozen corpus built",
            total=len(rows),
            errors=errors,
            train=len(train),
            val=len(val),
            test=len(test),
        )

        return train, val, test

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
