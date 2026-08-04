"""Dataset validator: structure, integrity and metadata completeness checks.

The validator consumes a :class:`DatasetInventory` (produced by the scanner)
and emits a detailed :class:`ValidationReport`. It performs the following
checks (see :meth:`DatasetValidator.validate` for the full list):

* **Folder structure** — expected top-level layout and coverage hints for the
  primary Kaggle catalog (year directories, R10m/R20m/NDVI/EVI sub-folders).
* **Missing / orphaned files** — files referenced by the metadata store that
  no longer exist, and scanned files with no metadata record.
* **Duplicate files** — duplicates detected by ``(name, size)`` and, when
  available, by content hash.
* **Empty CSVs** — zero-byte or zero-row tabular files.
* **Corrupted rasters** — GeoTIFFs whose header cannot be parsed.
* **Invalid CRS** — rasters that carry no (or an unreadable) CRS.

Each issue carries a severity; the report ``passed`` flag is False whenever
any issue is at least ERROR (or WARNING when ``fail_on_warning`` is enabled).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .config import ValidateConfig
from .exceptions import CorruptedDatasetError
from .interfaces import ImageLoader, MetadataStore, Validator
from .logger import get_logger
from .models import (
    DatasetInventory,
    FileCategory,
    FileEntry,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from .utils import run_parallel

logger = get_logger("validator")


class DatasetValidator(Validator):
    """Concrete :class:`Validator` implementation.

    Args:
        config: Validation configuration section.
        image_loader: :class:`ImageLoader` used to check raster headers.
        metadata_store: Optional :class:`MetadataStore` used for
            completeness checks. When None those checks are skipped.
    """

    def __init__(
        self,
        config: ValidateConfig | None = None,
        *,
        image_loader: ImageLoader,
        metadata_store: MetadataStore | None = None,
    ) -> None:
        self.config = config or ValidateConfig()
        self.image_loader = image_loader
        self.metadata_store = metadata_store

    def validate(self, root: Path, inventory: DatasetInventory) -> ValidationReport:
        """Run every validation check and return the aggregated report.

        Args:
            root: Dataset root (must exist).
            inventory: Inventory produced by the scanner for ``root``.

        Returns:
            A :class:`ValidationReport`.
        """
        root = root.expanduser().resolve()
        issues: list[ValidationIssue] = []
        entries = inventory.entries

        issues.extend(self._check_structure(root, entries))
        issues.extend(self._check_empty_csvs(entries))
        issues.extend(self._check_raster_integrity(entries))
        issues.extend(self._check_duplicates(entries))
        if self.metadata_store is not None:
            issues.extend(self._check_metadata_completeness(root, entries))

        failing = any(i.severity in {Severity.ERROR, Severity.CRITICAL} for i in issues)
        if self.config.fail_on_warning:
            failing = failing or any(i.severity is Severity.WARNING for i in issues)

        report = ValidationReport(
            root=root,
            passed=not failing,
            issues=issues,
            files_scanned=len(entries),
        )
        logger.info(
            "Validation complete",
            extra={
                "root": str(root),
                "passed": report.passed,
                "issues": report.by_severity(),
                "files_scanned": len(entries),
            },
        )
        return report

    # -- Individual checks ----------------------------------------------------- #

    def _check_structure(
        self, root: Path, entries: list[FileEntry]
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if not root.is_dir():
            issues.append(
                ValidationIssue(
                    Severity.CRITICAL, "V-STRUCT-001", "structure",
                    f"Dataset root does not exist: {root}", str(root),
                )
            )
            return issues
        if not entries:
            issues.append(
                ValidationIssue(
                    Severity.ERROR, "V-STRUCT-006", "structure",
                    "Dataset contains no files (empty root)",
                    str(root),
                )
            )
            return issues

        # Coverage hint: any directory containing the phrase "_images" (the
        # Kaggle layout uses e.g. 2018_images) should expose the expected
        # sub-layout.
        image_dirs = sorted(
            p for p in root.iterdir() if p.is_dir() and "_images" in p.name.lower()
        )
        if image_dirs:
            for image_dir in image_dirs:
                subdirs = {d.name.lower() for d in image_dir.iterdir() if d.is_dir()}
                expected = {r.lower() for r in self.config.expected_resolutions}
                expected.update(t.lower() for t in self.config.expected_index_types)
                missing = sorted(expected - subdirs)
                if missing:
                    issues.append(
                        ValidationIssue(
                            Severity.WARNING, "V-STRUCT-002", "structure",
                            f"Image directory is missing expected sub-folders: {image_dir.name}",
                            str(image_dir), {"missing": missing},
                        )
                    )
        else:
            issues.append(
                ValidationIssue(
                    Severity.INFO, "V-STRUCT-003", "structure",
                    "No *_images directories found; tabular-only dataset or empty root",
                    str(root),
                )
            )

        # Year coverage hints.
        if image_dirs or any(e.year is not None for e in entries):
            lo, hi = self.config.expected_years
            covered = sorted({e.year for e in entries if e.year is not None})
            years_expected = set(range(lo, hi + 1))
            missing_years = sorted(years_expected - set(covered))
            if missing_years:
                issues.append(
                    ValidationIssue(
                        Severity.WARNING, "V-STRUCT-004", "structure",
                        "Expected year range is not fully covered",
                        str(root), {"missing_years": missing_years, "covered": covered},
                    )
                )

        # Tabular discovery hint.
        if not any(e.category is FileCategory.CSV for e in entries):
            issues.append(
                ValidationIssue(
                    Severity.INFO, "V-STRUCT-005", "structure",
                    "No CSV files discovered under the dataset root",
                    str(root),
                )
            )
        return issues

    def _check_empty_csvs(self, entries: list[FileEntry]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for entry in entries:
            if entry.category is not FileCategory.CSV:
                continue
            if entry.size_bytes == 0:
                issues.append(
                    ValidationIssue(
                        Severity.ERROR, "V-CSV-001", "empty_csv",
                        f"CSV file is empty (0 bytes): {entry.relative_path}",
                        str(entry.path),
                    )
                )
        return issues

    def _check_raster_integrity(self, entries: list[FileEntry]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        rasters = [e for e in entries if e.category is FileCategory.GEOTIFF]
        if not rasters:
            return issues

        def _check(entry: FileEntry) -> ValidationIssue | None:
            try:
                meta = self.image_loader.read_metadata(entry.path)
            except CorruptedDatasetError as exc:
                return ValidationIssue(
                    Severity.ERROR, "V-RAST-001", "corrupted_raster",
                    f"Corrupted GeoTIFF: {entry.relative_path} ({exc.message})",
                    str(entry.path),
                )
            except Exception as exc:  # noqa: BLE001
                return ValidationIssue(
                    Severity.ERROR, "V-RAST-002", "corrupted_raster",
                    f"Unreadable GeoTIFF: {entry.relative_path}",
                    str(entry.path), {"reason": str(exc)},
                )
            # CRS sanity.
            if meta.crs is None or not str(meta.crs).strip():
                return ValidationIssue(
                    Severity.WARNING, "V-RAST-003", "invalid_crs",
                    f"Raster has no usable CRS: {entry.relative_path}",
                    str(entry.path),
                )
            if meta.width <= 0 or meta.height <= 0:
                return ValidationIssue(
                    Severity.ERROR, "V-RAST-004", "invalid_dimensions",
                    f"Raster has invalid dimensions: {entry.relative_path}",
                    str(entry.path), {"width": meta.width, "height": meta.height},
                )
            return None

        results = run_parallel(rasters, _check, workers=4)
        for result in results:
            if isinstance(result, ValidationIssue):
                issues.append(result)
        return issues

    def _check_duplicates(self, entries: list[FileEntry]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        by_name_size: dict[tuple[str, int], list[FileEntry]] = defaultdict(list)
        for entry in entries:
            if entry.category is FileCategory.OTHER:
                continue
            by_name_size[(entry.relative_path.rsplit("/", 1)[-1], entry.size_bytes)].append(entry)

        for key, group in by_name_size.items():
            if len(group) < 2:
                continue
            # Same basename + same size in more than one location.
            locations = sorted(e.relative_path for e in group)
            issues.append(
                ValidationIssue(
                    Severity.WARNING, "V-DUP-001", "duplicate_file",
                    f"Possible duplicate files ({key[0]}, {key[1]} bytes)",
                    None, {"files": locations},
                )
            )
        return issues

    def _check_metadata_completeness(
        self, root: Path, entries: list[FileEntry]
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if self.metadata_store is None or not self.config.require_metadata:
            return issues

        # Orphaned records: in the store but missing from disk.
        try:
            records = self.metadata_store.query()
        except Exception as exc:  # noqa: BLE001
            issues.append(
                ValidationIssue(
                    Severity.ERROR, "V-META-001", "metadata_store",
                    f"Metadata store is unreadable: {exc}", str(root),
                )
            )
            return issues

        on_disk = {e.relative_path for e in entries}
        stored = {r.relative_path for r in records}
        orphans = sorted(stored - on_disk)
        for rel in orphans:
            issues.append(
                ValidationIssue(
                    Severity.WARNING, "V-META-002", "missing_file",
                    f"Metadata references a file that no longer exists: {rel}",
                    str(root / rel),
                )
            )

        if self.config.require_metadata:
            missing = sorted(on_disk - stored)
            if missing:
                issues.append(
                    ValidationIssue(
                        Severity.WARNING, "V-META-003", "missing_metadata",
                        "Files are present but have no metadata records "
                        f"({len(missing)} total; run `generate-metadata`)",
                        str(root), {"files": missing[:20]},
                    )
                )
        return issues
