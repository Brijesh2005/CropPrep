"""Dataset module service — integrates the Phase 2 Dataset Manager."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.exceptions import DatasetError
from app.core.logging import get_logger, PerformanceTimer
from app.modules.dataset.schemas import DatasetStatus, DatasetSummary

logger = get_logger("dataset-service")


class DatasetService:
    """Wraps the Dataset Manager and STAM for the dataset endpoints."""

    def __init__(self, dataset_manager: Any, settings: Any) -> None:
        self.manager = dataset_manager
        self.settings = settings

    # ------------------------------------------------------------------ #
    # Status / summary
    # ------------------------------------------------------------------ #

    def status(self) -> DatasetStatus:
        settings = self.settings
        catalog = settings.dataset_root and Path(settings.dataset_root) / "raw" / settings.catalog_name
        return DatasetStatus(
            dataset_root=str(settings.dataset_root) if settings.dataset_root else "",
            catalog_name=settings.catalog_name,
            catalog_exists=catalog is not None and catalog.exists(),
            ready=self.manager is not None,
        )

    def summary(self) -> DatasetSummary:
        if self.manager is None:
            raise DatasetError("dataset manager is not initialised")
        try:
            with PerformanceTimer("dataset.summary"):
                files = self._list_catalog_files()
                image_files = [f for f in files if f.suffix.lower() in {".tif", ".tiff", ".jp2"}]
                csv_files = [f for f in files if f.suffix.lower() in {".csv", ".tsv"}]
                years = self._extract_years(files)
                index_types = self._extract_index_types(files)
        except Exception as exc:
            raise DatasetError("failed to summarise the dataset", detail=str(exc)) from exc
        return DatasetSummary(
            catalog_name=self.settings.catalog_name,
            files=len(files),
            image_files=len(image_files),
            csv_files=len(csv_files),
            years=sorted(years),
            index_types=sorted(index_types),
        )

    def reload(self) -> dict[str, Any]:
        if self.manager is None:
            raise DatasetError("dataset manager is not initialised")
        try:
            with PerformanceTimer("dataset.reload"):
                self.manager.generate_metadata(force=True)
        except Exception as exc:
            raise DatasetError("failed to reload the dataset", detail=str(exc)) from exc
        return {
            "message": "dataset metadata refreshed",
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _catalog_dir(self) -> Path:
        root = self.settings.dataset_root
        if not root:
            return Path("__no_dataset__")
        return Path(root) / "raw" / self.settings.catalog_name

    def _list_catalog_files(self) -> list[Path]:
        catalog = self._catalog_dir()
        if not catalog.exists():
            return []
        return [p for p in catalog.rglob("*") if p.is_file()]

    def _extract_years(self, files: list[Path]) -> list[int]:
        years: set[int] = set()
        for path in files:
            for part in path.parts:
                if len(part) == 4 and part.isdigit():
                    years.add(int(part))
        return sorted(years)

    def _extract_index_types(self, files: list[Path]) -> list[str]:
        types: set[str] = set()
        for path in files:
            name = path.stem.upper()
            for index in ("NDVI", "EVI"):
                if index in name:
                    types.add(index)
        return sorted(types)
