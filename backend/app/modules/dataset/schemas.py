"""Dataset module schemas."""

from __future__ import annotations

from pydantic import BaseModel


class DatasetStatus(BaseModel):
    dataset_root: str
    catalog_name: str
    metadata_db: str | None = None
    catalog_exists: bool = False
    ready: bool = False


class DatasetSummary(BaseModel):
    catalog_name: str
    files: int = 0
    image_files: int = 0
    csv_files: int = 0
    years: list[int] = []
    index_types: list[str] = []
    errors: list[str] = []


class ReloadResponse(BaseModel):
    message: str
    refreshed_at: str
