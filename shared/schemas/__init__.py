"""Shared schemas / metadata models for the CropFusion platforms.

These are platform-agnostic dataclasses used for metadata, reporting and
interchange between the Training and Application platforms.  They are NOT the
Application platform's API request models (those stay in
``application/backend/app/schemas``).
"""

from __future__ import annotations

from .dataset import (
    DatasetInventorySchema,
    DatasetSummarySchema,
    FileEntrySchema,
    MetadataRecordSchema,
)
from .image import (
    ImageDatasetLocationSchema,
    ImageDatasetRecordSchema,
    RasterMetadataSchema,
)
from .meta import (
    ConfigMetadataSchema,
    ReleaseMetadataSchema,
    TrainingRunSchema,
)
from .prediction import PredictionInputSchema, PredictionResultSchema
from .validation import ValidationIssueSchema, ValidationReportSchema

__all__ = [
    "ConfigMetadataSchema",
    "DatasetInventorySchema",
    "DatasetSummarySchema",
    "FileEntrySchema",
    "ImageDatasetLocationSchema",
    "ImageDatasetRecordSchema",
    "MetadataRecordSchema",
    "PredictionInputSchema",
    "PredictionResultSchema",
    "RasterMetadataSchema",
    "ReleaseMetadataSchema",
    "TrainingRunSchema",
    "ValidationIssueSchema",
    "ValidationReportSchema",
]
