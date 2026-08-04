"""Dataset module tests."""

from __future__ import annotations

from app.modules.dataset.service import DatasetService


class _FakeManager:
    def generate_metadata(self, force: bool = True):
        return None


class _FakeSettings:
    dataset_root = None
    catalog_name = "kaggle-crop-yield"


def test_dataset_status():
    service = DatasetService(_FakeManager(), _FakeSettings())
    status = service.status()
    assert status.catalog_name == "kaggle-crop-yield"
    assert status.ready is True


def test_dataset_summary_empty():
    service = DatasetService(_FakeManager(), _FakeSettings())
    summary = service.summary()
    assert summary.files == 0
    assert summary.catalog_name == "kaggle-crop-yield"


def test_dataset_reload():
    service = DatasetService(_FakeManager(), _FakeSettings())
    result = service.reload()
    assert result["message"] == "dataset metadata refreshed"
