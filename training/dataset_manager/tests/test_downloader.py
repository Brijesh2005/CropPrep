"""Tests for the Kaggle downloader (mock downloads, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.dataset_manager.config import DownloadConfig
from training.dataset_manager.downloader import KaggleDownloader
from training.dataset_manager.exceptions import DownloadFailedError


def _downloader(fake_kaggle, *, config: DownloadConfig | None = None) -> KaggleDownloader:
    dl = KaggleDownloader(config or DownloadConfig(), kagglehub_module=fake_kaggle.module)
    dl._cache_root = fake_kaggle.cache_root  # isolate from the real home cache
    return dl


def test_is_downloaded_detects_existing(fake_kaggle):
    dl = _downloader(fake_kaggle)
    assert not dl.is_downloaded("owner/crop")
    file_path = fake_kaggle.cache_root / "owner" / "crop" / "1" / "file.tif"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"x")
    assert dl.is_downloaded("owner/crop")


def test_download_calls_kagglehub_and_returns_path(fake_kaggle):
    dl = _downloader(fake_kaggle)
    path = dl.download("owner/crop")
    assert path == fake_kaggle.downloaded_root
    assert fake_kaggle.module.calls[-1]["handle"] == "owner/crop"
    assert fake_kaggle.module.calls[-1]["force"] is False


def test_download_reuses_existing_without_call(fake_kaggle):
    dl = _downloader(fake_kaggle)
    file_path = fake_kaggle.cache_root / "owner" / "crop" / "2" / "f.tif"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"y")
    before = len(fake_kaggle.module.calls)
    path = dl.download("owner/crop")
    assert path == fake_kaggle.cache_root / "owner" / "crop" / "2"
    assert len(fake_kaggle.module.calls) == before  # no network call


def test_download_force_propagates(fake_kaggle):
    dl = _downloader(fake_kaggle)
    dl.download("owner/crop", force=True)
    assert fake_kaggle.module.calls[-1]["force"] is True


def test_download_failure_wrapped(fake_kaggle):
    dl = _downloader(fake_kaggle)
    fake_kaggle.module.fail_next = True
    with pytest.raises(DownloadFailedError):
        dl.download("owner/crop")


def test_materialize_hardlink_or_copy(fake_kaggle, tmp_path: Path):
    dl = _downloader(fake_kaggle)
    source = fake_kaggle.downloaded_root
    (source / "files" / "extra.txt").write_text("hello", encoding="utf-8")
    dest = tmp_path / "raw" / "catalog"
    count = dl.materialize(source, dest)
    assert count == 2
    assert (dest / "files" / "S2_NDVI_2020.tif").exists()
    assert (dest / "files" / "extra.txt").read_text() == "hello"


def test_materialize_force_copy(fake_kaggle, tmp_path: Path):
    dl = _downloader(fake_kaggle, config=DownloadConfig(link_method="copy"))
    source = fake_kaggle.downloaded_root
    dest = tmp_path / "raw" / "catalog"
    count = dl.materialize(source, dest)
    assert count == 1


def test_verify_integrity_flags_empty_files(tmp_path: Path):
    dl = KaggleDownloader(DownloadConfig(), kagglehub_module=None)
    root = tmp_path / "bad"
    root.mkdir()
    (root / "empty.csv").write_bytes(b"")
    (root / "ok.csv").write_bytes(b"a,b\n1,2\n")
    assert dl.verify_integrity(root) is False
    (root / "empty.csv").write_bytes(b"a,b\n1,2\n")
    assert dl.verify_integrity(root) is True


def test_verify_integrity_flags_bad_tiff_magic(tmp_path: Path):
    dl = KaggleDownloader(DownloadConfig(), kagglehub_module=None)
    root = tmp_path / "bad"
    root.mkdir()
    (root / "img.tif").write_bytes(b"not a tiff")
    assert dl.verify_integrity(root) is False
