"""Dataset downloader backed by ``kagglehub``.

Responsibilities (per the SDD Dataset Manager module):

* **Automatic download** — ``kagglehub.dataset_download`` fetches the primary
  image dataset (``shathanandabhatn/crop-yield-forecasting-karnataka-
  dakshina-kannada``). No manual downloading is ever required.
* **Detect existing installations** — the Kaggle cache is probed on the file
  system first; a download only happens when the dataset is missing.
* **Re-download support** — ``force=True`` passes ``force_download`` through.
* **Progress reporting** — the materialisation phase reports byte progress.
* **Integrity verification** — a lightweight sanity pass checks for empty
  files and invalid raster magic bytes after materialisation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Protocol

from .config import DownloadConfig
from .exceptions import CorruptedDatasetError, DownloadFailedError
from .interfaces import Downloader
from .logger import get_logger
from .utils import copy_file_with_progress, is_geotiff_bytes, run_parallel

logger = get_logger("downloader")

ProgressCallback = Callable[[int, int, str], None]


class KaggleHubLike(Protocol):
    """Minimal structural type for ``kagglehub.dataset_download``."""

    def dataset_download(self, handle: str, force_download: bool = False) -> str: ...


def _default_kaggle_cache_root() -> Path:
    """Location where kagglehub stores downloaded datasets."""
    return Path.home() / ".cache" / "kagglehub" / "datasets"


class KaggleDownloader(Downloader):
    """Concrete :class:`Downloader` implementation for Kaggle datasets.

    Args:
        config: Download configuration section.
        kagglehub_module: Optional kagglehub module instance (injected for
            tests). Defaults to importing ``kagglehub`` on first use.
    """

    def __init__(
        self,
        config: DownloadConfig | None = None,
        *,
        kagglehub_module: KaggleHubLike | None = None,
    ) -> None:
        self.config = config or DownloadConfig()
        self._kagglehub = kagglehub_module
        self._cache_root = _default_kaggle_cache_root()

    # -- Public API ----------------------------------------------------------- #

    def download(self, handle: str, *, force: bool = False) -> Path:
        """Ensure ``handle`` is present locally; return its local root path.

        Args:
            handle: Kaggle dataset handle, e.g.
                ``"shathanandabhatn/..."``.
            force: Re-download even when an existing copy is found.

        Returns:
            The directory containing the downloaded dataset files.

        Raises:
            DownloadFailedError: When the download or materialisation fails.
        """
        if not force and self.is_downloaded(handle):
            logger.info("Dataset already present; skipping download", extra={"handle": handle})
            return self.resolve_downloaded(handle)

        try:
            module = self._kagglehub or self._import_kagglehub()
            logger.info("Downloading dataset from Kaggle", extra={"handle": handle})
            path = Path(module.dataset_download(handle, force_download=force))
        except Exception as exc:  # noqa: BLE001 - wrap any failure
            raise DownloadFailedError(
                f"Kaggle download failed for {handle}: {exc}", detail=str(exc)
            ) from exc

        if not path.is_dir() or not any(path.rglob("*")):
            raise DownloadFailedError(
                "Kaggle download produced an empty or missing directory",
                detail=str(path),
            )
        logger.info(
            "Dataset downloaded", extra={"handle": handle, "path": str(path)}
        )
        return path

    def is_downloaded(self, handle: str) -> bool:
        """True when ``handle`` already exists in the local Kaggle cache."""
        root = self._handle_cache_dir(handle)
        if not root.is_dir():
            return False
        for candidate in root.iterdir():
            if candidate.is_dir() and any(candidate.rglob("*")):
                return True
        return False

    def resolve_downloaded(self, handle: str) -> Path:
        """Return the path of an already-downloaded dataset (or raise)."""
        root = self._handle_cache_dir(handle)
        if not root.is_dir():
            raise DownloadFailedError(
                f"Dataset not found in local cache: {handle}", detail=str(root)
            )
        # Prefer the highest version directory (lexicographic == semver order).
        versions = sorted(
            (c for c in root.iterdir() if c.is_dir() and any(c.rglob("*"))),
            key=lambda p: p.name,
        )
        if not versions:
            raise DownloadFailedError(
                f"Dataset cache is empty: {handle}", detail=str(root)
            )
        return versions[-1]

    def materialize(
        self,
        source: Path,
        destination: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> int:
        """Mirror ``source`` into ``destination`` using hard links where possible.

        Hard links are attempted first (fast, no duplication); when the
        filesystem does not permit linking (e.g. cross-device), a streaming
        copy with progress is used as a fallback.

        Args:
            source: Directory to mirror.
            destination: Canonical location inside the managed raw root.
            progress: Optional ``(copied, total, filename)`` callback.

        Returns:
            Number of files materialised.
        """
        files = sorted(p for p in source.rglob("*") if p.is_file())
        total = len(files)
        done = 0
        for src in files:
            rel = src.relative_to(source)
            dst = destination / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if self._link_method() == "hardlink":
                try:
                    if not dst.exists():
                        dst.hardlink_to(src)
                except OSError:
                    copy_file_with_progress(src, dst)
            else:
                copy_file_with_progress(src, dst)
            done += 1
            if progress is not None:
                progress(done, total, str(rel))
        logger.info(
            "Materialised dataset", extra={"source": str(source), "files": done}
        )
        return done

    def verify_integrity(self, root: Path) -> bool:
        """Lightweight sanity check: no empty files, valid raster magic bytes.

        Full structural validation is performed by the validator; this is a
        fast pre-flight used right after a download.
        """
        issues: list[str] = []

        def _check(path: Path) -> str | None:
            if path.stat().st_size == 0:
                return f"empty file: {path}"
            if path.suffix.lower() in {".tif", ".tiff"} and not is_geotiff_bytes(path):
                return f"invalid TIFF magic: {path}"
            return None

        results = run_parallel(list(root.rglob("*")), _check, workers=4)
        for result in results:
            if isinstance(result, str):
                issues.append(result)
        if issues:
            logger.warning("Integrity issues found", extra={"count": len(issues)})
            return False
        return True

    # -- Internals ------------------------------------------------------------- #

    def _import_kagglehub(self) -> KaggleHubLike:
        try:
            import kagglehub  # type: ignore[import-not-found]

            return kagglehub  # type: ignore[return-value]
        except ImportError as exc:  # pragma: no cover - env dependency
            raise DownloadFailedError(
                "kagglehub is not installed; run `pip install kagglehub`",
                detail="kagglehub",
            ) from exc

    def _handle_cache_dir(self, handle: str) -> Path:
        parts = handle.strip("/").split("/")
        if len(parts) != 2:
            raise DownloadFailedError(
                f"Invalid Kaggle handle (expected owner/name): {handle}"
            )
        return self._cache_root / parts[0] / parts[1]

    def _link_method(self) -> str:
        return (self.config.link_method or "hardlink").lower()
