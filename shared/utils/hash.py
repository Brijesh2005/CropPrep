"""File hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..constants import CHUNK_SIZE


def sha256_file(path: str | Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Return the SHA-256 of a file, streaming it in bounded chunks.

    Args:
        path: File to hash.
        chunk_size: Read chunk size in bytes.

    Returns:
        Lower-case hex digest of the file contents.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: str | Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Return the MD5 of a file, streaming it in bounded chunks."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
