"""Training Cache for the Kaggle Training Platform (R2.1).

A small, dependency-free JSON-backed cache with the five buckets the Training
Platform needs:

* ``metadata``        — dataset / run metadata,
* ``preprocessing``   — preprocessing artefacts & fitted state keys,
* ``image_metadata``  — per-image (raster) metadata lookups,
* ``statistics``      — computed statistics (means / std / bounds),
* ``validation``      — validation results + report digests.

Each bucket is a JSON file under the workspace ``cache`` directory with per-key
TTLs and LRU eviction. Pure infrastructure — no training logic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

#: Valid cache buckets.
SECTIONS = (
    "metadata",
    "preprocessing",
    "image_metadata",
    "statistics",
    "validation",
)


class TrainingCache:
    """JSON-backed, namespaced cache for the Kaggle workspace.

    Args:
        cache_dir: Workspace cache directory.
        max_entries: Hard cap per bucket (LRU eviction).
        default_ttl_seconds: Default TTL for entries without an explicit TTL
            (None = never expire).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        max_entries: int = 1000,
        default_ttl_seconds: int | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        self._buckets: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #

    def ensure_layout(self) -> Path:
        """Create the cache directory."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir

    def _bucket_file(self, section: str) -> Path:
        if section not in SECTIONS:
            raise ValueError(
                f"unknown cache section: {section!r} (use one of {SECTIONS})"
            )
        return self.cache_dir / f"{section}.json"

    def _load(self, section: str) -> dict[str, Any]:
        if section in self._buckets:
            return self._buckets[section]
        data: dict[str, Any] = {"entries": {}, "order": []}
        path = self._bucket_file(section)
        if path.exists():
            try:
                with path.open(encoding="utf-8") as fh:
                    raw = json.load(fh)
                data["entries"] = raw.get("entries", {})
                data["order"] = raw.get("order", [])
            except (json.JSONDecodeError, OSError):
                data = {"entries": {}, "order": []}
        self._buckets[section] = data
        return data

    def _save(self, section: str) -> None:
        self.ensure_layout()
        data = self._buckets[section]
        with self._bucket_file(section).open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def get(self, section: str, key: str) -> Any | None:
        """Return the cached value for ``key`` or None (missing/expired)."""
        bucket = self._load(section)
        entry = bucket["entries"].get(key)
        if entry is None:
            return None
        if entry.get("expires_at") is not None and time.time() > entry["expires_at"]:
            self.delete(section, key)
            return None
        return entry.get("value")

    def has(self, section: str, key: str) -> bool:
        return self.get(section, key) is not None

    def set(
        self,
        section: str,
        key: str,
        value: Any,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store ``value`` under ``section/key`` with an optional TTL."""
        bucket = self._load(section)
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        expires_at = time.time() + ttl if ttl is not None else None
        if key not in bucket["entries"]:
            bucket["order"].append(key)
        bucket["entries"][key] = {"value": value, "expires_at": expires_at}
        self._evict(section, bucket)
        self._save(section)

    def delete(self, section: str, key: str) -> bool:
        """Remove ``key``; True when it existed."""
        bucket = self._load(section)
        existed = key in bucket["entries"]
        if existed:
            bucket["entries"].pop(key, None)
            if key in bucket["order"]:
                bucket["order"].remove(key)
            self._save(section)
        return existed

    def clear(self, section: str | None = None) -> int:
        """Clear one bucket (or all buckets); returns the number removed."""
        removed = 0
        targets = [section] if section else list(SECTIONS)
        for name in targets:
            bucket = self._load(name)
            removed += len(bucket["entries"])
            bucket["entries"] = {}
            bucket["order"] = []
            path = self._bucket_file(name)
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
            self._buckets[name] = bucket
        return removed

    def _evict(self, section: str, bucket: dict[str, Any]) -> None:
        while len(bucket["entries"]) > self.max_entries and bucket["order"]:
            oldest = bucket["order"].pop(0)
            bucket["entries"].pop(oldest, None)
        bucket["order"] = [
            k for k in bucket["order"] if k in bucket["entries"]
        ]

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """Per-bucket entry counts + total."""
        counts: dict[str, int] = {}
        for section in SECTIONS:
            counts[section] = len(self._load(section)["entries"])
        return {
            "cache_dir": str(self.cache_dir),
            "sections": counts,
            "total": sum(counts.values()),
            "max_entries_per_section": self.max_entries,
        }
