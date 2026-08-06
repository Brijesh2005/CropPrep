"""FeatureStoreManager — the single entry point for all R5 caches.

Responsibilities: create, load, update, invalidate, version and checksum
cached artefacts across four backends (NumPy ``.npy``/``.npz``, Parquet,
Torch tensors ``.pt``, and JSON metadata), plus a manifest that records the
config fingerprint each namespace was built against.

This module has NO dependency on Dataset Manager / STAM / model code — it is
a generic, reusable disk cache. The *_cache.py modules in this package wrap
it with domain-specific key/value logic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("training.feature_store")

Backend = Literal["numpy", "numpy_npz", "parquet", "torch", "json"]

_EXT = {
    "numpy": ".npy",
    "numpy_npz": ".npz",
    "parquet": ".parquet",
    "torch": ".pt",
    "json": ".json",
}


@dataclass
class FeatureStoreConfig:
    """Where the store lives and which namespaces are enabled.

    Mirrors ``training/config/performance.yaml`` — load with
    ``FeatureStoreConfig.from_dict(yaml_dict["feature_store"])``.
    """

    root: str = "training/feature_store"
    enabled: bool = True
    namespaces: tuple[str, ...] = (
        "image_embeddings",
        "temporal_sequences",
        "tabular_cache",
        "metadata",
        "manifests",
    )
    checksum_algo: str = "sha256"
    manifest_filename: str = "manifest.json"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FeatureStoreConfig":
        data = data or {}
        return cls(
            root=data.get("root", cls.root),
            enabled=data.get("enabled", cls.enabled),
            checksum_algo=data.get("checksum_algo", cls.checksum_algo),
        )


@dataclass
class CacheEntry:
    key: str
    namespace: str
    backend: Backend
    path: str
    checksum: str
    version: int
    created_at: float
    fingerprint: str
    extra: dict[str, Any] = field(default_factory=dict)


class FeatureStoreManager:
    """Generic disk-backed feature cache with versioning + checksums.

    Usage::

        store = FeatureStoreManager(FeatureStoreConfig())
        if not store.exists("image_embeddings", key):
            emb = compute_embedding(...)
            store.put("image_embeddings", key, emb, backend="numpy",
                       fingerprint=fp)
        emb = store.get("image_embeddings", key, backend="numpy")
    """

    def __init__(self, config: FeatureStoreConfig | None = None) -> None:
        self.config = config or FeatureStoreConfig()
        self.root = Path(self.config.root)
        self._manifest_path = self.root / "manifests" / self.config.manifest_filename
        self._manifest: dict[str, Any] = {}
        self._stats = {"hits": 0, "misses": 0, "writes": 0, "invalidations": 0}
        if self.config.enabled:
            self._ensure_dirs()
            self._load_manifest()

    # -- setup ---------------------------------------------------------- #

    def _ensure_dirs(self) -> None:
        for ns in self.config.namespaces:
            (self.root / ns).mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> None:
        if self._manifest_path.exists():
            try:
                self._manifest = json.loads(self._manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt manifest at %s — starting fresh", self._manifest_path)
                self._manifest = {}
        self._manifest.setdefault("entries", {})

    def _save_manifest(self) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._manifest, indent=2, sort_keys=True))
        tmp.replace(self._manifest_path)

    # -- paths / checksums ------------------------------------------------ #

    def _path_for(self, namespace: str, key: str, backend: Backend) -> Path:
        safe_key = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.root / namespace / f"{safe_key}{_EXT[backend]}"

    def _checksum(self, path: Path) -> str:
        h = hashlib.new(self.config.checksum_algo)
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _manifest_key(self, namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    # -- public API -------------------------------------------------------- #

    def exists(self, namespace: str, key: str, *, fingerprint: str | None = None) -> bool:
        """True if a valid (non-invalidated) cache entry exists."""
        if not self.config.enabled:
            return False
        mkey = self._manifest_key(namespace, key)
        entry = self._manifest.get("entries", {}).get(mkey)
        if entry is None:
            return False
        if fingerprint is not None and entry.get("fingerprint") != fingerprint:
            return False  # config/version drift -> treat as miss, caller will overwrite
        path = Path(entry["path"])
        return path.exists()

    def get(self, namespace: str, key: str, *, backend: Backend, verify_checksum: bool = False) -> Any:
        """Load a cached value. Raises KeyError if absent."""
        mkey = self._manifest_key(namespace, key)
        entry = self._manifest.get("entries", {}).get(mkey)
        if entry is None:
            self._stats["misses"] += 1
            raise KeyError(f"No cache entry for {mkey}")
        path = Path(entry["path"])
        if not path.exists():
            self._stats["misses"] += 1
            raise KeyError(f"Cache entry {mkey} references missing file {path}")
        if verify_checksum and self._checksum(path) != entry["checksum"]:
            self._stats["misses"] += 1
            raise ValueError(f"Checksum mismatch for {mkey} — cache corrupted, treat as invalidated")
        self._stats["hits"] += 1
        return self._read(path, backend)

    def put(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        backend: Backend,
        fingerprint: str = "",
        extra: dict[str, Any] | None = None,
    ) -> CacheEntry:
        """Write a value to the store and record it in the manifest."""
        path = self._path_for(namespace, key, backend)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write(path, value, backend)
        checksum = self._checksum(path)
        mkey = self._manifest_key(namespace, key)
        prev = self._manifest.get("entries", {}).get(mkey)
        version = (prev["version"] + 1) if prev else 1
        entry = CacheEntry(
            key=key,
            namespace=namespace,
            backend=backend,
            path=str(path),
            checksum=checksum,
            version=version,
            created_at=time.time(),
            fingerprint=fingerprint,
            extra=extra or {},
        )
        self._manifest.setdefault("entries", {})[mkey] = entry.__dict__
        self._save_manifest()
        self._stats["writes"] += 1
        return entry

    def invalidate(self, namespace: str, key: str | None = None) -> int:
        """Invalidate one key, or every key in a namespace when key is None."""
        entries = self._manifest.get("entries", {})
        removed = 0
        for mkey in list(entries):
            ns, k = mkey.split(":", 1)
            if ns != namespace:
                continue
            if key is not None and k != key:
                continue
            path = Path(entries[mkey]["path"])
            if path.exists():
                path.unlink(missing_ok=True)
            del entries[mkey]
            removed += 1
        if removed:
            self._save_manifest()
            self._stats["invalidations"] += removed
        return removed

    def stats(self) -> dict[str, int]:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total) if total else 0.0
        return {**self._stats, "hit_rate": round(hit_rate, 4)}

    # -- backend read/write ------------------------------------------------- #

    def _write(self, path: Path, value: Any, backend: Backend) -> None:
        if backend == "numpy":
            import numpy as np
            np.save(path, value)
        elif backend == "numpy_npz":
            import numpy as np
            np.savez_compressed(path, **value)
        elif backend == "parquet":
            value.to_parquet(path)  # value: pandas.DataFrame
        elif backend == "torch":
            import torch
            torch.save(value, path)
        elif backend == "json":
            path.write_text(json.dumps(value, indent=2))
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _read(self, path: Path, backend: Backend) -> Any:
        if backend == "numpy":
            import numpy as np
            return np.load(path, allow_pickle=False)
        elif backend == "numpy_npz":
            import numpy as np
            return dict(np.load(path))
        elif backend == "parquet":
            import pandas as pd
            return pd.read_parquet(path)
        elif backend == "torch":
            import torch
            return torch.load(path, map_location="cpu")
        elif backend == "json":
            return json.loads(path.read_text())
        else:
            raise ValueError(f"Unknown backend: {backend}")
