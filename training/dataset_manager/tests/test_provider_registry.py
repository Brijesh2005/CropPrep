"""Tests for :class:`ProviderRegistryImpl` — the R2.2 provider registry.

The registry is the single place providers are registered and resolved; the
manager never holds providers directly. These tests use lightweight stub
providers (no filesystem, no network).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

from training.dataset_manager.exceptions import DatasetNotFoundError
from training.dataset_manager.provider_registry import ProviderRegistryImpl
from training.dataset_manager.providers.models import (
    ProviderCapabilities,
    ProviderHealth,
    ProviderManifest,
    ProviderStatus,
)


class _StubProvider:
    def __init__(self, name: str, kind: str, available: bool = True):
        self.name = name
        self.kind = kind
        self._available = available

    def available(self) -> bool:
        return self._available

    def capabilities(self):
        return ProviderCapabilities(
            name=self.name, kind=self.kind,
            features=["manifest", "describe", f"{self.kind}_read"],
        )

    def health(self):
        return ProviderHealth(
            name=self.name, kind=self.kind, status=ProviderStatus.READY,
            available=self._available, latency_s=0.001,
            detail={"stub": True},
        )

    def manifest(self):
        return ProviderManifest(
            name=self.name, kind=self.kind, status=ProviderStatus.READY,
            available=self._available, details={"stub": True},
        )


def test_register_resolve_roundtrip():
    registry = ProviderRegistryImpl()
    provider = _StubProvider("a", "tabular")
    registry.register("a", "tabular", provider, enabled=True, priority=100)
    assert registry.has("a")
    assert registry.names() == ["a"]
    assert registry.resolve("a") is provider


def test_register_replaces_existing():
    registry = ProviderRegistryImpl()
    first = _StubProvider("a", "tabular")
    second = _StubProvider("a", "tabular")
    registry.register("a", "tabular", first)
    registry.register("a", "tabular", second)
    assert registry.resolve("a") is second
    assert len(registry.registrations()) == 1


def test_resolve_unknown_raises():
    registry = ProviderRegistryImpl()
    with pytest.raises(DatasetNotFoundError):
        registry.resolve("missing")


def test_resolve_disabled_raises():
    registry = ProviderRegistryImpl()
    provider = _StubProvider("a", "tabular")
    registry.register("a", "tabular", provider, enabled=False)
    assert registry.availability() == {"a": False}
    with pytest.raises(DatasetNotFoundError):
        registry.resolve("a")


def test_resolve_by_kind_priority_order():
    registry = ProviderRegistryImpl()
    low = _StubProvider("low", "image")
    high = _StubProvider("high", "image")
    registry.register("low", "image", low, priority=10)
    registry.register("high", "image", high, priority=100)
    resolved = registry.resolve_by_kind("image")
    assert resolved[0] is high
    assert resolved[1] is low


def test_resolve_by_kind_skips_disabled():
    registry = ProviderRegistryImpl()
    disabled = _StubProvider("off", "image")
    registry.register("off", "image", disabled, enabled=False)
    assert registry.resolve_by_kind("image") == []


def test_priority_and_config():
    registry = ProviderRegistryImpl()
    provider = _StubProvider("a", "tabular")
    registry.register("a", "tabular", provider, priority=42, config={"x": 1})
    assert registry.priority("a") == 42
    registration = registry.registrations()[0]
    assert registration.config == {"x": 1}


def test_availability_tracks_stub():
    registry = ProviderRegistryImpl()
    registry.register("ok", "tabular", _StubProvider("ok", "tabular", available=True))
    registry.register("down", "tabular", _StubProvider("down", "tabular", available=False))
    assert registry.availability() == {"down": False, "ok": True}


def test_capabilities_and_health_best_effort():
    registry = ProviderRegistryImpl()

    class _Broken:
        def available(self):
            return True

        def capabilities(self):
            raise RuntimeError("boom")

        def health(self):
            raise RuntimeError("boom")

    registry.register("good", "tabular", _StubProvider("good", "tabular"))
    registry.register("broken", "tabular", _Broken())
    caps = registry.capabilities()
    assert "good" in caps and "read" in str(caps["good"]["features"])
    assert "error" in caps["broken"]
    health = registry.health()
    assert health["good"]["available"] is True
    assert health["broken"]["available"] is False


def test_discovery_records():
    registry = ProviderRegistryImpl()
    registry.register("a", "tabular", _StubProvider("a", "tabular"), priority=7)
    records = registry.discovery()
    assert records[0]["name"] == "a"
    assert records[0]["kind"] == "tabular"
    assert records[0]["enabled"] is True
    assert records[0]["priority"] == 7
    assert records[0]["config"] == {}


def test_register_requires_name_and_kind():
    registry = ProviderRegistryImpl()
    provider = _StubProvider("a", "tabular")
    with pytest.raises(ValueError):
        registry.register("", "tabular", provider)
    with pytest.raises(ValueError):
        registry.register("a", "", provider)
