"""Tests for shared interfaces (providers, data, platform contracts)."""

from __future__ import annotations

import abc

from shared.enums import ProviderType
from shared.interfaces import (
    Cache,
    ConfigurationProvider,
    DatasetProvider,
    ImageProvider,
    Logger,
    ModelExporter,
    Provider,
    Repository,
    Serializer,
    Storage,
    TabularProvider,
    VersionProvider,
)


def _abstract_names(cls) -> set[str]:
    names = set()
    for base in cls.__mro__:
        for name, member in vars(base).items():
            if getattr(member, "__isabstractmethod__", False):
                names.add(name)
    return names


def test_provider_defaults() -> None:
    assert Provider.name == "provider"
    assert DatasetProvider.provider_type is ProviderType.UNKNOWN
    assert TabularProvider.provider_type is ProviderType.TABULAR
    assert ImageProvider.provider_type is ProviderType.IMAGE


def test_provider_abstract_methods() -> None:
    assert _abstract_names(Provider) == {"health", "describe"}
    assert {"fetch", "exists", "version"} <= _abstract_names(DatasetProvider)
    assert {"discover", "load", "preview"} <= _abstract_names(TabularProvider)
    assert {"catalog", "read_metadata", "read_window", "iterate"} <= _abstract_names(ImageProvider)


def test_data_ports_abstract() -> None:
    assert _abstract_names(Repository) == {"save", "save_many", "get", "query", "count", "close"}
    assert _abstract_names(Cache) == {"get", "set", "delete", "delete_prefix", "clear", "prune"}
    assert _abstract_names(Storage) == {"exists", "read_bytes", "write_bytes", "delete", "list"}


def test_platform_ports_abstract() -> None:
    assert _abstract_names(ModelExporter) == {"export", "export_report"}
    assert _abstract_names(Logger) == {"get_logger", "setup"}
    assert _abstract_names(ConfigurationProvider) == {"get", "load"}
    assert _abstract_names(Serializer) == {"dump", "load"}
    assert _abstract_names(VersionProvider) == {"current", "bump"}


def test_all_ports_are_abstract() -> None:
    for cls in (Provider, DatasetProvider, TabularProvider, ImageProvider, Repository,
                Cache, Storage, ModelExporter, Logger, ConfigurationProvider, Serializer,
                VersionProvider):
        assert issubclass(cls, abc.ABC)
        assert cls.__abstractmethods__
