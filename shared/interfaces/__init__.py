"""Reusable ports (abstract interfaces) for the CropFusion shared framework.

These ports let both the Training and Application platforms depend on
abstractions (dependency inversion) without importing each other:

* :class:`Provider` / :class:`DatasetProvider` / :class:`TabularProvider` /
  :class:`ImageProvider`  — data acquisition and access.
* :class:`Repository` / :class:`Cache` / :class:`Storage` — persistence.
* :class:`ModelExporter` / :class:`Logger` / :class:`ConfigurationProvider` /
  :class:`Serializer` / :class:`VersionProvider` — platform services.

Platform-specific ports (e.g. the Dataset Manager's ``Downloader``,
``Scanner``, ``Validator``) remain defined in their owning package and may be
aliased to these shared ports over time.
"""

from __future__ import annotations

from .data import Cache, Repository, Storage
from .platform import (
    ConfigurationProvider,
    Logger,
    ModelExporter,
    Serializer,
    VersionProvider,
)
from .providers import (
    DatasetProvider,
    ImageProvider,
    Provider,
    TabularProvider,
)

__all__ = [
    "Cache",
    "ConfigurationProvider",
    "DatasetProvider",
    "ImageProvider",
    "Logger",
    "ModelExporter",
    "Provider",
    "Repository",
    "Serializer",
    "Storage",
    "TabularProvider",
    "VersionProvider",
]
