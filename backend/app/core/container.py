"""Dependency-injection container.

A lightweight, hand-rolled provider registry that wires the application
("dependency injector" without a third-party dependency). Supports singleton /
transient providers and test overrides.

Containers:
* :class:`ConfigurationContainer` — settings + derived components.
* :class:`RepositoryContainer` — database session + repositories.
* :class:`ModelContainer` — dataset manager, STAM, preprocessor, model and the
  inference engine.
* :class:`ServiceContainer` — application services.
* :class:`ApplicationContainer` — the composition root.
"""

from __future__ import annotations

from typing import Any, Callable


class Container:
    """A provider registry with lazy resolution and test overrides."""

    def __init__(self) -> None:
        self._providers: dict[str, tuple[Callable[[], Any], bool]] = {}
        self._instances: dict[str, Any] = {}
        self._overrides: dict[str, Any] = {}

    def register(self, name: str, factory: Callable[[], Any], *, singleton: bool = True) -> None:
        """Register a provider under ``name``."""
        self._providers[name] = (factory, singleton)

    def override(self, name: str, instance: Any) -> None:
        """Replace a provider with an instance (used by tests)."""
        self._overrides[name] = instance

    def clear_override(self, name: str) -> None:
        self._overrides.pop(name, None)

    def resolve(self, name: str) -> Any:
        """Resolve a provider, constructing (and caching) as needed."""
        if name in self._overrides:
            return self._overrides[name]
        if name in self._instances:
            return self._instances[name]
        if name not in self._providers:
            raise KeyError(f"no provider registered for {name!r}")
        factory, singleton = self._providers[name]
        instance = factory()
        if singleton:
            self._instances[name] = instance
        return instance

    def __getattr__(self, name: str) -> Any:
        return self.resolve(name)

    def has(self, name: str) -> bool:
        return name in self._overrides or name in self._instances or name in self._providers

    def registered(self) -> list[str]:
        return sorted(self._providers)
