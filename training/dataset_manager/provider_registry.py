"""Provider registry — the single place providers are registered and resolved.

The R2.2 provider architecture is strict:

    DatasetManager --> ProviderRegistry --> providers

The manager **never constructs or holds providers directly** (the legacy
``tabular_provider`` / ``image_provider`` attributes are resolved *through*
the registry to keep the R1.2 surface). Consumers resolve a provider by name
or by kind; the registry answers health, capability, priority, availability
and discovery queries.

Future plugins (a third data source, a fallback image provider, ...) plug in
by registering an instance — nothing else needs to change.
"""

from __future__ import annotations

from typing import Any

from .exceptions import DatasetNotFoundError
from .interfaces import ProviderRegistry
from .logger import get_logger
from .providers.models import ProviderRegistration

logger = get_logger("provider_registry")


class ProviderRegistryImpl(ProviderRegistry):
    """Concrete :class:`ProviderRegistry`.

    Providers are stored by name. Resolution honours ``enabled`` and sorts
    providers of a kind by ``priority`` (higher wins) so a high-priority
    provider shadows lower-priority fallbacks of the same kind.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ProviderRegistration] = {}

    # -- Registration ---------------------------------------------------------- #

    def register(
        self,
        name: str,
        kind: str,
        provider: Any,
        *,
        enabled: bool = True,
        priority: int = 100,
        config: dict[str, Any] | None = None,
    ) -> ProviderRegistration:
        """Register a provider instance under ``name``.

        Re-registering an existing name replaces the previous registration.
        """
        if not name or not kind:
            raise ValueError("Provider registration requires a name and a kind")
        registration = ProviderRegistration(
            name=name,
            kind=kind,
            provider=provider,
            enabled=bool(enabled),
            priority=int(priority),
            config=dict(config or {}),
        )
        self._providers[name] = registration
        logger.info(
            "Registered provider",
            extra={
                "provider_name": name,
                "kind": kind,
                "enabled": enabled,
                "priority": priority,
            },
        )
        return registration

    # -- Resolution ------------------------------------------------------------ #

    def resolve(self, name: str) -> Any:
        """Return the live provider instance for ``name``.

        Raises:
            DatasetNotFoundError: When the provider is unknown or disabled.
        """
        registration = self._providers.get(name)
        if registration is None:
            raise DatasetNotFoundError(
                f"Provider not registered: {name}",
                detail={"registered": sorted(self._providers)},
            )
        if not registration.enabled:
            raise DatasetNotFoundError(
                f"Provider is disabled: {name}", detail={"name": name}
            )
        return registration.provider

    def resolve_by_kind(self, kind: str) -> list[Any]:
        """Enabled providers of ``kind``, sorted by priority (highest first)."""
        matches = [
            r
            for r in self._providers.values()
            if r.kind == kind and r.enabled
        ]
        matches.sort(key=lambda r: r.priority, reverse=True)
        return [r.provider for r in matches]

    # -- Introspection --------------------------------------------------------- #

    def names(self) -> list[str]:
        return sorted(self._providers)

    def registrations(self) -> list[ProviderRegistration]:
        """All registrations (enabled and disabled), by name."""
        return [self._providers[n] for n in self.names()]

    def has(self, name: str) -> bool:
        return name in self._providers

    def priority(self, name: str) -> int:
        registration = self._providers.get(name)
        if registration is None:
            raise DatasetNotFoundError(
                f"Provider not registered: {name}",
                detail={"registered": sorted(self._providers)},
            )
        return registration.priority

    def availability(self) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for name, registration in self._providers.items():
            if not registration.enabled:
                out[name] = False
                continue
            try:
                out[name] = bool(registration.provider.available())
            except Exception:  # noqa: BLE001 - availability is best-effort
                out[name] = False
        return out

    def capabilities(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, registration in self._providers.items():
            try:
                caps = registration.provider.capabilities()
                out[name] = caps.to_dict() if hasattr(caps, "to_dict") else dict(caps)
            except Exception as exc:  # noqa: BLE001 - best-effort
                out[name] = {"error": str(exc)}
        return out

    def health(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, registration in self._providers.items():
            if not registration.enabled:
                out[name] = {"name": name, "enabled": False, "available": False}
                continue
            try:
                snapshot = registration.provider.health()
                out[name] = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
            except Exception as exc:  # noqa: BLE001 - best-effort
                out[name] = {"name": name, "error": str(exc), "available": False}
        return out

    def discovery(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._providers.values()]
