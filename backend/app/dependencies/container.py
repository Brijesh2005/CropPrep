"""Access to the application's dependency-injection container."""

from __future__ import annotations

from typing import Any

from fastapi import Request


def get_container(request: Request) -> Any:
    """Resolve the application container attached to ``app.state``."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("application container is not initialised")
    return container


def get_service(request: Request, name: str) -> Any:
    """Resolve a named service from the container."""
    return get_container(request).services.resolve(name)


def get_model_container(request: Request) -> Any:
    return get_container(request).model
