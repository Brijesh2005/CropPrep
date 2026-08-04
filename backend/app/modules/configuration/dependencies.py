"""Configuration module dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

from app.dependencies.container import get_container
from app.modules.configuration.service import ConfigurationService


def get_configuration_service(container: Any = Depends(get_container)) -> ConfigurationService:
    return ConfigurationService(container.config.resolve("settings"))
