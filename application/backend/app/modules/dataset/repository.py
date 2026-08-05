"""Dataset module repository.

Dataset state is owned by the Phase 2 Dataset Manager (SQLite metadata store),
so this module exposes a thin repository over that manager rather than the app
database.
"""

from __future__ import annotations

from typing import Any


class DatasetRepository:
    """Reads dataset state through the Dataset Manager."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager

    def metadata_path(self) -> str | None:
        try:
            return str(self.manager.settings.metadata_db_path())
        except Exception:
            return None

    def registered_catalogs(self) -> list[str]:
        try:
            return list(self.manager.catalog_names() or [])
        except Exception:
            return []
