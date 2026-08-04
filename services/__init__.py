"""CropFusion shared service code.

``services`` is a namespace-style container for cross-cutting service code.
In Phase 2 it hosts the **Dataset Management System**:

* :mod:`services.dataset_manager` — the only module allowed to touch datasets
  (download, scan, validate, metadata, versioning, caching, CSV/GeoTIFF reads).

Future phases add further service packages (``gis_service``, ``inference``,
``api``) under the same container.
"""

from __future__ import annotations

__version__ = "0.1.0"
