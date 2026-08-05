"""R1.4 skeleton tests: the inference-only architecture imports cleanly and is
free of Training Platform imports."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]  # application/


def _sources(*rel_dirs: str) -> list[Path]:
    out: list[Path] = []
    for rel in rel_dirs:
        out.extend((ROOT / rel).rglob("*.py"))
    return out


NEW_LAYERS = ("inference", "gis", "history", "inference_package", "models")

_TRAINING_IMPORT = re.compile(
    r"^\s*(import training\b|from training\b|import training\.|from training import)",
    re.MULTILINE,
)


def test_new_packages_import_cleanly() -> None:
    import application.gis as gis
    import application.gis.historical_context
    import application.gis.resolver
    import application.gis.reverse_geocoding
    import application.gis.spatial_resolver
    import application.history as history
    import application.inference as inference
    import application.inference.cache
    import application.inference.engine
    import application.inference.explainability
    import application.inference.loaders
    import application.inference.services
    import application.inference.validation
    import application.inference.versioning
    import application.inference_package as inference_package
    import application.models as models

    assert inference.__version__ == "0.1.0"
    assert gis.__version__ == "0.1.0"
    assert history.__version__ == "0.1.0"
    assert models.__version__ == "0.1.0"
    assert len(inference_package.INFERENCE_PACKAGE_FILES) >= 9


def test_no_training_imports_in_new_layers() -> None:
    sources = _sources(*NEW_LAYERS)
    assert sources, "expected new-layer python sources"
    offenders: list[str] = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        if _TRAINING_IMPORT.search(text):
            offenders.append(str(path))
    assert not offenders, f"training imports found in: {offenders}"


def test_prediction_request_is_location_only() -> None:
    from application.inference import PredictionRequest

    req = PredictionRequest(lon=74.9, lat=12.85)
    assert req.lon == 74.9
    assert req.lat == 12.85
    assert req.include_explanation is False
    # No year / season fields — they are auto-resolved by the GIS layer.
    assert not hasattr(req, "year")
    assert not hasattr(req, "season")


def test_prediction_request_validates_bounds() -> None:
    from application.inference import PredictionRequest

    with pytest.raises(ValueError):
        PredictionRequest(lon=181.0, lat=0.0)
    with pytest.raises(ValueError):
        PredictionRequest(lon=0.0, lat=-91.0)


def test_inference_package_manifest() -> None:
    from application.inference_package import INFERENCE_PACKAGE_FILES

    names = {a.filename for a in INFERENCE_PACKAGE_FILES}
    for expected in (
        "metadata.db",
        "historical_context.parquet",
        "location_index.parquet",
        "feature_scalers.pkl",
        "label_encoder.pkl",
        "model_config.yaml",
        "dataset_version.json",
        "model_version.json",
        "metrics.json",
    ):
        assert expected in names, f"missing manifest entry: {expected}"


def test_model_naming_convention_documented() -> None:
    from application.inference_package import (
        MODEL_WEIGHTS_DEFAULT_NAME,
        MODEL_WEIGHTS_FUTURE_PATTERN,
    )

    assert MODEL_WEIGHTS_DEFAULT_NAME == "cropfusion.pt"
    assert "v1" in MODEL_WEIGHTS_FUTURE_PATTERN.format(version="v1")


def test_gis_chain_contract() -> None:
    from application.gis.models import GeoPoint, GeoContext

    point = GeoPoint(lon=74.9, lat=12.85)
    ctx = GeoContext(point=point)
    d = ctx.to_dict()
    assert d["point"] == {"lon": 74.9, "lat": 12.85}
    assert d["historical"]["season"] == "unknown"
    with pytest.raises(ValueError):
        GeoPoint(lon=200.0, lat=0.0)


def test_history_store_contract() -> None:
    from application.history import HistoryFilters, PredictionHistoryStore

    filters = HistoryFilters(crop="rice", limit=10)
    assert filters.crop == "rice"
    assert filters.limit == 10
    # Port must remain abstract (no implementation in R1.4).
    with pytest.raises(TypeError):
        PredictionHistoryStore()
