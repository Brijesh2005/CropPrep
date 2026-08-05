"""Tests for the shared serialization registry and formats."""

from __future__ import annotations

import numpy as np
import pytest

from shared.exceptions import SerializationError
from shared.serialization import (
    default_registry,
    dump,
    get_serializer,
    load,
    serializer_for_path,
)


@pytest.mark.parametrize(
    "name",
    ["csv", "json", "numpy", "parquet", "pickle", "torch", "yaml"],
)
def test_default_registry_has_builtins(name: str) -> None:
    assert name in default_registry.names()


def test_serializer_for_path_dispatches(tmp_path) -> None:
    assert serializer_for_path(tmp_path / "x.json").name == "json"
    assert serializer_for_path(tmp_path / "x.yaml").name == "yaml"
    assert serializer_for_path(tmp_path / "x.npy").name == "numpy"
    assert serializer_for_path(tmp_path / "x.csv").name == "csv"


def test_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "m.json"
    payload = {"a": 1, "b": [1, 2], "nested": {"x": "y"}}
    dump(payload, path)
    assert load(path) == payload


def test_yaml_roundtrip(tmp_path) -> None:
    path = tmp_path / "m.yaml"
    dump({"a": 1, "b": {"c": 2}}, path)
    assert load(path) == {"a": 1, "b": {"c": 2}}


def test_pickle_roundtrip(tmp_path) -> None:
    path = tmp_path / "m.pkl"
    payload = {"list": [1, 2, 3], "text": "hi"}
    dump(payload, path)
    assert load(path) == payload


def test_numpy_roundtrip(tmp_path) -> None:
    path = tmp_path / "arr.npy"
    dump(np.arange(6), path)
    loaded = load(path)
    assert isinstance(loaded, np.ndarray)
    assert loaded.tolist() == [0, 1, 2, 3, 4, 5]


def test_get_serializer_unknown_raises() -> None:
    with pytest.raises(SerializationError):
        get_serializer("does-not-exist")


def test_dump_unsupported_extension_raises(tmp_path) -> None:
    with pytest.raises(SerializationError):
        dump({"a": 1}, tmp_path / "x.unknown_ext")
