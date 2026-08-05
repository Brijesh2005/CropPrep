"""Built-in serializer implementations (JSON / YAML / pickle / parquet / CSV / numpy / torch)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from ..exceptions import SerializationError
from ..utils import dump_yaml, load_yaml, yaml_safe
from .registry import Serializer


class JsonSerializer(Serializer):
    """JSON serializer using the shared default encoder (enums/paths/dates)."""

    name = "json"
    extensions = (".json",)

    def dump(self, data: Any, path: str | Path) -> Path:
        from ..utils import write_json

        return write_json(path, data)

    def load(self, path: str | Path) -> Any:
        from ..utils import read_json

        return read_json(path)


class YamlSerializer(Serializer):
    """YAML serializer (safe load/dump)."""

    name = "yaml"
    extensions = (".yaml", ".yml")

    def dump(self, data: Any, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(dump_yaml(data), encoding="utf-8")
        return out

    def load(self, path: str | Path) -> Any:
        return load_yaml(path)


class PickleSerializer(Serializer):
    """Pickle serializer."""

    name = "pickle"
    extensions = (".pkl", ".pickle")

    def dump(self, data: Any, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return out

    def load(self, path: str | Path) -> Any:
        with open(path, "rb") as fh:
            return pickle.load(fh)


class ParquetSerializer(Serializer):
    """Parquet serializer via pandas (optional dependency)."""

    name = "parquet"
    extensions = (".parquet", ".pq")

    def dump(self, data: Any, path: str | Path) -> Path:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise SerializationError(
                "Parquet serialization requires pandas", suggested_resolution="pip install pandas pyarrow"
            ) from exc
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data) if isinstance(data, dict) else pd.DataFrame([data])
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out, index=False)
        return out

    def load(self, path: str | Path) -> Any:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise SerializationError(
                "Parquet deserialization requires pandas", suggested_resolution="pip install pandas pyarrow"
            ) from exc
        return pd.read_parquet(path)


class CsvSerializer(Serializer):
    """CSV serializer via pandas (optional dependency)."""

    name = "csv"
    extensions = (".csv",)

    def dump(self, data: Any, path: str | Path) -> Path:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise SerializationError(
                "CSV serialization requires pandas", suggested_resolution="pip install pandas"
            ) from exc
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data) if isinstance(data, dict) else pd.DataFrame([data])
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(out, index=False)
        return out

    def load(self, path: str | Path) -> Any:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise SerializationError(
                "CSV deserialization requires pandas", suggested_resolution="pip install pandas"
            ) from exc
        return pd.read_csv(path)


class NumpySerializer(Serializer):
    """NumPy .npz archive serializer (optional dependency)."""

    name = "numpy"
    extensions = (".npz", ".npy")

    def dump(self, data: Any, path: str | Path) -> Path:
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise SerializationError(
                "NumPy serialization requires numpy", suggested_resolution="pip install numpy"
            ) from exc
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".npy":
            np.save(out, data)
        elif isinstance(data, dict):
            np.savez(out, **data)
        else:
            np.savez(out, data=data)
        return out

    def load(self, path: str | Path) -> Any:
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise SerializationError(
                "NumPy deserialization requires numpy", suggested_resolution="pip install numpy"
            ) from exc
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.ndarray):
            return loaded
        with loaded as archive:
            if "data" in archive and len(archive.files) == 1:
                return archive["data"]
            return {key: archive[key] for key in archive.files}


class TorchSerializer(Serializer):
    """PyTorch model/state-dict serializer (optional dependency)."""

    name = "torch"
    extensions = (".pt", ".pth", ".ckpt")

    def dump(self, data: Any, path: str | Path) -> Path:
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise SerializationError(
                "PyTorch serialization requires torch", suggested_resolution="pip install torch"
            ) from exc
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out)
        return out

    def load(self, path: str | Path) -> Any:
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise SerializationError(
                "PyTorch deserialization requires torch", suggested_resolution="pip install torch"
            ) from exc
        return torch.load(path, map_location="cpu")


def register_builtins(registry) -> None:
    """Register every built-in serializer into ``registry``."""
    for serializer in (
        JsonSerializer(),
        YamlSerializer(),
        PickleSerializer(),
        ParquetSerializer(),
        CsvSerializer(),
        NumpySerializer(),
        TorchSerializer(),
    ):
        registry.register(serializer)
