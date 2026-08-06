"""Model exporter for the inference package (Phase R5).

Wraps the Phase 5 :class:`~training.models.exporter.ModelExporter` (TorchScript
trace + ONNX) and adds the PyTorch format the inference package is built
around: a self-describing ``.pt`` file holding the config, the state dict and
export metadata. :meth:`ModelExporter.export_bundle` produces all three formats
plus ``model_config.yaml``, ``metrics.json``, ``metadata.json`` and
``checksums.json`` in one call.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .config import InferenceConfig
from .exceptions import ExportError
from .versioning import file_sha256, model_fingerprint

#: All model files written by :meth:`ModelExporter.export_bundle`.
BUNDLE_FORMAT_FILES: dict[str, str] = {
    "pytorch": "cropfusion.pt",
    "torchscript": "cropfusion.torchscript.pt",
    "onnx": "cropfusion.onnx",
}


class ModelExporter:
    """Export a :class:`CropFusionModel` to PyTorch / TorchScript / ONNX.

    Args:
        model: A trained :class:`CropFusionModel`.
        sample_batch: Batch dict used to derive example inputs (defaults to
            :meth:`CropFusionModel.sample_batch`).
    """

    def __init__(self, model: nn.Module, sample_batch: Mapping[str, Any] | None = None) -> None:
        self.model = model
        self.sample_batch = (
            dict(sample_batch) if sample_batch is not None else None
        )

    # ------------------------------------------------------------------ #
    # PyTorch
    # ------------------------------------------------------------------ #

    def export_pytorch(
        self, path: str | Path, *, metadata: Mapping[str, Any] | None = None
    ) -> Path:
        """Export the model as a self-describing PyTorch file.

        The saved payload is ``{"format", "format_version", "config",
        "state_dict", "metadata"}`` so a consumer can rebuild the architecture
        and load the weights without the training package.

        Raises:
            ExportError: When serialisation fails.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        config = getattr(self.model, "config", None)
        if config is None:
            raise ExportError(
                "PyTorch export requires a model exposing a ModelConfig"
            )
        payload = {
            "format": "cropfusion-pytorch",
            "format_version": 1,
            "config": config.model_dump(),
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in self.model.state_dict().items()
            },
            "metadata": {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "model_name": config.name,
                "model_version": config.version,
                "architecture_version": getattr(config, "architecture_version", None),
                "parameter_count": sum(
                    p.numel() for p in self.model.parameters()
                ),
                "fingerprint": model_fingerprint(self.model),
                **(dict(metadata) if metadata else {}),
            },
        }
        try:
            torch.save(payload, out)
        except Exception as exc:  # pragma: no cover - IO / pickling failure
            raise ExportError(
                f"PyTorch export failed: {exc}", detail=str(out)
            ) from exc
        return out

    # ------------------------------------------------------------------ #
    # TorchScript / ONNX
    # ------------------------------------------------------------------ #

    def export_torchscript(
        self, path: str | Path, *, mode: str | None = None
    ) -> Path:
        """Export a traced TorchScript module (delegates to the Phase 5
        exporter)."""
        from training.models.exporter import ModelExporter as _ModelExporter

        try:
            exporter = _ModelExporter(self.model, self.sample_batch)
            return exporter.export_torchscript(path, mode=mode)
        except Exception as exc:
            if hasattr(exc, "code") and exc.code.startswith("MDL-"):
                raise
            raise ExportError(
                f"TorchScript export failed: {exc}", detail=str(path)
            ) from exc

    def export_onnx(
        self,
        path: str | Path,
        *,
        opset: int | None = None,
        dynamic_batch: bool = True,
    ) -> Path:
        """Export an ONNX graph (delegates to the Phase 5 exporter)."""
        from training.models.exporter import ModelExporter as _ModelExporter

        try:
            exporter = _ModelExporter(self.model, self.sample_batch)
            return exporter.export_onnx(
                path, opset=opset, dynamic_batch=dynamic_batch
            )
        except Exception as exc:
            if hasattr(exc, "code") and exc.code.startswith("MDL-"):
                raise
            raise ExportError(
                f"ONNX export failed: {exc}", detail=str(path)
            ) from exc

    # ------------------------------------------------------------------ #
    # Bundle
    # ------------------------------------------------------------------ #

    def export_bundle(
        self,
        directory: str | Path,
        *,
        config: InferenceConfig | None = None,
        metrics: Mapping[str, Any] | None = None,
        versions: Any = None,
        training_config: Mapping[str, Any] | None = None,
        dataset_fingerprint: str | None = None,
        git_commit_sha: str | None = None,
    ) -> dict[str, Path]:
        """Export every requested format + the metadata sidecars.

        Returns a mapping of artifact key -> written path.
        """
        from .versioning import resolve_versions

        inference_cfg = config or InferenceConfig()
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        formats = inference_cfg.exporter.formats

        paths: dict[str, Path] = {}
        if "pytorch" in formats:
            paths["pytorch"] = self.export_pytorch(out_dir / BUNDLE_FORMAT_FILES["pytorch"])
        if "torchscript" in formats:
            paths["torchscript"] = self.export_torchscript(
                out_dir / BUNDLE_FORMAT_FILES["torchscript"],
                mode=inference_cfg.exporter.torchscript_mode,
            )
        if "onnx" in formats:
            paths["onnx"] = self.export_onnx(
                out_dir / BUNDLE_FORMAT_FILES["onnx"],
                opset=inference_cfg.exporter.onnx_opset,
            )

        model_cfg = getattr(self.model, "config", None)
        if model_cfg is not None:
            yaml_path = out_dir / "model_config.yaml"
            yaml_path.write_text(
                model_cfg.to_yaml() if hasattr(model_cfg, "to_yaml") else _dump_yaml(model_cfg),
                encoding="utf-8",
            )
            paths["model_config"] = yaml_path

        if metrics is not None:
            paths["metrics"] = _write_json(out_dir / "metrics.json", metrics)

        if training_config is not None:
            paths["training_config"] = _write_json(
                out_dir / "training_config.json", training_config
            )

        resolved = resolve_versions(
            package_version=inference_cfg.general.version,
            model_version=inference_cfg.general.model_version,
            dataset_version=inference_cfg.general.dataset_version,
            model_config_version=getattr(model_cfg, "version", "1.0.0")
            if model_cfg else "1.0.0",
        )
        fp = model_fingerprint(self.model)
        metadata = {
            "package_name": inference_cfg.general.package_name,
            "package_version": str(resolved.package_version),
            "model_version": str(resolved.model_version),
            "dataset_version": str(resolved.dataset_version),
            "model_fingerprint": fp,
            "dataset_fingerprint": dataset_fingerprint,
            "git_commit": git_commit_sha or self._git_commit(),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "formats": sorted(paths.keys() - {"model_config", "metrics", "training_config"}),
            "architecture": (
                {"name": model_cfg.name, "version": model_cfg.version}
                if model_cfg else None
            ),
        }
        paths["metadata"] = _write_json(out_dir / "metadata.json", metadata)

        checksums = {
            path.name: file_sha256(path)
            for path in paths.values()
            if path.exists()
        }
        paths["checksums"] = _write_json(out_dir / "checksums.json", checksums)
        return paths

    @staticmethod
    def _git_commit() -> str | None:
        from .versioning import git_commit

        return git_commit()


def _dump_yaml(model_cfg: Any) -> str:
    import yaml

    return yaml.safe_dump(
        json.loads(json.dumps(model_cfg.model_dump(), default=str)),
        sort_keys=False,
    )


def _write_json(path: Path, data: Any) -> Path:
    path.write_text(
        json.dumps(data, indent=2, default=_json_default), encoding="utf-8"
    )
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"not JSON serializable: {type(value)}")


def load_pytorch_model(path: str | Path) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a ``CropFusionModel`` from a PyTorch export file.

    Args:
        path: A file written by :meth:`ModelExporter.export_pytorch`.

    Returns:
        ``(model, metadata)`` with the model in ``eval()`` mode.
    """
    from training.models import ModelConfig, ModelFactory

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "cropfusion-pytorch":
        raise ExportError(
            "not a CropFusion PyTorch export file",
            detail=str(path),
        )
    config = ModelConfig.model_validate(payload["config"])
    model = ModelFactory.create(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("metadata", {})


__all__ = [
    "BUNDLE_FORMAT_FILES",
    "ModelExporter",
    "load_pytorch_model",
]
