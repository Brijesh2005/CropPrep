"""Tests for the export package."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from training.export import (
    ExportConfig,
    ExportError,
    ExportFormatError,
    attach_meta,
    export_dataset,
    frame_to_records,
    load_export_config,
    save_export_template,
)


class TestConfig:
    def test_defaults(self):
        config = ExportConfig()
        assert config.formats == ["json", "parquet"]
        assert config.prefix == "cropfusion"
        assert config.include_meta is True

    def test_unknown_format_rejected(self):
        with pytest.raises(Exception, match="Unsupported export format"):
            ExportConfig(formats=["csv"])

    def test_load_from_env(self):
        config = load_export_config(env={"EX_PREFIX": "my-ds", "EX_FORMATS": '["torch"]'})
        assert config.prefix == "my-ds"
        assert config.formats == ["torch"]

    def test_load_from_yaml(self, tmp_path):
        path = tmp_path / "export.yml"
        save_export_template(path)
        config = load_export_config(path)
        assert config.prefix == "cropfusion"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(ExportError):
            load_export_config(tmp_path / "missing.yml")


class TestRecords:
    def test_nan_becomes_none(self):
        frame = pd.DataFrame({"a": [1.0, float("nan")], "b": ["x", None]})
        records = frame_to_records(frame)
        assert records[1]["a"] is None
        assert records[1]["b"] is None

    def test_attach_meta_aligns_corpus(self, corpus, frame):
        out = attach_meta(frame, corpus)
        accepted = [s for s in corpus.samples if s.status == "accepted"]
        assert "sample_id" in out.columns
        assert "quality_score" in out.columns
        first_id = out.iloc[0]["sample_id"]
        assert first_id == getattr(accepted[0], "sample_id", None) or first_id is None

    def test_attach_meta_without_corpus(self, frame):
        out = attach_meta(frame, None)
        assert len(out) == len(frame)


class TestExportDataset:
    def test_writes_requested_formats(self, frame, tmp_path):
        config = ExportConfig(
            output_dir=str(tmp_path),
            formats=["json", "parquet"],
            prefix="demo",
        )
        artifacts = export_dataset(frame, config=config)
        assert (tmp_path / "demo.json").exists()
        assert (tmp_path / "demo.parquet").exists()
        assert set(artifacts) == {"json", "parquet"}

    def test_writes_torch(self, frame, tmp_path):
        config = ExportConfig(output_dir=str(tmp_path), formats=["torch"])
        artifacts = export_dataset(frame, config=config)
        assert (tmp_path / "cropfusion.pt").exists()
        assert "torch" in artifacts

    def test_json_records_valid(self, frame, tmp_path):
        config = ExportConfig(output_dir=str(tmp_path), formats=["json"])
        export_dataset(frame, config=config)
        payload = json.loads((tmp_path / "cropfusion.json").read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        assert len(payload) == len(frame)

    def test_parquet_roundtrip(self, frame, tmp_path):
        config = ExportConfig(output_dir=str(tmp_path), formats=["parquet"])
        export_dataset(frame, config=config)
        restored = pd.read_parquet(tmp_path / "cropfusion.parquet")
        assert len(restored) == len(frame)

    def test_manifest_written(self, frame, tmp_path):
        config = ExportConfig(output_dir=str(tmp_path), formats=["json"])
        export_dataset(frame, config=config)
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["rows"] == len(frame)
        assert "json" in manifest["formats"]

    def test_unsupported_format_raises(self, frame, tmp_path):
        config = ExportConfig.model_construct(
            output_dir=str(tmp_path), formats=["csv"], include_meta=False
        )
        with pytest.raises(ExportFormatError):
            export_dataset(frame, config=config)

    def test_meta_attached_when_corpus_given(self, frame, corpus, tmp_path):
        config = ExportConfig(output_dir=str(tmp_path), formats=["json"])
        export_dataset(frame, corpus=corpus, config=config)
        payload = json.loads((tmp_path / "cropfusion.json").read_text(encoding="utf-8"))
        assert "quality_score" in payload[0]

    def test_meta_omitted_when_disabled(self, frame, tmp_path):
        config = ExportConfig(output_dir=str(tmp_path), formats=["json"], include_meta=False)
        export_dataset(frame, config=config)
        payload = json.loads((tmp_path / "cropfusion.json").read_text(encoding="utf-8"))
        assert "quality_score" not in payload[0]


class TestTorchPayload:
    def test_payload_shape(self, frame, tmp_path):
        config = ExportConfig(output_dir=str(tmp_path), formats=["torch"], include_meta=False)
        export_dataset(frame, config=config)
        import torch

        payload = torch.load(tmp_path / "cropfusion.pt", map_location="cpu")
        assert payload["features"].shape[0] == len(frame)
        assert payload["n_samples"] == len(frame)
        assert payload["feature_names"]
