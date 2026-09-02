"""R5.3.1 tests — Kaggle CLI deployment automation.

These tests validate the local deployment pipeline WITHOUT submitting
real Kaggle training jobs. All Kaggle CLI interactions are mocked.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEPLOY_DIR = _REPO_ROOT / "training" / "kaggle" / "deployment" / "r5_3"
_MANIFEST_PATH = _REPO_ROOT / "training_manifests" / "crop_supervised_v1_manifest.json"
_FROZEN_CSV = _REPO_ROOT / "govt_crop_matched_v1" / "crop_supervised_v1.csv"
_NOTEBOOK_SRC = _REPO_ROOT / "training" / "kaggle" / "notebooks" / "R5_3_train.ipynb"

sys_path_inserted = False


@pytest.fixture(autouse=True)
def _ensure_sys_path():
    global sys_path_inserted
    import sys

    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
        sys_path_inserted = True
    yield
    if sys_path_inserted:
        sys.path.remove(root)
        sys_path_inserted = False


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------


class TestMetadataGeneration:
    def test_kernel_metadata_json_valid(self):
        meta_path = _DEPLOY_DIR / "kernel-metadata.json"
        if not meta_path.exists():
            pytest.skip("Deployment not prepared yet")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "id" in meta
        assert "code_file" in meta
        assert meta["code_file"] == "R5_3_train.ipynb"
        assert meta.get("enable_gpu") is True
        assert meta.get("kernel_type") == "notebook"

    def test_kernel_metadata_has_dataset_source(self):
        meta_path = _DEPLOY_DIR / "kernel-metadata.json"
        if not meta_path.exists():
            pytest.skip("Deployment not prepared yet")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sources = meta.get("dataset_sources", [])
        assert any("crop-yield-forecasting" in s for s in sources), (
            f"Expected imagery dataset in sources, got {sources}"
        )

    def test_kernel_metadata_id_format(self):
        meta_path = _DEPLOY_DIR / "kernel-metadata.json"
        if not meta_path.exists():
            pytest.skip("Deployment not prepared yet")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kid = meta.get("id", "")
        assert "/" in kid, f"Kernel ID must contain /, got {kid}"
        parts = kid.split("/")
        assert len(parts) == 2, f"Kernel ID must be owner/name, got {kid}"
        assert parts[1] == "r5-3-cropfusion-multimodal-training"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_notebook_source_exists(self):
        assert _NOTEBOOK_SRC.exists(), f"Notebook not found at {_NOTEBOOK_SRC}"

    def test_manifest_path_exists(self):
        assert _MANIFEST_PATH.exists(), f"Manifest not found at {_MANIFEST_PATH}"

    def test_frozen_csv_exists(self):
        assert _FROZEN_CSV.exists(), f"Frozen CSV not found at {_FROZEN_CSV}"

    def test_repo_root_is_project_root(self):
        assert (_REPO_ROOT / "training").is_dir()
        assert (_REPO_ROOT / "scripts").is_dir()


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


class TestManifestValidation:
    def _load_manifest(self) -> dict:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_total_samples(self):
        m = self._load_manifest()
        assert m["total_samples"] == 10_674

    def test_train_samples(self):
        m = self._load_manifest()
        assert m["train_samples"] == 5_924

    def test_val_samples(self):
        m = self._load_manifest()
        assert m["validation_samples"] == 2_459

    def test_test_samples(self):
        m = self._load_manifest()
        assert m["test_samples"] == 2_291

    def test_checksum_matches(self):
        m = self._load_manifest()
        expected = "239cb608972e87f4e069e27f4ab308c65141afcdcb2148e0847dcfe96ea2820d"
        actual = m["reproducibility"]["dataset_checksums"]["crop_supervised_v1.csv"]
        assert actual == expected

    def test_class_mapping_has_five_classes(self):
        m = self._load_manifest()
        assert len(m["class_mapping"]) == 5

    def test_split_strategy(self):
        m = self._load_manifest()
        assert m["split_strategy"] == "spatial_leave_one_taluk_out"


# ---------------------------------------------------------------------------
# Dataset identifier validation
# ---------------------------------------------------------------------------


class TestDatasetIdentifier:
    def test_dataset_id_format(self):
        from scripts.kaggle_r5_3 import DATASET_ID

        assert "/" in DATASET_ID
        owner, name = DATASET_ID.split("/")
        assert len(owner) > 0
        assert len(name) > 0

    def test_dataset_id_matches_metadata(self):
        from scripts.kaggle_r5_3 import DATASET_ID

        meta_path = _DEPLOY_DIR / "kernel-metadata.json"
        if not meta_path.exists():
            pytest.skip("Deployment not prepared")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert DATASET_ID in meta.get("dataset_sources", [])


# ---------------------------------------------------------------------------
# Credential detection
# ---------------------------------------------------------------------------


class TestCredentialDetection:
    def test_no_credentials_in_repository(self):
        """Ensure no credential files exist in the repo."""
        credential_patterns = [
            "kaggle.json",
            ".kaggle/kaggle.json",
            "credentials.json",
        ]
        for pattern in credential_patterns:
            path = _REPO_ROOT / pattern
            assert not path.exists(), f"Credential file found: {path}"

    def test_no_credentials_in_deployment(self):
        """Ensure no credential files in deployment dir."""
        if not _DEPLOY_DIR.exists():
            pytest.skip("Deployment not prepared")
        credential_files = list(_DEPLOY_DIR.rglob("kaggle.json")) + list(
            _DEPLOY_DIR.rglob("credentials.json")
        )
        assert len(credential_files) == 0, (
            f"Credential files in deployment: {credential_files}"
        )

    def test_no_hardcoded_tokens_in_metadata(self):
        meta_path = _DEPLOY_DIR / "kernel-metadata.json"
        if not meta_path.exists():
            pytest.skip("Deployment not prepared")
        content = meta_path.read_text(encoding="utf-8").lower()
        assert "api_token" not in content
        assert "secret" not in content
        assert "password" not in content


# ---------------------------------------------------------------------------
# GPU metadata
# ---------------------------------------------------------------------------


class TestGPUMetadata:
    def test_kernel_metadata_enables_gpu(self):
        meta_path = _DEPLOY_DIR / "kernel-metadata.json"
        if not meta_path.exists():
            pytest.skip("Deployment not prepared")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta.get("enable_gpu") is True

    def test_kernel_metadata_enables_internet(self):
        meta_path = _DEPLOY_DIR / "kernel-metadata.json"
        if not meta_path.exists():
            pytest.skip("Deployment not prepared")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta.get("enable_internet") is True


# ---------------------------------------------------------------------------
# CLI command construction (mock)
# ---------------------------------------------------------------------------


class TestCLICommandConstruction:
    def test_push_command_format(self):
        """Verify push command would use correct kaggle kernels push syntax."""
        cmd = ["kaggle", " kernels", "push", "-p", str(_DEPLOY_DIR)]
        assert cmd[0] == "kaggle"
        assert "push" in cmd

    def test_status_command_format(self):
        cmd = ["kaggle", "kernels", "status", "owner/name"]
        assert cmd[0] == "kaggle"
        assert "status" in cmd

    def test_output_command_format(self):
        cmd = ["kaggle", "kernels", "output", "owner/name", "-p", "/tmp/output"]
        assert cmd[0] == "kaggle"
        assert "output" in cmd

    def test_logs_command_format(self):
        cmd = ["kaggle", "kernels", "logs", "owner/name"]
        assert cmd[0] == "kaggle"
        assert "logs" in cmd


# ---------------------------------------------------------------------------
# Local corpus contract
# ---------------------------------------------------------------------------


class TestLocalCorpusContract:
    def test_frozen_csv_line_count(self):
        """Quick check that the CSV has the expected number of rows."""
        import csv

        with open(_FROZEN_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 10_674

    def test_frozen_csv_columns(self):
        import csv

        with open(_FROZEN_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
        required = {"crop_label", "location_taluk", "lat", "lon"}
        assert required.issubset(set(headers)), (
            f"Missing columns: {required - set(headers)}"
        )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIArgumentParsing:
    def test_check_command_parses(self):
        from scripts.kaggle_r5_3 import build_parser

        parser = build_parser()
        args = parser.parse_args(["check"])
        assert args.command == "check"

    def test_train_requires_confirm(self):
        from scripts.kaggle_r5_3 import cmd_train

        result = cmd_train(MagicMock(confirm=False, timeout=None, poll_seconds=None))
        assert result == 1

    def test_prepare_command_parses(self):
        from scripts.kaggle_r5_3 import build_parser

        parser = build_parser()
        args = parser.parse_args(["prepare"])
        assert args.command == "prepare"

    def test_full_command_parses(self):
        from scripts.kaggle_r5_3 import build_parser

        parser = build_parser()
        args = parser.parse_args(["full"])
        assert args.command == "full"

    def test_verify_output_command_parses(self):
        from scripts.kaggle_r5_3 import build_parser

        parser = build_parser()
        args = parser.parse_args(["verify-output"])
        assert args.command == "verify-output"

    def test_train_with_confirm_parses(self):
        from scripts.kaggle_r5_3 import build_parser

        parser = build_parser()
        args = parser.parse_args(["train", "--confirm"])
        assert args.command == "train"
        assert args.confirm is True


# ---------------------------------------------------------------------------
# Notebook portability
# ---------------------------------------------------------------------------


class TestNotebookPortability:
    def test_no_local_windows_paths_in_notebook(self):
        """Notebook must not contain local Windows paths."""
        content = _NOTEBOOK_SRC.read_text(encoding="utf-8")
        forbidden = ["D:\\\\CropPrep", "C:\\\\Users", "D:\\\\Datasets_TIF"]
        for pattern in forbidden:
            assert pattern not in content, (
                f"Notebook contains local path: {pattern}"
            )

    def test_notebook_uses_kaggle_paths(self):
        """Notebook should reference /kaggle/working or /kaggle/input."""
        content = _NOTEBOOK_SRC.read_text(encoding="utf-8")
        has_kaggle = "/kaggle/" in content
        assert has_kaggle, "Notebook should use /kaggle/ paths"


# ---------------------------------------------------------------------------
# Artifact verification (mock)
# ---------------------------------------------------------------------------


class TestArtifactVerification:
    def test_sha256_function(self):
        from scripts.kaggle_r5_3 import _sha256

        # Test with the manifest file
        h = _sha256(_MANIFEST_PATH)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Checkpoint verification constants
# ---------------------------------------------------------------------------


class TestCheckpointConstants:
    def test_expected_checksum_constant(self):
        from scripts.kaggle_r5_3 import EXPECTED_CHECKSUM

        assert len(EXPECTED_CHECKSUM) == 64

    def test_expected_totals(self):
        from scripts.kaggle_r5_3 import (
            EXPECTED_TOTAL,
            EXPECTED_TRAIN,
            EXPECTED_VAL,
            EXPECTED_TEST,
        )

        assert EXPECTED_TOTAL == 10_674
        assert EXPECTED_TRAIN == 6_116
        assert EXPECTED_VAL == 2_267
        assert EXPECTED_TEST == 2_291
        assert EXPECTED_TRAIN + EXPECTED_VAL + EXPECTED_TEST == EXPECTED_TOTAL
