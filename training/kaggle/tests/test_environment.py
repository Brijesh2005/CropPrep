"""R2.1 environment detection tests: runtime / system / GPU / dependencies."""

from __future__ import annotations

from pathlib import Path

from training.kaggle.environment import EnvironmentManager
from training.kaggle.environment.dependencies import detect_dependencies
from training.kaggle.environment.gpu import detect_gpu
from training.kaggle.environment.runtime import detect_runtime
from training.kaggle.environment.system import detect_disk, detect_system


def test_runtime_detection() -> None:
    report = detect_runtime()
    assert isinstance(report["is_kaggle"], bool)
    assert "input_dir" in report


def test_gpu_detection_never_raises() -> None:
    report = detect_gpu()
    assert report["available"] in (True, False)
    assert "devices" in report
    assert isinstance(report["device_count"], int)


def test_system_detection() -> None:
    report = detect_system()
    assert report["python_version"]
    assert report["cpu_count"] and report["cpu_count"] > 0
    assert report["ram_total_gb"] is None or report["ram_total_gb"] > 0


def test_disk_detection(tmp_path: Path) -> None:
    disk = detect_disk(tmp_path)
    assert disk["path"] == str(tmp_path)
    assert disk["free_gb"] is None or disk["free_gb"] > 0


def test_dependency_detection() -> None:
    deps = detect_dependencies(["numpy", "pandas"])
    assert deps["numpy"]["installed"] is True
    assert deps["numpy"]["version"]
    assert deps["pandas"]["import"] == "pandas"


def test_dependency_alias_mapping() -> None:
    deps = detect_dependencies(["scikit-learn", "opencv-python", "PyYAML"])
    assert deps["scikit-learn"]["import"] == "sklearn"
    assert deps["opencv-python"]["import"] == "cv2"
    assert deps["PyYAML"]["import"] == "yaml"


def test_manager_report_structure(tmp_path: Path) -> None:
    manager = EnvironmentManager(repo_root=str(tmp_path))
    report = manager.report(requirements=["numpy"])
    assert set(("runtime", "system", "gpu", "dependencies", "capable")) <= set(report)
    assert report["capable"]["gpu"] in (True, False)
    assert "missing_dependencies" in report["capable"]
    assert report["system"]["disk"]["path"] == str(tmp_path.resolve())
