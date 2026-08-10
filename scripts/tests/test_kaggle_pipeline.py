"""Unit tests for the Kaggle pipeline drivers (network-free).

Covers ``scripts.run_kaggle_notebook`` (push / wait / log parsing / report
download plumbing) and ``scripts.run_full_pipeline`` (stage gating, the
checkpoint dataset handoff and the combined summary). All Kaggle interactions
are mocked; nothing here touches the network.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts import run_full_pipeline as p
from scripts.run_kaggle_notebook import (
    TERMINAL_FAIL,
    TERMINAL_OK,
    build_push_dir,
    parse_log,
    wait_for_completion,
)

# ---------------------------------------------------------------------------
# run_kaggle_notebook: pure helpers
# ---------------------------------------------------------------------------


class _EnumName:
    def __init__(self, name):
        self.name = name


class _Status:
    """Mirrors ApiGetKernelSessionStatusResponse: ``.status`` enum + message."""

    def __init__(self, name, failure_message=None):
        self.status = _EnumName(name)
        self.failure_message = failure_message


def test_parse_log_extracts_known_fields():
    log = (
        "[bootstrap] status -> READY\n"
        "[system_check] passed: True\n"
        "!python training/kaggle/scripts/system_check.py; echo "
        "system_check_exit=0\n"
    )
    info = parse_log(log)
    assert info["system_check_exit"] == 0
    assert info["passed"] is True
    assert info["bootstrap"] == "READY"


def test_parse_log_missing_fields_are_absent():
    info = parse_log("no markers here")
    assert info == {}


def test_wait_for_completion_success(monkeypatch):
    calls = []

    class _Api:
        def kernels_status(self, ref):
            calls.append(ref)
            return _Status("RUNNING" if len(calls) == 1 else "COMPLETE")

    monkeypatch.setattr("time.sleep", lambda *_: None)
    assert wait_for_completion(_Api(), "owner/kernel", 60, 1) == "COMPLETE"
    assert calls == ["owner/kernel", "owner/kernel"]


def test_wait_for_completion_failure_message(monkeypatch):
    class _Api:
        def kernels_status(self, ref):
            return _Status("ERROR", "out of memory")

    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="out of memory"):
        wait_for_completion(_Api(), "owner/kernel", 60, 1)


def test_wait_for_completion_timeout(monkeypatch):
    class _Api:
        def kernels_status(self, ref):
            return _Status("QUEUED")

    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(TimeoutError):
        wait_for_completion(_Api(), "owner/kernel", 0.05, 0.01)


def test_terminal_sets_cover_all_terminal_states():
    from kagglesdk.kernels.types import kernels_enums

    names = {
        m
        for m in dir(kernels_enums.KernelWorkerStatus)
        if not m.startswith("_")
    }
    assert {"COMPLETE"}.issubset(names)
    assert {"ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}.issubset(names)
    assert names == TERMINAL_OK | TERMINAL_FAIL | {"QUEUED", "RUNNING", "NEW_SCRIPT"}


def test_build_push_dir_writes_metadata_with_extra_datasets():
    notebook = {
        "slug": "cropfusion-system-check",
        "title": "CropFusion System Check",
        "file": "system_check.ipynb",
    }
    push_dir = build_push_dir(
        notebook, "testowner", extra_dataset_sources=("testowner/ckpt",),
        keep=True,
    )
    try:
        assert (push_dir / "system_check.ipynb").exists()
        meta = json.loads(
            (push_dir / "kernel-metadata.json").read_text(encoding="utf-8")
        )
        assert meta["id"] == "testowner/cropfusion-system-check"
        assert meta["dataset_sources"] == [
            "shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada",
            "testowner/ckpt",
        ]
    finally:
        shutil.rmtree(push_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# run_full_pipeline: report analysis helpers
# ---------------------------------------------------------------------------


def test_corpus_accepted_prefers_pipeline_report():
    assert p._corpus_accepted({"accepted": 7}, {"accepted": 9}) == 9
    assert p._corpus_accepted({"accepted": 7}, {}) == 7
    assert p._corpus_accepted({"accepted": None}, {"accepted": 3}) == 3


def test_corpus_breakdown_formats_fields():
    text = p._corpus_breakdown(
        {"total": 10, "accepted": 5, "rejected": 3, "errors": 2,
         "acceptance_rate": 0.5}
    )
    assert "accepted=5" in text
    assert "acceptance_rate=0.5" in text


def test_year_range_uses_accepted_samples_then_plan():
    samples = [
        {"status": "accepted", "year": 2018},
        {"status": "rejected", "year": 2019},
        {"status": "accepted", "year": 2025},
    ]
    assert p._year_range({"samples": samples}) == (2018, 2025)
    assert p._year_range({"plan": {"years": [2018, 2025]}}) == (2018, 2025)
    assert p._year_range({"config": {"years": []}, "plan": {}}) == (None, None)


def test_load_json_failures_are_stage_failures(tmp_path):
    with pytest.raises(p.StageFailure, match="was not produced"):
        p._load_json(None, "validation.json")
    bogus = tmp_path / "bogus.json"
    bogus.write_text("{not json", encoding="utf-8")
    with pytest.raises(p.StageFailure, match="unreadable/corrupt"):
        p._load_json(bogus, "validation.json")


# ---------------------------------------------------------------------------
# run_full_pipeline: stage gates
# ---------------------------------------------------------------------------


def _report(tmp_path, data, name="report.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _stage(reports, **extra):
    stage = {
        "name": "x",
        "run_id": "x-v1",
        "run_dir": Path.cwd(),
        "kernel_ref": "owner/x",
        "version": 1,
        "url": "https://www.kaggle.com/code/owner/x",
        "status": "COMPLETE",
        "reports": reports,
        "log": {},
    }
    stage.update(extra)
    return stage


def test_check_system_check_pass(tmp_path):
    validation = _report(
        tmp_path, {"passed": True, "by_severity": {"info": 2},
                   "issues": [{"code": "GPU_OK", "severity": "info"}]},
        "validation.json",
    )
    stage = _check_system_check_stage({"validation": validation})
    assert stage["validation"]["passed"] is True


def _check_system_check_stage(reports):
    return p._check_system_check(_stage(reports))


def test_check_system_check_fails_on_passed_false(tmp_path):
    validation = _report(
        tmp_path,
        {"passed": False, "issues": [{"code": "GPU_BAD", "message": "no gpu"}]},
        "validation.json",
    )
    with pytest.raises(p.StageFailure, match="GPU_BAD.*no gpu"):
        _check_system_check_stage({"validation": validation})


def test_check_system_check_fails_on_error_severity(tmp_path):
    validation = _report(
        tmp_path,
        {"passed": True, "issues": [{"code": "DISK_LOW", "severity": "error",
                                     "message": "1 GB free"}]},
        "validation.json",
    )
    with pytest.raises(p.StageFailure, match="DISK_LOW"):
        _check_system_check_stage({"validation": validation})


def test_check_system_check_missing_report():
    with pytest.raises(p.StageFailure, match="validation.json"):
        _check_system_check_stage({"validation": None})


def test_check_train_pass(tmp_path, monkeypatch):
    pipeline = _report(
        tmp_path,
        {
            "training": {"status": "completed", "accepted_observations": 5,
                         "run_dir": "/kaggle/working/CropPrep/training/artifacts/"
                                    "runs/test",
                         "report": {"evaluation": {"loss": 0.1}}},
            "corpus": {"accepted": 5},
        },
        "pipeline.json",
    )
    corpus = _report(
        tmp_path,
        {"total": 8, "accepted": 5, "rejected": 3, "errors": 0,
         "acceptance_rate": 0.625, "plan": {"years": [2018, 2025]},
         "samples": [{"status": "accepted", "year": 2020}]},
        "corpus.json",
    )
    ckpt = _report(
        tmp_path,
        {"found": True, "repo_relative": "training/artifacts/checkpoints/test/"
                                         "checkpoint.pt"},
        "checkpoint.json",
    )
    ckpt_local = tmp_path / "checkpoint.pt"
    ckpt_local.write_bytes(b"torch")
    monkeypatch.setattr(
        p, "download_paths",
        lambda ref, paths, run_dir: {"checkpoint.pt": ckpt_local},
    )

    stage = p._check_train(
        _stage({"pipeline": pipeline, "corpus": corpus,
                "checkpoint.json": ckpt})
    )
    assert stage["checkpoint"]["pt"] == str(ckpt_local)
    assert stage["checkpoint"]["repo_relative"].endswith("checkpoint.pt")
    assert stage["pipeline"]["training"]["status"] == "completed"
    assert stage["corpus"]["accepted"] == 5


@pytest.mark.parametrize(
    ("pipeline_training", "corpus", "ckpt_data", "ckpt_download",
     "match"),
        [
            ({"status": "skipped", "reason": "no imagery"},
             {"accepted": 0}, None, None, "did not complete"),
            ({"status": "completed"}, {"accepted": 0}, None, None,
             "0 accepted observations"),
            ({"status": "completed"}, {"accepted": 2}, {"found": False}, None,
             "no checkpoint"),
            ({"status": "completed"}, {"accepted": 2}, {"found": True,
                                                        "repo_relative": "r/c.pt"},
             None, "missing in train kernel output"),
        ],
)
def test_check_train_failure_modes(
    tmp_path, monkeypatch, pipeline_training, corpus, ckpt_data,
    ckpt_download, match,
):
    reports = {
        "pipeline": _report(tmp_path, {"training": pipeline_training},
                            "pipeline.json"),
    }
    if corpus is not None:
        reports["corpus"] = _report(tmp_path, corpus, "corpus.json")
    if ckpt_data is not None:
        reports["checkpoint.json"] = _report(tmp_path, ckpt_data,
                                             "checkpoint.json")
    if ckpt_download is not None:
        monkeypatch.setattr(
            p, "download_paths", lambda *a, **k: {"checkpoint.pt": None}
        )
    with pytest.raises(p.StageFailure, match=match):
        p._check_train(_stage(reports))


def test_check_export_pass(tmp_path):
    release = {name: _report(tmp_path, {"ok": True}, name)
               for name in p.RELEASE_PATHS}
    stage = p._check_export(_stage(release))
    assert stage["manifest"] == {"ok": True}


def test_check_export_fails_on_missing_artifacts(tmp_path):
    release = {"release.json": _report(tmp_path, {}, "release.json"),
               "model.onnx": None,
               "model.torchscript.pt": None,
               "model.yaml": None}
    with pytest.raises(p.StageFailure, match="model.onnx.*model.torchscript"):
        p._check_export(_stage(release))


# ---------------------------------------------------------------------------
# run_full_pipeline: checkpoint dataset handoff
# ---------------------------------------------------------------------------


class _FakeDatasetApi:
    def __init__(self, versions_ok=True, statuses=("ready",)):
        self.versions_ok = versions_ok
        self.statuses = list(statuses)
        self.version_calls = 0
        self.new_calls = 0
        self.status_calls = 0
        self.seen_metadata = None
        self.seen_notes = None

    def dataset_create_version(self, folder, version_notes, quiet=False):
        self.version_calls += 1
        self.seen_notes = version_notes
        if not self.versions_ok:
            raise RuntimeError("dataset not found")
        self._inspect(folder)

    def dataset_create_new(self, folder, public=False, quiet=False):
        self.new_calls += 1
        self._inspect(folder)

    def dataset_status(self, ref):
        self.status_calls += 1
        return self.statuses.pop(0)

    def _inspect(self, folder):
        folder = Path(folder)
        assert (folder / "checkpoint.pt").exists()
        meta = json.loads(
            (folder / "dataset-metadata.json").read_text(encoding="utf-8")
        )
        assert meta["id"] == "testowner/cropfusion-checkpoints"
        assert meta["licenses"][0]["name"] == "other"
        self.seen_metadata = meta


def test_publish_checkpoint_versions_existing_dataset(tmp_path):
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"torch")
    api = _FakeDatasetApi(statuses=("pending", "ready"))

    ref = p.publish_checkpoint(api, "testowner", ckpt, "notes",
                               p.CHECKPOINT_DATASET_SLUG)
    assert ref == "testowner/cropfusion-checkpoints"
    assert api.version_calls == 1
    assert api.new_calls == 0
    assert api.seen_notes == "notes"


def test_publish_checkpoint_creates_when_missing(tmp_path):
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"torch")
    api = _FakeDatasetApi(versions_ok=False, statuses=("ready",))

    p.publish_checkpoint(api, "testowner", ckpt, "notes",
                         p.CHECKPOINT_DATASET_SLUG)
    assert api.version_calls == 1
    assert api.new_calls == 1


def test_publish_checkpoint_create_also_fails(tmp_path):
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"torch")

    class _Broken(_FakeDatasetApi):
        def dataset_create_new(self, folder, public=False, quiet=False):
            self.new_calls += 1
            raise RuntimeError("forbidden")

    with pytest.raises(p.StageFailure, match="forbidden"):
        p.publish_checkpoint(_Broken(versions_ok=False), "testowner", ckpt,
                             "notes", p.CHECKPOINT_DATASET_SLUG)


def test_publish_checkpoint_waits_until_ready(monkeypatch, tmp_path):
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"torch")
    monkeypatch.setattr("time.sleep", lambda *_: None)
    api = _FakeDatasetApi(statuses=("not_yet_persisted", "blobs_received",
                                    "ready"))
    p.publish_checkpoint(api, "testowner", ckpt, "notes",
                         p.CHECKPOINT_DATASET_SLUG)
    assert api.status_calls == 3


def test_publish_checkpoint_fails_on_dataset_error(tmp_path):
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"torch")
    api = _FakeDatasetApi(statuses=("failed",))
    with pytest.raises(p.StageFailure, match="ended in status failed"):
        p.publish_checkpoint(api, "testowner", ckpt, "notes",
                             p.CHECKPOINT_DATASET_SLUG)


def test_publish_checkpoint_times_out(tmp_path):
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"torch")

    class _Pending(_FakeDatasetApi):
        def dataset_status(self, ref):
            self.status_calls += 1
            return "not_yet_persisted"

    api = _Pending()
    with pytest.raises(p.StageFailure, match="not ready within 0.6s"):
        p.publish_checkpoint(api, "testowner", ckpt, "notes",
                             p.CHECKPOINT_DATASET_SLUG, poll_timeout=0.6,
                             poll_interval=0.01)
    assert api.status_calls >= 1


# ---------------------------------------------------------------------------
# run_full_pipeline: _wait_dataset_file (input-mirror readiness)
# ---------------------------------------------------------------------------


class _FileList:
    def __init__(self, names):
        self.dataset_files = [_EnumName(n) for n in names]


def test_wait_dataset_file_succeeds_when_listed():
    class _Api:
        def dataset_list_files(self, ref):
            return _FileList(["checkpoint.pt"])

    p._wait_dataset_file(_Api(), "o/d", "checkpoint.pt", timeout=10,
                        interval=0.01)


def test_wait_dataset_file_retries_until_listed(monkeypatch):
    calls = []

    class _Api:
        def dataset_list_files(self, ref):
            calls.append(ref)
            return _FileList([] if len(calls) < 2 else ["checkpoint.pt"])

    monkeypatch.setattr("time.sleep", lambda *_: None)
    p._wait_dataset_file(_Api(), "o/d", "checkpoint.pt", timeout=10,
                        interval=0.01)
    assert len(calls) == 2


def test_wait_dataset_file_times_out(monkeypatch):
    class _Api:
        def dataset_list_files(self, ref):
            return _FileList(["other.pt"])

    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(p.StageFailure, match="never exposed 'checkpoint.pt'"):
        p._wait_dataset_file(_Api(), "o/d", "checkpoint.pt", timeout=0.1,
                            interval=0.01)


def test_wait_dataset_file_survives_transient_api_errors(monkeypatch):
    calls = []

    class _Api:
        def dataset_list_files(self, ref):
            calls.append(ref)
            if len(calls) == 1:
                raise RuntimeError("connection reset")
            return _FileList(["checkpoint.pt"])

    monkeypatch.setattr("time.sleep", lambda *_: None)
    p._wait_dataset_file(_Api(), "o/d", "checkpoint.pt", timeout=10,
                        interval=0.01)
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# run_full_pipeline: _run_stage + main()
# ---------------------------------------------------------------------------


def test_run_stage_writes_log_and_summary(tmp_path, monkeypatch):
    class _Api:
        def kernels_logs(self, ref):
            return "passed: True\n"

    def _download(ref, paths, run_dir):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for name, rel in paths.items():
            f = run_dir / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("{}", encoding="utf-8")
            out[name] = f
        return out

    monkeypatch.setattr(p, "build_push_dir",
                        lambda *a, **k: tmp_path / "push")
    monkeypatch.setattr(p, "push_notebook", lambda api, d: (3, "https://url"))
    monkeypatch.setattr(p, "wait_for_completion",
                        lambda api, ref, t, i: "COMPLETE")
    monkeypatch.setattr(p, "download_paths", _download)

    stage = p._run_stage(_Api(), "owner", "train", tmp_path,
                         {"train": 60.0}, 1.0, True)
    assert stage["run_id"] == "train-v3"
    assert stage["status"] == "COMPLETE"
    assert (stage["run_dir"] / "kernel.log").read_text(encoding="utf-8") \
        == "passed: True\n"
    summary = json.loads(
        (stage["run_dir"] / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["kernel"] == "owner/cropfusion-train"


def test_run_stage_surfaces_infrastructure_failure(monkeypatch, tmp_path):
    def _boom(api, ref, t, i):
        raise RuntimeError("Kaggle kernel run failed: crashed")

    monkeypatch.setattr(p, "build_push_dir",
                        lambda *a, **k: tmp_path / "push")
    monkeypatch.setattr(p, "push_notebook", lambda api, d: (1, "https://url"))
    monkeypatch.setattr(p, "wait_for_completion", _boom)
    with pytest.raises(p.StageFailure, match="infrastructure failure.*crashed"):
        p._run_stage(object(), "owner", "train", tmp_path,
                     {"train": 60.0}, 1.0, True)


def _fake_stage(name):
    base = {"run_id": f"{name}-v1", "status": "COMPLETE", "url": "url",
            "kernel_ref": f"testowner/cropfusion-{name}", "version": 1,
            "run_dir": str(Path("runs") / f"{name}-v1")}
    if name == "system_check":
        return {**base,
                "validation": {"passed": True,
                               "by_severity": {"info": 1}}}
    if name == "train":
        return {**base,
                "corpus": {"accepted": 5, "total": 8, "rejected": 3,
                           "errors": 0, "acceptance_rate": 0.625,
                           "plan": {"years": [2018, 2025]}},
                "pipeline": {"training": {"status": "completed",
                                          "accepted_observations": 5,
                                          "report": {"evaluation":
                                                     {"loss": 0.1}}}},
                "checkpoint": {"pt": "C:/x/checkpoint.pt",
                               "repo_relative": "training/artifacts/"
                                                "checkpoints/x/checkpoint.pt"}}
    return {**base,
            "reports": {name_: str(Path("release") / name_)
                        for name_ in p.RELEASE_PATHS}}


def _patch_pipeline(monkeypatch, tmp_path, raise_stage=None):
    def _stage(api, owner, name, runs_dir, timeouts, poll, keep, **kw):
        return _fake_stage(name)

    def _check(stage, raise_stage=raise_stage):
        if raise_stage is not None:
            raise raise_stage
        return stage

    def _whoami():
        return {"username": "testowner"}

    class _Api:
        def authenticate(self):
            pass

    monkeypatch.setattr(p.kagglehub, "whoami", _whoami)
    monkeypatch.setattr(p, "KaggleApi", lambda: _Api())
    monkeypatch.setattr(p, "_run_stage", _stage)
    monkeypatch.setattr(p, "_check_system_check", _check)
    monkeypatch.setattr(p, "_check_train", _check)
    monkeypatch.setattr(p, "_check_export", _check)
    monkeypatch.setattr(p, "publish_checkpoint",
                        lambda *a, **k: "testowner/cropfusion-checkpoints")
    monkeypatch.setattr(p, "_wait_dataset_file", lambda *a, **k: None)


def test_main_success(tmp_path, monkeypatch, capsys):
    _patch_pipeline(monkeypatch, tmp_path)
    rc = p.main([
        "--runs-dir", str(tmp_path),
        "--system-check-timeout", "1",
        "--train-timeout", "1",
        "--export-timeout", "1",
    ])
    assert rc == 0
    assert "CROPFUSION PIPELINE COMPLETE" in capsys.readouterr().out
    summaries = list(tmp_path.glob("pipeline-*.json"))
    assert len(summaries) == 1
    data = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert data["checkpoint_dataset"] == "testowner/cropfusion-checkpoints"
    assert data["training"]["status"] == "completed"
    assert data["corpus"]["year_range"] == [2018, 2025]


def test_main_aborts_on_stage_failure(tmp_path, monkeypatch, capsys):
    _patch_pipeline(monkeypatch, tmp_path,
                    raise_stage=p.StageFailure("system check FAILED: GPU_BAD"))
    rc = p.main(["--runs-dir", str(tmp_path)])
    assert rc == 1
    assert "GPU_BAD" in capsys.readouterr().err
