"""Tests for the shared logging setup and formatters."""

from __future__ import annotations

import io
import json
import logging

import pytest

from shared.exceptions import LoggingConfigurationError
from shared.logging import (
    JsonFormatter,
    get_logger,
    is_configured,
    setup_logging,
)


def _call_with_stream(formatter: logging.Formatter) -> dict:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello world", args=(), exc_info=None,
    )
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    handler.emit(record)
    return json.loads(stream.getvalue())


def test_json_formatter_emits_structured_record() -> None:
    payload = _call_with_stream(JsonFormatter())
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert "timestamp" in payload


def test_json_formatter_folds_extra_fields() -> None:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="boom", args=(), exc_info=None,
    )
    record.event = "training.started"
    record.dataset_id = "ds-1"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.emit(record)
    payload = json.loads(stream.getvalue())
    assert payload["event"] == "training.started"
    assert payload["dataset_id"] == "ds-1"


def test_setup_logging_idempotent() -> None:
    first = setup_logging(profile="default")
    second = setup_logging(profile="default")
    assert first is second
    assert first.handlers
    assert is_configured()
    assert get_logger("shared.tests").name == "cropfusion.shared.tests"


def test_setup_logging_attaches_file_handler(tmp_path) -> None:
    logger = setup_logging(profile="training", log_dir=tmp_path, console=False)
    files = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(files) == 1
    assert (tmp_path / "training.log").exists()


def test_setup_logging_unknown_profile_raises() -> None:
    with pytest.raises(LoggingConfigurationError):
        setup_logging(profile="does-not-exist")
