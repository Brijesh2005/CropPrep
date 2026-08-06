"""Tests for the sample-quality reporting package."""

from __future__ import annotations

import json

import pytest

from training.quality.samples import SampleQualityError, SampleQualityReport, build_report


class TestFromCorpus:
    def test_counts_match_corpus(self, corpus):
        report = SampleQualityReport.from_corpus(corpus)
        assert report.total == corpus.total
        assert report.status_counts == corpus.status_counts()
        assert 0.0 < report.acceptance_rate <= 1.0

    def test_quality_score_summary(self, corpus):
        report = SampleQualityReport.from_corpus(corpus)
        summary = report.quality_score
        accepted = [s for s in corpus.samples if s.status == "accepted"]
        if accepted:
            assert summary["count"] == len(accepted)
            assert 0.0 <= summary["min"] <= summary["max"] <= 100.0
            assert summary["mean"] is not None
        else:
            assert summary["count"] == 0

    def test_issue_codes_present_for_accepted(self, corpus):
        report = SampleQualityReport.from_corpus(corpus)
        accepted = [s for s in corpus.samples if s.status == "accepted"]
        assert all(
            isinstance(code, str) and code
            for code in report.issue_codes
        )
        assert set(report.severity_counts) <= {"info", "warning", "error", "critical"}
        expected_issue_total = sum(
            len(s.observation.quality.issues)
            for s in accepted
            if s.observation is not None
        )
        assert sum(report.issue_codes.values()) == expected_issue_total

    def test_grouped_rates(self, corpus):
        report = SampleQualityReport.from_corpus(corpus)
        for by in (report.by_crop, report.by_year, report.by_season):
            for key, entry in by.items():
                assert entry["total"] >= entry["accepted"]
                assert 0.0 <= entry["rate"] <= 1.0
        if corpus.samples:
            assert report.by_year
            assert report.by_season

    def test_error_cells_leave_top_error_codes(self, corpus):
        report = SampleQualityReport.from_corpus(corpus)
        errors = [s for s in corpus.samples if s.status == "error"]
        if errors:
            assert report.top_error_codes
        else:
            assert report.top_error_codes == {}

    def test_raises_without_samples(self):
        with pytest.raises(SampleQualityError):
            SampleQualityReport.from_corpus(None)
        with pytest.raises(SampleQualityError):
            SampleQualityReport.from_corpus([])


class TestBuildReport:
    def test_build_writes_json(self, corpus, tmp_path):
        report = build_report(corpus, output_dir=tmp_path)
        path = tmp_path / "sample_quality_report.json"
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["total"] == corpus.total
        assert payload["status"] == corpus.status_counts()
        assert report.to_dict()["total"] == payload["total"]

    def test_build_accepts_plain_sample_list(self, corpus):
        report = build_report(list(corpus.samples))
        assert report.total == corpus.total

    def test_to_dict_roundtrip(self, corpus):
        report = SampleQualityReport.from_corpus(corpus)
        data = report.to_dict()
        assert set(data) >= {
            "total",
            "status",
            "acceptance_rate",
            "quality_score",
            "issue_codes",
            "severity_counts",
            "by_crop",
            "by_year",
            "by_season",
            "top_error_codes",
        }
