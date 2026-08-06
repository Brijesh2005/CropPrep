"""Ablation study tests (Phase R5)."""

from __future__ import annotations

import pytest
import torch

from training.evaluation.ablation import (
    ABLATION_VARIANTS,
    AblationStudy,
    DEFAULT_VARIANTS,
    build_variant_config,
)
from training.evaluation.config import AblationConfig, EvaluationConfig
from training.evaluation.exceptions import AblationStudyError

from .conftest import small_full_config


class TestVariantRegistry:
    def test_seven_variants(self):
        assert len(ABLATION_VARIANTS) == 7
        assert DEFAULT_VARIANTS == (
            "without_tabtransformer",
            "without_efficientnet",
            "without_temporal_encoder",
            "without_cross_attention",
            "without_adaptive_gate",
            "without_confidence_fusion",
            "without_temporal_branch",
        )

    def test_build_variant_config(self):
        base = small_full_config()
        variant = build_variant_config(base, "without_efficientnet")
        assert variant.image_encoder.backbone is None
        assert variant.fusion.use_temporal_stream is False
        assert variant.tabular.numeric_dim == 3

        tabular_only = build_variant_config(base, "without_tabtransformer")
        assert tabular_only.tabular.numeric_dim == 0
        assert tabular_only.tabular.categorical_cardinalities == []

    def test_unknown_variant_raises(self):
        with pytest.raises(AblationStudyError):
            build_variant_config(small_full_config(), "not_a_variant")

    def test_each_variant_builds_a_valid_model(self):
        base = small_full_config()
        from training.models import ModelFactory

        for spec in ABLATION_VARIANTS:
            config = build_variant_config(base, spec["name"])
            model = ModelFactory.create(config)
            assert model is not None, spec["name"]


class TestAblationStudy:
    def test_run_subset(self, full_model, fake_loader):
        config = EvaluationConfig(
            general={"device": "cpu"},
            ablation=AblationConfig(
                benchmark_iterations=2, benchmark_warmup=1
            ),
        )
        study = AblationStudy(full_model, config)
        report = study.run(
            fake_loader,
            variants=["without_temporal_encoder", "without_cross_attention"],
        )
        assert report.best_variant is not None
        assert set(report.results) == {
            "without_temporal_encoder", "without_cross_attention"
        }
        assert report.results["without_temporal_encoder"]["parameter_count"] > 0
        assert "crop/accuracy" in report.comparison["columns"]
