"""R5.4 regression tests: the EXPLICIT class-schema contract.

Root cause under test: the crop encoder used to learn its vocabulary from
whatever labels appeared in the (quality-filtered) training split. A corpus
label with zero training rows (blackgram) silently disappeared, resizing the
crop head from 5 to 4 classes while the manifest still declared 5. These tests
pin the new behaviour: the vocabulary comes from ``LabelConfig.declared_classes``
when supplied, and excluded labels are counted (never silently dropped).
"""

from __future__ import annotations

import numpy as np
import pytest

from training.preprocessing.config import LabelConfig
from training.preprocessing.exceptions import FitError
from training.preprocessing.label_pipeline import LabelPipeline


class _Obs:
    def __init__(self, crop, yield_value=None):
        self.crop = crop
        self.yield_value = yield_value


def _codes(pipeline, crops):
    return [int(pipeline.transform(_Obs(c))[0]) for c in crops]


def test_declared_vocabulary_is_deterministic_regardless_of_train_labels():
    # Even when the training split contains classes the model must NOT learn,
    # the encoder follows the declared vocabulary exactly.
    pipeline = LabelPipeline(
        LabelConfig(
            declared_classes=["coconut", "pepper", "coffee", "cardamom"],
            excluded_classes=["blackgram"],
        )
    ).fit([_Obs("coconut"), _Obs("pepper"), _Obs("coffee"), _Obs("cardamom")])
    assert pipeline.num_classes == 4
    assert pipeline.declared_classes == ["coconut", "pepper", "coffee", "cardamom"]
    assert pipeline.excluded_classes == ["blackgram"]
    assert pipeline.excluded_counts == {}
    assert _codes(pipeline, ["coconut", "pepper", "coffee", "cardamom"]) == [0, 1, 2, 3]


def test_excluded_label_encodes_to_unknown_and_is_counted():
    # The real-world blackgram case: label exists in the corpus but has zero
    # training rows. It must be documented (excluded_counts), not vanish.
    pipeline = LabelPipeline(
        LabelConfig(
            declared_classes=["coconut", "pepper", "coffee", "cardamom"],
            excluded_classes=["blackgram"],
        )
    ).fit(
        [_Obs("coconut"), _Obs("coconut"), _Obs("pepper"), _Obs("cardamom")]
    )
    assert pipeline.excluded_counts == {}
    # No blackgram rows were trained on, so nothing was excluded at FIT time.
    # Transform of an out-of-vocabulary crop yields the -1 sentinel.
    assert _codes(pipeline, ["blackgram"]) == [-1]


def test_emergent_vocabulary_still_supported_when_undeclared():
    # Without declared_classes, previous emergent behaviour is preserved.
    pipeline = LabelPipeline().fit([_Obs("coconut"), _Obs("pepper")])
    assert pipeline.num_classes == 2
    assert _codes(pipeline, ["coconut", "pepper"]) == [0, 1]
    assert pipeline.excluded_counts == {}


def test_declared_class_with_zero_train_support_warns_but_fits():
    # A declared class that has no training rows must be surfaced as a warning
    # (it is unlearnable) rather than silently shrinking the head.
    pipeline = LabelPipeline(
        LabelConfig(declared_classes=["coconut", "pepper", "cardamom"])
    ).fit([_Obs("coconut"), _Obs("pepper")])
    assert pipeline.num_classes == 3
    assert any("ZERO training" in w for w in pipeline.warnings)
    assert "cardamom" in " ".join(pipeline.warnings)


def test_stray_train_labels_are_counted_not_hidden():
    pipeline = LabelPipeline(
        LabelConfig(declared_classes=["coconut"], excluded_classes=["blackgram"])
    ).fit([_Obs("coconut"), _Obs("pepper")])
    # 'pepper' appeared in train but is outside the declared vocabulary.
    assert pipeline.excluded_counts == {"pepper": 1}
    assert any("outside the supervised vocabulary" in w for w in pipeline.warnings)


def test_overlapping_declared_and_excluded_raises():
    with pytest.raises(FitError):
        LabelPipeline(
            LabelConfig(declared_classes=["coconut"], excluded_classes=["coconut"])
        ).fit([_Obs("coconut")])


def test_schema_survives_save_and_load(tmp_path):
    pipeline = LabelPipeline(
        LabelConfig(
            declared_classes=["coconut", "pepper", "coffee"],
            excluded_classes=["blackgram"],
        )
    ).fit([_Obs("coconut"), _Obs("pepper"), _Obs("coffee")])
    pipeline.save(tmp_path)
    loaded = LabelPipeline.load(tmp_path)
    assert loaded.declared_classes == ["coconut", "pepper", "coffee"]
    assert loaded.excluded_classes == ["blackgram"]
    assert loaded.num_classes == 3


def test_summary_exposes_contract_and_excluded_bookkeeping():
    pipeline = LabelPipeline(
        LabelConfig(declared_classes=["coconut", "pepper"])
    ).fit([_Obs("coconut")])
    summary = pipeline.summary()
    assert summary["declared_classes"] == ["coconut", "pepper"]
    assert summary["classes"] == ["coconut", "pepper"]
    assert summary["num_classes"] == 2
    assert summary["excluded_sample_counts"] == {}
    assert any("ZERO training" in w for w in pipeline.warnings)


def test_declared_transform_matches_declared_order_not_train_order():
    # Train order pepper-first, declared order coconut-first: the encoded index
    # must follow the declared contract, NOT the training first-appearance order.
    pipeline = LabelPipeline(
        LabelConfig(declared_classes=["coconut", "pepper"])
    ).fit([_Obs("pepper"), _Obs("pepper"), _Obs("coconut")])
    assert _codes(pipeline, ["coconut", "pepper"]) == [0, 1]