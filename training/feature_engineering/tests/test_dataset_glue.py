"""Tests for the corpus -> preprocessing dataset bridge."""

from __future__ import annotations

from training.feature_engineering.dataset import build_cropfusion_datasets


def test_injected_split_and_dataset(corpus):
    """Verify splitting + wrapping without touching the real preprocessor."""

    class FakeDataset:
        def __init__(self, preprocessor, observations, split, extractor=None):
            self.observations = observations
            self.split = split

        @classmethod
        def build(cls, preprocessor, observations, split="train", extractor=None):
            return cls(preprocessor, observations, split, extractor)

    def fake_split(observations, config=None):
        mid = max(1, len(observations) // 3)
        return (
            list(observations[:mid]),
            list(observations[mid: 2 * mid]),
            list(observations[2 * mid:]),
        )

    train, val, test = build_cropfusion_datasets(
        corpus,
        preprocessor="preprocessor",
        split_config=None,
        extractor="extractor",
        split_observations=fake_split,
        cropfusion_dataset=FakeDataset,
    )
    expected_total = len(corpus.accepted())
    assert len(train.observations) + len(val.observations) + len(test.observations) == expected_total
    assert train.split == "train"
    assert val.split == "val"
    assert test.split == "test"


def test_empty_corpus(tmp_path):
    from training.stam.observation_resolver import ObservationCorpus

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            self.observations = args[1]

        @classmethod
        def build(cls, preprocessor, observations, split="train", extractor=None):
            return cls(preprocessor, observations, split, extractor)

    empty = ObservationCorpus(samples=[])
    train, val, test = build_cropfusion_datasets(
        empty, preprocessor=None, cropfusion_dataset=FakeDataset
    )
    assert len(train.observations) == 0
    assert len(val.observations) == 0
    assert len(test.observations) == 0
