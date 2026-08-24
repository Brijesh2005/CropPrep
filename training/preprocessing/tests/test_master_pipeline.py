"""Integration tests for the master preprocessing pipeline."""

from __future__ import annotations

import numpy as np

from training.preprocessing import Preprocessor
from training.preprocessing.validators import filter_observations


def test_extract_patch_out_of_bounds_returns_none(
    observations, preprocessing_config
):
    """A patch outside the raster becomes a missing band, not a crash.

    Regression: the PatchOutOfBoundsError handler logged
    ``observation.id`` (nonexistent attribute), so the handler itself
    raised AttributeError and killed the DataLoader worker (R5.3 v6).
    """
    from training.stam.exceptions import PatchOutOfBoundsError

    pre = Preprocessor(preprocessing_config)
    pre.fit(observations[:1], extractor=lambda *a, **k: None)
    obs = observations[0]

    def boom(path, lon, lat, size=None):
        raise PatchOutOfBoundsError(
            f"Patch is entirely outside the raster: {path}",
            detail={"lon": lon, "lat": lat},
        )

    assert pre._extract_patch(boom, "r.tif", 74.8, 13.1, 32, obs) is None


def test_fit_transform_shapes(observations, extractor, preprocessing_config):
    accepted, _ = filter_observations(observations, preprocessing_config.quality)
    assert len(accepted) == len(observations)

    preprocessor = Preprocessor(preprocessing_config)
    preprocessor.fit(accepted, extractor=extractor)
    sample = preprocessor.transform(accepted[0], extractor=extractor)

    assert set(sample.keys()) == {
        "observation_id", "tabular", "ndvi", "evi", "temporal_mask",
        "crop_label", "yield_label", "metadata",
    }
    # Feature count matches the tabular summary.
    assert sample["tabular"].shape[0] == preprocessor.tabular.feature_names.__len__()
    # Sequences: [T, 1, H, W].
    assert sample["ndvi"].shape[0] == preprocessing_config.temporal.max_observations
    assert sample["ndvi"].shape[2] == preprocessing_config.image.size
    assert sample["evi"].shape == sample["ndvi"].shape
    assert sample["temporal_mask"].shape[0] == preprocessing_config.temporal.max_observations
    assert float(sample["temporal_mask"].sum()) == accepted[0].num_observations()
    # Labels.
    assert sample["crop_label"].dim() == 0
    assert sample["yield_label"].dim() == 0
    # Metadata.
    assert sample["metadata"]["year"] == 2020


def test_fit_transform_all(observations, extractor, preprocessing_config):
    accepted, _ = filter_observations(observations, preprocessing_config.quality)
    preprocessor = Preprocessor(preprocessing_config)
    samples = preprocessor.fit_transform(accepted, extractor=extractor)
    assert len(samples) == len(accepted)
    assert all(s["ndvi"].shape == samples[0]["ndvi"].shape for s in samples)


def test_fit_is_train_only_no_leakage(observations, extractor, preprocessing_config):
    # Fit on a subset; transform must work for a held-out observation with
    # consistent feature ordering.
    preprocessor = Preprocessor(preprocessing_config)
    preprocessor.fit(observations[:2], extractor=extractor)
    sample = preprocessor.transform(observations[2], extractor=extractor)
    assert sample["tabular"].shape[0] == len(preprocessor.tabular.feature_names)


def test_quality_rejection(observations, preprocessing_config):
    from training.preprocessing.config import QualityConfig

    config = preprocessing_config.model_copy(
        update={"quality": QualityConfig(min_quality_score=100.0)}
    )
    accepted, _ = filter_observations(observations, config.quality)
    assert len(accepted) < len(observations)  # the unpaired 2021 obs is dropped


def test_summary(observations, extractor, preprocessing_config):
    preprocessor = Preprocessor(preprocessing_config).fit(observations, extractor=extractor)
    summary = preprocessor.summary()
    assert summary["fitted"] is True
    assert summary["label"]["num_classes"] >= 1
    assert summary["temporal"]["max_observations"] == 8
    assert summary["image"]["size"] == 32


def test_validate_returns_list(observations, extractor, preprocessing_config):
    preprocessor = Preprocessor(preprocessing_config).fit(observations, extractor=extractor)
    issues = preprocessor.validate(observations[0])
    assert isinstance(issues, list)


def test_save_load_roundtrip(observations, extractor, preprocessing_config, tmp_path):
    preprocessor = Preprocessor(preprocessing_config).fit(observations, extractor=extractor)
    out = preprocessor.save(tmp_path / "pp")
    loaded = Preprocessor.load(out)
    a = preprocessor.transform(observations[0], extractor=extractor)
    b = loaded.transform(observations[0], extractor=extractor)
    assert np.allclose(a["tabular"].numpy(), b["tabular"].numpy())
    assert np.allclose(a["ndvi"].numpy(), b["ndvi"].numpy())


def test_from_config(tmp_path):
    from training.preprocessing import save_preprocessing_template

    template = tmp_path / "pre.yaml"
    save_preprocessing_template(template)
    preprocessor = Preprocessor.from_config(template)
    assert preprocessor.config.image.size == 128
