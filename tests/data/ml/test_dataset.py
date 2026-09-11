from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from weather_radar_ml.config.ml_dataset import (
    FeatureConfig,
    FeatureKind,
    FeatureSchemaConfig,
    MLDatasetConfig,
    PreparedSourceConfig,
    SourceRole,
    SpatialSamplingConfig,
    SplitIntervalConfig,
    SplitName,
    TemporalSplitConfig,
    TemporalWindowConfig,
)
from weather_radar_ml.data.ml.catalog import SQLiteSampleCatalog
from weather_radar_ml.data.ml.dataset import LazyRadarDataset
from weather_radar_ml.data.ml.domain import SampleSelection, SpatialWindowMode


def _dt(minutes: int) -> datetime:
    return datetime(2023, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes)


def _config(zarr_path) -> MLDatasetConfig:
    return MLDatasetConfig(
        name="dataset",
        sources=(
            PreparedSourceConfig(
                "radar",
                "artifact",
                SourceRole.PRIMARY,
                zarr_path,
            ),
        ),
        features=FeatureSchemaConfig(
            dynamic_inputs=(
                FeatureConfig("rain", "radar", "rain", FeatureKind.DYNAMIC),
            ),
            static_inputs=(
                FeatureConfig("height", "radar", "height", FeatureKind.STATIC),
            ),
            targets=(
                FeatureConfig(
                    "rain_target",
                    "radar",
                    "rain",
                    FeatureKind.DYNAMIC,
                ),
            ),
        ),
        temporal=TemporalWindowConfig(
            step_minutes=5,
            history_minutes=10,
            forecast_minutes=10,
        ),
        spatial=SpatialSamplingConfig(),
        splits=TemporalSplitConfig(
            (
                SplitIntervalConfig(SplitName.TRAIN, _dt(0), _dt(60)),
                SplitIntervalConfig(
                    SplitName.VALIDATION,
                    _dt(60),
                    _dt(120),
                ),
                SplitIntervalConfig(SplitName.TEST, _dt(120), _dt(180)),
            )
        ),
    )


def _prepared(tmp_path):
    times = np.array(
        [np.datetime64(_dt(index * 5).replace(tzinfo=None)) for index in range(6)]
    )
    dataset = xr.Dataset(
        {
            "rain": (
                ("time", "y", "x"),
                np.arange(6 * 4 * 5).reshape(6, 4, 5),
            ),
            "height": (("y", "x"), np.arange(4 * 5).reshape(4, 5)),
        },
        coords={"time": times, "y": np.arange(4), "x": np.arange(5)},
    )
    path = tmp_path / "prepared.zarr"
    dataset.to_zarr(path, mode="w")
    return path


def _catalog(tmp_path) -> SQLiteSampleCatalog:
    catalog = SQLiteSampleCatalog(tmp_path / "samples.sqlite")
    catalog.initialize()
    catalog.add_classification_profile(
        profile_id="precip_event_v1",
        definition_json="{}",
    )
    catalog.add_dataset_build(
        dataset_build_id="build",
        dataset_name="dataset",
        sample_schema_version=1,
        feature_schema_id="features",
        classification_profile_id="precip_event_v1",
    )
    catalog.add_source(
        dataset_build_id="build",
        source_id="radar",
        role="primary",
        artifact_id="artifact",
        checksum="checksum",
    )
    split_id = catalog.add_split(dataset_build_id="build", name="train")
    window_id = catalog.add_spatial_window(
        dataset_build_id="build",
        x=1,
        y=1,
        width=3,
        height=2,
        mode=SpatialWindowMode.PATCH,
    )
    catalog.add_sample(
        sample_id="sample",
        dataset_build_id="build",
        feature_schema_id="features",
        input_start=_dt(0),
        input_end=_dt(5),
        target_start=_dt(10),
        target_end=_dt(15),
        window_id=window_id,
        split_id=split_id,
        input_event_class="light",
        target_event_class="moderate",
    )
    return catalog


def test_lazy_dataset_loads_expected_sample(tmp_path) -> None:
    path = _prepared(tmp_path)
    dataset = LazyRadarDataset(catalog=_catalog(tmp_path), config=_config(path))

    sample = dataset[0]

    assert sample.inputs.dynamic.shape == (2, 1, 2, 3)
    assert sample.inputs.static is not None
    assert sample.inputs.static.shape == (1, 2, 3)
    assert sample.targets.dynamic.shape == (2, 1, 2, 3)
    assert sample.context.sample_id == "sample"
    assert sample.context.metadata["split"] == "train"


def test_dataset_get_returns_same_sample(tmp_path) -> None:
    path = _prepared(tmp_path)
    dataset = LazyRadarDataset(catalog=_catalog(tmp_path), config=_config(path))

    assert dataset.get("sample").context == dataset[0].context


def test_selection_filters_view(tmp_path) -> None:
    path = _prepared(tmp_path)
    selected = LazyRadarDataset(
        catalog=_catalog(tmp_path),
        config=_config(path),
        selection=SampleSelection(split="validation"),
    )

    assert len(selected) == 0


def test_get_respects_selection(tmp_path) -> None:
    path = _prepared(tmp_path)
    dataset = LazyRadarDataset(
        catalog=_catalog(tmp_path),
        config=_config(path),
        selection=SampleSelection(split="validation"),
    )

    with pytest.raises(KeyError):
        dataset.get("sample")


def test_explicit_source_path_override_is_supported(tmp_path) -> None:
    path = _prepared(tmp_path)
    dataset = LazyRadarDataset(
        catalog=_catalog(tmp_path),
        config=_config(None),
        source_paths={"radar": path},
    )

    assert dataset[0].inputs.dynamic.shape == (2, 1, 2, 3)


def test_unknown_source_path_override_is_rejected(tmp_path) -> None:
    path = _prepared(tmp_path)

    with pytest.raises(ValueError, match="Unknown source path overrides"):
        LazyRadarDataset(
            catalog=_catalog(tmp_path),
            config=_config(path),
            source_paths={"missing": path},
        )


def test_missing_source_path_fails_on_lazy_load(tmp_path) -> None:
    dataset = LazyRadarDataset(
        catalog=_catalog(tmp_path),
        config=_config(None),
    )

    with pytest.raises(ValueError, match="No storage path configured"):
        _ = dataset[0]


def test_catalog_resolves_spatial_window(tmp_path) -> None:
    catalog = _catalog(tmp_path)
    record = catalog.get("sample")

    window = catalog.get_spatial_window(record.window_id)

    assert (window.x, window.y, window.width, window.height) == (1, 1, 3, 2)
    assert window.mode is SpatialWindowMode.PATCH


def test_select_creates_independent_view(tmp_path) -> None:
    path = _prepared(tmp_path)
    dataset = LazyRadarDataset(catalog=_catalog(tmp_path), config=_config(path))

    selected = dataset.select(SampleSelection(split="train"))

    assert len(dataset) == 1
    assert len(selected) == 1
