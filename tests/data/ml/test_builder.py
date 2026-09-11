from __future__ import annotations

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
    SpatialMode,
    SpatialSamplingConfig,
    SplitIntervalConfig,
    SplitName,
    TemporalSplitConfig,
    TemporalWindowConfig,
)
from weather_radar_ml.data.ml.builder import (
    PrecipitationStatistics,
    SampleIndexBuilder,
    classify_precipitation,
    dataset_build_id,
    feature_schema_id,
    sample_id,
)
from weather_radar_ml.data.ml.catalog import SQLiteSampleCatalog
from weather_radar_ml.data.ml.domain import SpatialWindow, SpatialWindowMode


def _time(hour: int, minute: int = 0) -> datetime:
    return datetime(2023, 1, 1, hour, minute, tzinfo=UTC)


def _dataset(
    times: list[datetime],
    *,
    height: int = 4,
    width: int = 4,
    value: float = 1.0,
) -> xr.Dataset:
    data = np.full((len(times), height, width), value, dtype=np.float32)
    return xr.Dataset(
        {"precipitation_rate": (("time", "y", "x"), data)},
        coords={
            "time": np.asarray(
                [np.datetime64(value.replace(tzinfo=None)) for value in times]
            ),
            "y": np.arange(height),
            "x": np.arange(width),
        },
    )


def _config(*, spatial: SpatialSamplingConfig | None = None) -> MLDatasetConfig:
    source = PreparedSourceConfig(
        source_id="radklim",
        artifact_id="radklim-test-v1",
        role=SourceRole.PRIMARY,
    )
    input_feature = FeatureConfig(
        name="precipitation",
        source_id="radklim",
        variable="precipitation_rate",
        kind=FeatureKind.DYNAMIC,
    )
    target_feature = FeatureConfig(
        name="precipitation_target",
        source_id="radklim",
        variable="precipitation_rate",
        kind=FeatureKind.DYNAMIC,
    )
    return MLDatasetConfig(
        name="radklim-test",
        sources=(source,),
        features=FeatureSchemaConfig(
            dynamic_inputs=(input_feature,),
            targets=(target_feature,),
        ),
        temporal=TemporalWindowConfig(
            step_minutes=5,
            history_minutes=10,
            forecast_minutes=10,
        ),
        spatial=spatial or SpatialSamplingConfig(mode=SpatialMode.FULL_FRAME),
        splits=TemporalSplitConfig(
            (
                SplitIntervalConfig(SplitName.TRAIN, _time(0), _time(1)),
                SplitIntervalConfig(SplitName.VALIDATION, _time(2), _time(3)),
                SplitIntervalConfig(SplitName.TEST, _time(4), _time(5)),
            )
        ),
    )


def test_builder_persists_valid_full_frame_samples(tmp_path) -> None:
    times = [_time(0) + timedelta(minutes=5 * index) for index in range(6)]
    catalog = SQLiteSampleCatalog(tmp_path / "samples.sqlite")
    report = SampleIndexBuilder(
        config=_config(),
        source_datasets={"radklim": _dataset(times)},
        source_checksums={"radklim": "abc123"},
        catalog=catalog,
    ).build()

    assert report.valid_temporal_windows == 3
    assert report.spatial_windows == 1
    assert report.samples == 3
    assert report.samples_by_split == {"train": 3}
    assert len(catalog) == 3

    rows = catalog.query()
    assert [row.input_start for row in rows] == [_time(0), _time(0, 5), _time(0, 10)]
    assert all(row.input_event_class == "light" for row in rows)
    assert all(row.target_event_class == "light" for row in rows)


def test_missing_frame_rejects_affected_temporal_windows(tmp_path) -> None:
    times = [
        _time(0, 0),
        _time(0, 5),
        _time(0, 15),
        _time(0, 20),
        _time(0, 25),
        _time(0, 30),
    ]
    catalog = SQLiteSampleCatalog(tmp_path / "samples.sqlite")
    report = SampleIndexBuilder(
        config=_config(),
        source_datasets={"radklim": _dataset(times)},
        source_checksums={"radklim": "abc123"},
        catalog=catalog,
    ).build()

    assert report.rejected_missing_frames == 2
    assert report.valid_temporal_windows == 1
    assert len(catalog) == 1


def test_patch_sampling_uses_only_complete_windows(tmp_path) -> None:
    times = [_time(0) + timedelta(minutes=5 * index) for index in range(4)]
    spatial = SpatialSamplingConfig(
        mode=SpatialMode.PATCH,
        patch_height=3,
        patch_width=3,
        stride_y=2,
        stride_x=2,
    )
    catalog = SQLiteSampleCatalog(tmp_path / "samples.sqlite")
    report = SampleIndexBuilder(
        config=_config(spatial=spatial),
        source_datasets={"radklim": _dataset(times, height=6, width=6)},
        source_checksums={"radklim": "abc123"},
        catalog=catalog,
    ).build()

    assert report.valid_temporal_windows == 1
    assert report.spatial_windows == 4
    assert report.samples == 4


def test_sample_must_fit_completely_inside_one_split(tmp_path) -> None:
    times = [_time(0) + timedelta(minutes=5 * index) for index in range(4)]
    config = _config()
    config = MLDatasetConfig(
        name=config.name,
        sources=config.sources,
        features=config.features,
        temporal=config.temporal,
        spatial=config.spatial,
        splits=TemporalSplitConfig(
            (
                SplitIntervalConfig(SplitName.TRAIN, _time(0), _time(0, 15)),
                SplitIntervalConfig(SplitName.VALIDATION, _time(1), _time(2)),
                SplitIntervalConfig(SplitName.TEST, _time(3), _time(4)),
            )
        ),
        events=config.events,
        sample_schema_version=config.sample_schema_version,
    )
    catalog = SQLiteSampleCatalog(tmp_path / "samples.sqlite")
    report = SampleIndexBuilder(
        config=config,
        source_datasets={"radklim": _dataset(times)},
        source_checksums={"radklim": "abc123"},
        catalog=catalog,
    ).build()

    assert report.valid_temporal_windows == 0
    assert report.rejected_outside_splits == 1
    assert len(catalog) == 0


def test_event_classification_uses_wet_fraction_and_p95() -> None:
    profile = _config().events
    dry = classify_precipitation(
        PrecipitationStatistics(0.0, 0.0, 0.0, 0.0),
        profile,
    )
    assert dry is not None
    assert dry.value == "dry"
    assert classify_precipitation(None, profile) is None
    extreme = classify_precipitation(
        PrecipitationStatistics(30.0, 40.0, 30.0, 1.0),
        profile,
    )
    assert extreme is not None
    assert extreme.value == "extreme"


def test_all_nan_precipitation_keeps_sample_with_null_event_metadata(tmp_path) -> None:
    times = [_time(0) + timedelta(minutes=5 * index) for index in range(4)]
    dataset = _dataset(times)
    dataset["precipitation_rate"][:] = np.nan
    catalog = SQLiteSampleCatalog(tmp_path / "samples.sqlite")
    report = SampleIndexBuilder(
        config=_config(),
        source_datasets={"radklim": dataset},
        source_checksums={"radklim": "abc123"},
        catalog=catalog,
    ).build()

    assert report.samples == 1
    row = catalog.query()[0]
    assert row.input_precip_mean is None
    assert row.target_precip_mean is None
    assert row.input_event_class is None
    assert row.target_event_class is None


def test_feature_schema_and_build_ids_are_deterministic() -> None:
    config = _config()
    checksums = {"radklim": "abc123"}

    assert feature_schema_id(config) == feature_schema_id(config)
    assert dataset_build_id(config, checksums) == dataset_build_id(config, checksums)
    assert dataset_build_id(config, checksums) != dataset_build_id(
        config, {"radklim": "different"}
    )


def test_sample_id_changes_with_spatial_window_and_checksum() -> None:
    config = _config()
    inputs = (_time(0), _time(0, 5))
    targets = (_time(0, 10), _time(0, 15))
    first = SpatialWindow(0, 0, 2, 2, SpatialWindowMode.PATCH)
    second = SpatialWindow(2, 0, 2, 2, SpatialWindowMode.PATCH)

    base = sample_id(
        config,
        {"radklim": "abc123"},
        input_times=inputs,
        target_times=targets,
        window=first,
    )
    assert base == sample_id(
        config,
        {"radklim": "abc123"},
        input_times=inputs,
        target_times=targets,
        window=first,
    )
    assert base != sample_id(
        config,
        {"radklim": "abc123"},
        input_times=inputs,
        target_times=targets,
        window=second,
    )
    assert base != sample_id(
        config,
        {"radklim": "different"},
        input_times=inputs,
        target_times=targets,
        window=first,
    )


def test_builder_rejects_dynamic_source_with_different_time_grid(tmp_path) -> None:
    config = _config()
    auxiliary = PreparedSourceConfig(
        source_id="aux",
        artifact_id="aux-v1",
        role=SourceRole.AUXILIARY,
    )
    aux_feature = FeatureConfig(
        name="aux_dynamic",
        source_id="aux",
        variable="precipitation_rate",
        kind=FeatureKind.DYNAMIC,
    )
    config = MLDatasetConfig(
        name=config.name,
        sources=(*config.sources, auxiliary),
        features=FeatureSchemaConfig(
            dynamic_inputs=(*config.features.dynamic_inputs, aux_feature),
            targets=config.features.targets,
        ),
        temporal=config.temporal,
        spatial=config.spatial,
        splits=config.splits,
        events=config.events,
    )
    primary_times = [_time(0) + timedelta(minutes=5 * index) for index in range(4)]
    auxiliary_times = primary_times.copy()
    auxiliary_times[-1] += timedelta(minutes=5)

    with pytest.raises(ValueError, match="primary time grid"):
        SampleIndexBuilder(
            config=config,
            source_datasets={
                "radklim": _dataset(primary_times),
                "aux": _dataset(auxiliary_times),
            },
            source_checksums={"radklim": "abc123", "aux": "def456"},
            catalog=SQLiteSampleCatalog(tmp_path / "samples.sqlite"),
        )


def test_sample_id_ignores_split_assignment() -> None:
    config = _config()
    changed_split_config = MLDatasetConfig(
        name=config.name,
        sources=config.sources,
        features=config.features,
        temporal=config.temporal,
        spatial=config.spatial,
        splits=TemporalSplitConfig(
            (
                SplitIntervalConfig(SplitName.TRAIN, _time(0), _time(0, 30)),
                SplitIntervalConfig(SplitName.VALIDATION, _time(1), _time(2)),
                SplitIntervalConfig(SplitName.TEST, _time(3), _time(4)),
            )
        ),
        events=config.events,
        sample_schema_version=config.sample_schema_version,
    )
    inputs = (_time(0), _time(0, 5))
    targets = (_time(0, 10), _time(0, 15))
    window = SpatialWindow(0, 0, 2, 2, SpatialWindowMode.PATCH)
    checksums = {"radklim": "abc123"}

    assert sample_id(
        config,
        checksums,
        input_times=inputs,
        target_times=targets,
        window=window,
    ) == sample_id(
        changed_split_config,
        checksums,
        input_times=inputs,
        target_times=targets,
        window=window,
    )


def test_builder_rejects_same_shape_but_shifted_spatial_grid(tmp_path) -> None:
    config = _config()
    auxiliary = PreparedSourceConfig(
        source_id="aux",
        artifact_id="aux-v1",
        role=SourceRole.AUXILIARY,
    )
    static_feature = FeatureConfig(
        name="elevation",
        source_id="aux",
        variable="elevation",
        kind=FeatureKind.STATIC,
    )
    config = MLDatasetConfig(
        name=config.name,
        sources=(*config.sources, auxiliary),
        features=FeatureSchemaConfig(
            dynamic_inputs=config.features.dynamic_inputs,
            static_inputs=(static_feature,),
            targets=config.features.targets,
        ),
        temporal=config.temporal,
        spatial=config.spatial,
        splits=config.splits,
        events=config.events,
    )
    times = [_time(0) + timedelta(minutes=5 * index) for index in range(4)]
    auxiliary_dataset = xr.Dataset(
        {"elevation": (("y", "x"), np.ones((4, 4), dtype=np.float32))},
        coords={"y": np.arange(4), "x": np.arange(4) + 1},
    )

    with pytest.raises(ValueError, match="not aligned"):
        SampleIndexBuilder(
            config=config,
            source_datasets={
                "radklim": _dataset(times),
                "aux": auxiliary_dataset,
            },
            source_checksums={"radklim": "abc123", "aux": "def456"},
            catalog=SQLiteSampleCatalog(tmp_path / "samples.sqlite"),
        )
