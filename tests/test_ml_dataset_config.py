from datetime import UTC, datetime

import pytest

from weather_radar_ml.config.ml_dataset import (
    EventClass,
    FeatureConfig,
    FeatureKind,
    FeatureSchemaConfig,
    PrecipEventProfileConfig,
    SpatialMode,
    SpatialSamplingConfig,
    SplitIntervalConfig,
    SplitName,
    TemporalSplitConfig,
    TemporalWindowConfig,
    ml_dataset_config_from_mapping,
)


def _mapping() -> dict:
    return {
        "name": "radklim-yw-60m-120m-v1",
        "sources": [
            {
                "source_id": "radklim",
                "artifact_id": "radklim-yw-2023-09-01-012f-v1",
                "role": "primary",
                "path": "data/prepared/radklim.zarr",
            }
        ],
        "features": {
            "dynamic_inputs": [
                {
                    "name": "precipitation",
                    "source_id": "radklim",
                    "variable": "precipitation_rate",
                }
            ],
            "targets": [
                {
                    "name": "precipitation_target",
                    "source_id": "radklim",
                    "variable": "precipitation_rate",
                }
            ],
        },
        "temporal": {
            "step_minutes": 5,
            "history_minutes": 60,
            "forecast_minutes": 120,
        },
        "spatial": {"mode": "full_frame"},
        "splits": {
            "intervals": [
                {
                    "name": "train",
                    "start": "2023-01-01T00:00:00Z",
                    "end": "2023-08-01T00:00:00Z",
                },
                {
                    "name": "validation",
                    "start": "2023-08-05T00:00:00Z",
                    "end": "2023-10-01T00:00:00Z",
                },
                {
                    "name": "test",
                    "start": "2023-10-05T00:00:00Z",
                    "end": "2024-01-01T00:00:00Z",
                },
            ]
        },
    }


def test_mapping_builds_resolved_config_with_defaults() -> None:
    config = ml_dataset_config_from_mapping(_mapping())

    assert config.temporal.history_steps == 12
    assert config.temporal.forecast_steps == 24
    assert config.events.profile_id == "precip_event_v1"
    assert config.events.thresholds[-1].event_class is EventClass.HEAVY

    resolved = config.to_resolved_dict()
    assert resolved["spatial"]["mode"] == "full_frame"
    assert resolved["events"]["min_wet_fraction"] == 0.01
    assert resolved["sources"][0]["path"] == "data/prepared/radklim.zarr"


def test_unknown_keys_are_rejected_strictly() -> None:
    data = _mapping()
    data["stride_xx"] = 128

    with pytest.raises(ValueError, match="Unknown dataset config keys"):
        ml_dataset_config_from_mapping(data)


def test_exactly_one_primary_source_is_required() -> None:
    data = _mapping()
    data["sources"][0]["role"] = "auxiliary"

    with pytest.raises(ValueError, match="Exactly one primary"):
        ml_dataset_config_from_mapping(data)


def test_features_must_reference_known_source() -> None:
    data = _mapping()
    data["features"]["dynamic_inputs"][0]["source_id"] = "missing"

    with pytest.raises(ValueError, match="unknown source"):
        ml_dataset_config_from_mapping(data)


def test_feature_order_is_preserved() -> None:
    data = _mapping()
    data["features"]["dynamic_inputs"].append(
        {
            "name": "second",
            "source_id": "radklim",
            "variable": "quality_flag",
        }
    )

    config = ml_dataset_config_from_mapping(data)

    assert [feature.name for feature in config.features.dynamic_inputs] == [
        "precipitation",
        "second",
    ]


def test_temporal_window_requires_exact_step_divisibility() -> None:
    with pytest.raises(ValueError, match="history_minutes"):
        TemporalWindowConfig(
            step_minutes=5,
            history_minutes=61,
            forecast_minutes=120,
        )


def test_patch_mode_requires_complete_positive_geometry() -> None:
    with pytest.raises(ValueError, match="requires"):
        SpatialSamplingConfig(mode=SpatialMode.PATCH, patch_height=128)

    config = SpatialSamplingConfig(
        mode=SpatialMode.PATCH,
        patch_height=128,
        patch_width=256,
        stride_y=64,
        stride_x=128,
    )
    assert config.patch_height == 128


def test_full_frame_rejects_patch_parameters() -> None:
    with pytest.raises(ValueError, match="must be omitted"):
        SpatialSamplingConfig(mode=SpatialMode.FULL_FRAME, stride_x=128)


def test_split_intervals_allow_gaps_but_not_overlap() -> None:
    valid = TemporalSplitConfig(
        (
            SplitIntervalConfig(
                SplitName.TRAIN,
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 2, 1, tzinfo=UTC),
            ),
            SplitIntervalConfig(
                SplitName.VALIDATION,
                datetime(2023, 2, 2, tzinfo=UTC),
                datetime(2023, 3, 1, tzinfo=UTC),
            ),
            SplitIntervalConfig(
                SplitName.TEST,
                datetime(2023, 3, 2, tzinfo=UTC),
                datetime(2023, 4, 1, tzinfo=UTC),
            ),
        )
    )
    assert len(valid.intervals) == 3

    with pytest.raises(ValueError, match="must not overlap"):
        TemporalSplitConfig(
            (
                SplitIntervalConfig(
                    SplitName.TRAIN,
                    datetime(2023, 1, 1, tzinfo=UTC),
                    datetime(2023, 2, 5, tzinfo=UTC),
                ),
                SplitIntervalConfig(
                    SplitName.VALIDATION,
                    datetime(2023, 2, 1, tzinfo=UTC),
                    datetime(2023, 3, 1, tzinfo=UTC),
                ),
                SplitIntervalConfig(
                    SplitName.TEST,
                    datetime(2023, 3, 2, tzinfo=UTC),
                    datetime(2023, 4, 1, tzinfo=UTC),
                ),
            )
        )


def test_split_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        SplitIntervalConfig(
            SplitName.TRAIN,
            datetime(2023, 1, 1),
            datetime(2023, 2, 1, tzinfo=UTC),
        )


def test_precip_event_v1_has_five_result_classes() -> None:
    profile = PrecipEventProfileConfig()

    assert EventClass.DRY.value == "dry"
    assert EventClass.EXTREME.value == "extreme"
    assert [threshold.event_class for threshold in profile.thresholds] == [
        EventClass.LIGHT,
        EventClass.MODERATE,
        EventClass.HEAVY,
    ]


def test_feature_schema_rejects_duplicate_feature_names() -> None:
    feature = FeatureConfig(
        "precipitation",
        "radklim",
        "precipitation_rate",
        FeatureKind.DYNAMIC,
    )

    with pytest.raises(ValueError, match="Feature names must be unique"):
        FeatureSchemaConfig(
            dynamic_inputs=(feature,),
            targets=(feature,),
        )
