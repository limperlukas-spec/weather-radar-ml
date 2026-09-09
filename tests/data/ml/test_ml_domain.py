from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from weather_radar_ml.data.ml import (
    RadarInputs,
    RadarSample,
    RadarTargets,
    SampleContext,
    SampleSelection,
    SpatialWindow,
    SpatialWindowMode,
    TemporalSplitRule,
)


def _times(count: int, *, start_minute: int = 0) -> tuple[datetime, ...]:
    start = datetime(2023, 9, 1, tzinfo=UTC) + timedelta(minutes=start_minute)
    return tuple(start + timedelta(minutes=5 * index) for index in range(count))


def _sample() -> RadarSample:
    inputs = RadarInputs(
        dynamic=np.zeros((12, 1, 4, 5), dtype=np.float64),
        dynamic_features=("precipitation_rate",),
        static=np.ones((2, 4, 5), dtype=np.float32),
        static_features=("elevation", "slope"),
    )
    targets = RadarTargets(
        dynamic=np.zeros((24, 1, 4, 5), dtype=np.float64),
        features=("precipitation_rate",),
    )
    context = SampleContext(
        sample_id="20230901T000000_x0000_y0000_deadbeef",
        dataset_id="radklim-yw-60m-120m-v1-deadbeef",
        input_times=_times(12),
        target_times=_times(24, start_minute=60),
        spatial_window=SpatialWindow(x=0, y=0, width=5, height=4),
        metadata={"split": "train"},
    )
    return RadarSample(inputs=inputs, targets=targets, context=context)


def test_spatial_window_exposes_slices_and_mode() -> None:
    window = SpatialWindow(
        x=10,
        y=20,
        width=5,
        height=4,
        mode=SpatialWindowMode.FULL_FRAME,
    )

    assert window.x_slice == slice(10, 15)
    assert window.y_slice == slice(20, 24)
    assert window.mode is SpatialWindowMode.FULL_FRAME


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x": -1, "y": 0, "width": 1, "height": 1}, "non-negative"),
        ({"x": 0, "y": 0, "width": 0, "height": 1}, "positive"),
    ],
)
def test_spatial_window_rejects_invalid_geometry(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SpatialWindow(**kwargs)


def test_radar_inputs_use_read_only_views_without_freezing_source_array() -> None:
    source = np.zeros((2, 1, 3, 4), dtype=np.float64)

    inputs = RadarInputs(
        dynamic=source,
        dynamic_features=("precipitation_rate",),
    )

    assert inputs.dynamic.shape == (2, 1, 3, 4)
    assert not inputs.dynamic.flags.writeable
    assert source.flags.writeable
    source[0, 0, 0, 0] = 7.0
    assert inputs.dynamic[0, 0, 0, 0] == 7.0


def test_radar_inputs_validate_channels_static_shape_and_features() -> None:
    with pytest.raises(ValueError, match="2 entries"):
        RadarInputs(
            dynamic=np.zeros((2, 2, 3, 4)),
            dynamic_features=("only-one",),
        )

    with pytest.raises(ValueError, match="same spatial shape"):
        RadarInputs(
            dynamic=np.zeros((2, 1, 3, 4)),
            dynamic_features=("precipitation",),
            static=np.zeros((1, 2, 4)),
            static_features=("elevation",),
        )

    with pytest.raises(ValueError, match="require a static input"):
        RadarInputs(
            dynamic=np.zeros((2, 1, 3, 4)),
            dynamic_features=("precipitation",),
            static_features=("elevation",),
        )


def test_radar_inputs_reject_invalid_arrays_and_feature_names() -> None:
    with pytest.raises(ValueError, match="4 dimensions"):
        RadarInputs(
            dynamic=np.zeros((2, 3, 4)),
            dynamic_features=("precipitation",),
        )

    with pytest.raises(ValueError, match="empty dimensions"):
        RadarInputs(
            dynamic=np.zeros((0, 1, 3, 4)),
            dynamic_features=("precipitation",),
        )

    with pytest.raises(ValueError, match="duplicates"):
        RadarInputs(
            dynamic=np.zeros((2, 2, 3, 4)),
            dynamic_features=("precipitation", "precipitation"),
        )

    with pytest.raises(ValueError, match="empty names"):
        RadarInputs(
            dynamic=np.zeros((2, 1, 3, 4)),
            dynamic_features=(" ",),
        )


def test_radar_targets_validate_shape_and_are_read_only() -> None:
    targets = RadarTargets(
        dynamic=np.zeros((3, 1, 2, 2)),
        features=("precipitation_rate",),
    )
    assert not targets.dynamic.flags.writeable

    with pytest.raises(ValueError, match="4 dimensions"):
        RadarTargets(
            dynamic=np.zeros((3, 2, 2)),
            features=("precipitation_rate",),
        )


def test_sample_context_normalizes_text_and_copies_metadata() -> None:
    metadata = {"split": "train"}
    context = SampleContext(
        sample_id=" sample-1 ",
        dataset_id=" dataset-1 ",
        input_times=_times(2),
        target_times=_times(2, start_minute=10),
        spatial_window=SpatialWindow(x=0, y=0, width=2, height=2),
        metadata=metadata,
    )
    metadata["split"] = "test"

    assert context.sample_id == "sample-1"
    assert context.dataset_id == "dataset-1"
    assert context.metadata["split"] == "train"
    with pytest.raises(TypeError):
        context.metadata["split"] = "validation"  # type: ignore[index]


@pytest.mark.parametrize("field", ["sample_id", "dataset_id"])
def test_sample_context_rejects_empty_identity(field: str) -> None:
    values = {
        "sample_id": "sample",
        "dataset_id": "dataset",
        "input_times": _times(2),
        "target_times": _times(2, start_minute=10),
        "spatial_window": SpatialWindow(x=0, y=0, width=2, height=2),
    }
    values[field] = " "
    with pytest.raises(ValueError, match=field):
        SampleContext(**values)  # type: ignore[arg-type]


def test_sample_context_rejects_empty_or_invalid_time_sequences() -> None:
    window = SpatialWindow(x=0, y=0, width=2, height=2)

    with pytest.raises(ValueError, match="at least one input"):
        SampleContext(
            sample_id="sample",
            dataset_id="dataset",
            input_times=(),
            target_times=_times(1, start_minute=5),
            spatial_window=window,
        )

    with pytest.raises(ValueError, match="at least one target"):
        SampleContext(
            sample_id="sample",
            dataset_id="dataset",
            input_times=_times(1),
            target_times=(),
            spatial_window=window,
        )

    duplicate = (_times(1)[0], _times(1)[0])
    with pytest.raises(ValueError, match="strictly increasing"):
        SampleContext(
            sample_id="sample",
            dataset_id="dataset",
            input_times=duplicate,
            target_times=_times(1, start_minute=10),
            spatial_window=window,
        )

    with pytest.raises(ValueError, match="strictly after"):
        SampleContext(
            sample_id="sample",
            dataset_id="dataset",
            input_times=_times(2),
            target_times=_times(1, start_minute=5),
            spatial_window=window,
        )


def test_radar_sample_validates_time_and_spatial_axes() -> None:
    sample = _sample()
    assert sample.inputs.dynamic.shape == (12, 1, 4, 5)
    assert sample.targets.dynamic.shape == (24, 1, 4, 5)

    with pytest.raises(ValueError, match="input timestamps"):
        RadarSample(
            inputs=sample.inputs,
            targets=sample.targets,
            context=SampleContext(
                sample_id="sample-2",
                dataset_id=sample.context.dataset_id,
                input_times=_times(11),
                target_times=sample.context.target_times,
                spatial_window=sample.context.spatial_window,
            ),
        )

    wrong_window = SpatialWindow(x=0, y=0, width=6, height=4)
    with pytest.raises(ValueError, match="Input spatial shape"):
        RadarSample(
            inputs=sample.inputs,
            targets=sample.targets,
            context=SampleContext(
                sample_id="sample-3",
                dataset_id=sample.context.dataset_id,
                input_times=sample.context.input_times,
                target_times=sample.context.target_times,
                spatial_window=wrong_window,
            ),
        )

    wrong_targets = RadarTargets(
        dynamic=np.zeros((24, 1, 3, 5)),
        features=("precipitation_rate",),
    )
    with pytest.raises(ValueError, match="Target spatial shape"):
        RadarSample(
            inputs=sample.inputs,
            targets=wrong_targets,
            context=sample.context,
        )


def test_sample_selection_is_typed_and_validated() -> None:
    selection = SampleSelection(
        split=" train ",
        input_event_classes=("dry", "light"),
        target_event_classes=("heavy", "extreme"),
        start_time=_times(1)[0],
        end_time=_times(1, start_minute=60)[0],
    )
    assert selection.split == "train"

    with pytest.raises(ValueError, match="split"):
        SampleSelection(split=" ")
    with pytest.raises(ValueError, match="before"):
        SampleSelection(start_time=_times(1)[0], end_time=_times(1)[0])
    with pytest.raises(ValueError, match="duplicates"):
        SampleSelection(input_event_classes=("dry", "dry"))


def test_temporal_split_rule_uses_half_open_valid_interval() -> None:
    start = _times(1)[0]
    end = _times(1, start_minute=60)[0]
    rule = TemporalSplitRule(name=" train ", start=start, end=end)
    assert rule.name == "train"

    with pytest.raises(ValueError, match="name"):
        TemporalSplitRule(name=" ", start=start, end=end)
    with pytest.raises(ValueError, match="before"):
        TemporalSplitRule(name="train", start=start, end=start)
