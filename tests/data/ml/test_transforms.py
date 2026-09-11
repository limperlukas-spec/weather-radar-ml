from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pytest

from weather_radar_ml.data.ml.domain import (
    RadarInputs,
    RadarSample,
    RadarTargets,
    SampleContext,
    SpatialWindow,
)
from weather_radar_ml.data.ml.transforms import (
    CastDType,
    Compose,
    InputTargetTransform,
)


def _sample() -> RadarSample:
    dynamic = np.arange(2 * 2 * 2 * 3).reshape(2, 2, 2, 3)
    static = np.arange(2 * 2 * 3).reshape(2, 2, 3)
    targets = np.arange(2 * 1 * 2 * 3).reshape(2, 1, 2, 3)
    return RadarSample(
        inputs=RadarInputs(
            dynamic=dynamic,
            dynamic_features=("rain", "temperature"),
            static=static,
            static_features=("height", "land_use"),
        ),
        targets=RadarTargets(
            dynamic=targets,
            features=("rain_target",),
        ),
        context=SampleContext(
            sample_id="sample",
            dataset_id="dataset",
            input_times=(
                datetime(2023, 1, 1, 0, 0),
                datetime(2023, 1, 1, 0, 5),
            ),
            target_times=(
                datetime(2023, 1, 1, 0, 10),
                datetime(2023, 1, 1, 0, 15),
            ),
            spatial_window=SpatialWindow(x=0, y=0, width=3, height=2),
            metadata={"split": "train"},
        ),
    )


def test_cast_dtype_is_explicit_and_shape_preserving() -> None:
    array = np.arange(12, dtype=np.int16).reshape(3, 4)

    transformed = CastDType(np.float32).transform(array)

    assert transformed.dtype == np.float32
    assert transformed.shape == array.shape


def test_input_target_transform_can_cast_all_paths() -> None:
    sample = _sample()
    transform = InputTargetTransform(
        dynamic_inputs=(CastDType(np.float32),),
        static_inputs=(CastDType(np.float64),),
        targets=(CastDType(np.float32),),
    )

    transformed = transform.transform(sample)

    assert transformed.inputs.dynamic.dtype == np.float32
    assert transformed.inputs.static is not None
    assert transformed.inputs.static.dtype == np.float64
    assert transformed.targets.dynamic.dtype == np.float32


def test_input_target_transform_preserves_context_and_feature_order() -> None:
    sample = _sample()

    transformed = InputTargetTransform(
        dynamic_inputs=(CastDType(np.float32),),
    ).transform(sample)

    assert transformed.context is sample.context
    assert transformed.inputs.dynamic_features == ("rain", "temperature")
    assert transformed.inputs.static_features == ("height", "land_use")
    assert transformed.targets.features == ("rain_target",)


def test_transform_does_not_mutate_original_sample() -> None:
    sample = _sample()
    original_dtype = sample.inputs.dynamic.dtype

    transformed = InputTargetTransform(
        dynamic_inputs=(CastDType(np.float32),),
    ).transform(sample)

    assert sample.inputs.dynamic.dtype == original_dtype
    assert transformed is not sample


@dataclass
class AddValue:
    value: float

    def transform(self, array: np.ndarray) -> np.ndarray:
        return array + self.value


def test_multiple_array_transforms_follow_declared_order() -> None:
    sample = _sample()

    transformed = InputTargetTransform(
        dynamic_inputs=(AddValue(2.0), AddValue(3.0)),
    ).transform(sample)

    np.testing.assert_array_equal(
        transformed.inputs.dynamic,
        sample.inputs.dynamic + 5.0,
    )


@dataclass
class DropLastColumn:
    def transform(self, array: np.ndarray) -> np.ndarray:
        return array[..., :-1]


def test_array_transform_must_preserve_shape() -> None:
    sample = _sample()

    with pytest.raises(ValueError, match="must preserve tensor shape"):
        InputTargetTransform(
            dynamic_inputs=(DropLastColumn(),),
        ).transform(sample)


@dataclass
class AddMetadata:
    key: str
    value: object

    def transform(self, sample: RadarSample) -> RadarSample:
        from dataclasses import replace

        metadata = dict(sample.context.metadata)
        metadata[self.key] = self.value
        return replace(
            sample,
            context=replace(sample.context, metadata=metadata),
        )


def test_compose_applies_sample_transforms_in_order() -> None:
    sample = _sample()

    transformed = Compose(
        (
            AddMetadata("first", 1),
            AddMetadata("second", 2),
        )
    ).transform(sample)

    assert transformed.context.metadata["first"] == 1
    assert transformed.context.metadata["second"] == 2


def test_compose_empty_pipeline_is_identity() -> None:
    sample = _sample()

    assert Compose(()).transform(sample) is sample


def test_static_pipeline_is_skipped_when_sample_has_no_static_inputs() -> None:
    sample = _sample()
    without_static = RadarSample(
        inputs=RadarInputs(
            dynamic=sample.inputs.dynamic,
            dynamic_features=sample.inputs.dynamic_features,
        ),
        targets=sample.targets,
        context=sample.context,
    )

    transformed = InputTargetTransform(
        static_inputs=(CastDType(np.float32),),
    ).transform(without_static)

    assert transformed.inputs.static is None
    assert transformed.inputs.static_features == ()
