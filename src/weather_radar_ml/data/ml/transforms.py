"""Framework-neutral transformations for ML-ready radar samples."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import DTypeLike

from weather_radar_ml.data.ml.contracts import SampleTransform
from weather_radar_ml.data.ml.domain import RadarInputs, RadarSample, RadarTargets


class ArrayTransform(Protocol):
    """Transform one NumPy array without changing its semantic axes."""

    def transform(self, array: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class Compose:
    """Apply sample transforms sequentially."""

    transforms: tuple[SampleTransform, ...]

    def __init__(self, transforms: Sequence[SampleTransform]) -> None:
        object.__setattr__(self, "transforms", tuple(transforms))

    def transform(self, sample: RadarSample) -> RadarSample:
        transformed = sample
        for transform in self.transforms:
            transformed = transform.transform(transformed)
        return transformed


@dataclass(frozen=True, slots=True)
class InputTargetTransform:
    """Apply independent transform pipelines to inputs and targets.

    Dynamic and static inputs may be transformed separately. Targets use their
    own pipeline. Sample context and feature ordering remain unchanged.
    """

    dynamic_inputs: tuple[ArrayTransform, ...] = ()
    static_inputs: tuple[ArrayTransform, ...] = ()
    targets: tuple[ArrayTransform, ...] = ()

    def __init__(
        self,
        *,
        dynamic_inputs: Sequence[ArrayTransform] = (),
        static_inputs: Sequence[ArrayTransform] = (),
        targets: Sequence[ArrayTransform] = (),
    ) -> None:
        object.__setattr__(self, "dynamic_inputs", tuple(dynamic_inputs))
        object.__setattr__(self, "static_inputs", tuple(static_inputs))
        object.__setattr__(self, "targets", tuple(targets))

    def transform(self, sample: RadarSample) -> RadarSample:
        dynamic = _apply_array_transforms(
            sample.inputs.dynamic,
            self.dynamic_inputs,
        )
        static = sample.inputs.static
        if static is not None:
            static = _apply_array_transforms(static, self.static_inputs)
        target_dynamic = _apply_array_transforms(
            sample.targets.dynamic,
            self.targets,
        )

        return replace(
            sample,
            inputs=RadarInputs(
                dynamic=dynamic,
                dynamic_features=sample.inputs.dynamic_features,
                static=static,
                static_features=sample.inputs.static_features,
            ),
            targets=RadarTargets(
                dynamic=target_dynamic,
                features=sample.targets.features,
            ),
        )


@dataclass(frozen=True, slots=True)
class CastDType:
    """Cast an array to one explicit NumPy dtype."""

    dtype: np.dtype

    def __init__(self, dtype: DTypeLike) -> None:
        object.__setattr__(self, "dtype", np.dtype(dtype))

    def transform(self, array: np.ndarray) -> np.ndarray:
        return array.astype(self.dtype, copy=False)


class FittableArrayTransform(Protocol):
    """Array transform with explicit fitting from an array iterable."""

    def fit(self, arrays: Iterable[np.ndarray]) -> None: ...

    def transform(self, array: np.ndarray) -> np.ndarray: ...


def _apply_array_transforms(
    array: np.ndarray,
    transforms: tuple[ArrayTransform, ...],
) -> np.ndarray:
    transformed = array
    expected_shape = array.shape
    for transform in transforms:
        transformed = np.asarray(transform.transform(transformed))
        if transformed.shape != expected_shape:
            raise ValueError(
                "Array transforms must preserve tensor shape; "
                f"expected {expected_shape}, got {transformed.shape}."
            )
    return transformed
