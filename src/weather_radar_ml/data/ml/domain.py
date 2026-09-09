"""Framework-neutral domain models for ML-ready radar samples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]


class SpatialWindowMode(StrEnum):
    """How a spatial window relates to the prepared source domain."""

    PATCH = "patch"
    FULL_FRAME = "full_frame"


@dataclass(frozen=True, slots=True)
class SpatialWindow:
    """Integer pixel window on the primary prepared source grid."""

    x: int
    y: int
    width: int
    height: int
    mode: SpatialWindowMode = SpatialWindowMode.PATCH

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("Spatial window coordinates must be non-negative.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Spatial window dimensions must be positive.")

    @property
    def x_slice(self) -> slice:
        """Return the NumPy/Zarr slice for the x dimension."""
        return slice(self.x, self.x + self.width)

    @property
    def y_slice(self) -> slice:
        """Return the NumPy/Zarr slice for the y dimension."""
        return slice(self.y, self.y + self.height)


@dataclass(frozen=True, slots=True)
class RadarInputs:
    """Dynamic and optional static inputs for one radar sample.

    Dynamic arrays use ``[T, C, H, W]``. Static arrays use ``[C, H, W]``.
    """

    dynamic: Array
    dynamic_features: tuple[str, ...]
    static: Array | None = None
    static_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        dynamic = _readonly_array(self.dynamic, ndim=4, name="dynamic inputs")
        dynamic_features = _validate_feature_names(
            self.dynamic_features,
            expected_count=dynamic.shape[1],
            name="dynamic input features",
        )

        static = self.static
        static_features = tuple(self.static_features)
        if static is None:
            if static_features:
                raise ValueError("Static feature names require a static input array.")
        else:
            static = _readonly_array(static, ndim=3, name="static inputs")
            static_features = _validate_feature_names(
                static_features,
                expected_count=static.shape[0],
                name="static input features",
            )
            if static.shape[1:] != dynamic.shape[2:]:
                raise ValueError(
                    "Static and dynamic inputs must use the same spatial shape."
                )

        object.__setattr__(self, "dynamic", dynamic)
        object.__setattr__(self, "dynamic_features", dynamic_features)
        object.__setattr__(self, "static", static)
        object.__setattr__(self, "static_features", static_features)


@dataclass(frozen=True, slots=True)
class RadarTargets:
    """Forecast targets for one sample using ``[T, C, H, W]``."""

    dynamic: Array
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        dynamic = _readonly_array(self.dynamic, ndim=4, name="targets")
        features = _validate_feature_names(
            self.features,
            expected_count=dynamic.shape[1],
            name="target features",
        )
        object.__setattr__(self, "dynamic", dynamic)
        object.__setattr__(self, "features", features)


@dataclass(frozen=True, slots=True)
class SampleContext:
    """Identity, temporal coordinates, window and immutable sample metadata."""

    sample_id: str
    dataset_id: str
    input_times: tuple[datetime, ...]
    target_times: tuple[datetime, ...]
    spatial_window: SpatialWindow
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_id = self.sample_id.strip()
        dataset_id = self.dataset_id.strip()
        if not sample_id:
            raise ValueError("sample_id must not be empty.")
        if not dataset_id:
            raise ValueError("dataset_id must not be empty.")

        input_times = tuple(self.input_times)
        target_times = tuple(self.target_times)
        if not input_times:
            raise ValueError("A radar sample must contain at least one input time.")
        if not target_times:
            raise ValueError("A radar sample must contain at least one target time.")
        _validate_strictly_increasing(input_times, "input_times")
        _validate_strictly_increasing(target_times, "target_times")
        if input_times[-1] >= target_times[0]:
            raise ValueError("Targets must start strictly after the input history.")

        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "input_times", input_times)
        object.__setattr__(self, "target_times", target_times)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class RadarSample:
    """One framework-neutral ML sample backed by NumPy arrays."""

    inputs: RadarInputs
    targets: RadarTargets
    context: SampleContext

    def __post_init__(self) -> None:
        if len(self.context.input_times) != self.inputs.dynamic.shape[0]:
            raise ValueError(
                "Number of input timestamps must match the dynamic input time axis."
            )
        if len(self.context.target_times) != self.targets.dynamic.shape[0]:
            raise ValueError(
                "Number of target timestamps must match the target time axis."
            )

        expected_spatial_shape = (
            self.context.spatial_window.height,
            self.context.spatial_window.width,
        )
        if self.inputs.dynamic.shape[2:] != expected_spatial_shape:
            raise ValueError(
                "Input spatial shape must match the declared spatial window."
            )
        if self.targets.dynamic.shape[2:] != expected_spatial_shape:
            raise ValueError(
                "Target spatial shape must match the declared spatial window."
            )


@dataclass(frozen=True, slots=True)
class SampleSelection:
    """Typed, storage-independent selection criteria for a sample catalog."""

    split: str | None = None
    input_event_classes: tuple[str, ...] = ()
    target_event_classes: tuple[str, ...] = ()
    start_time: datetime | None = None
    end_time: datetime | None = None

    def __post_init__(self) -> None:
        split = self.split.strip() if self.split is not None else None
        if split == "":
            raise ValueError("split must not be empty when provided.")
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time >= self.end_time
        ):
            raise ValueError("Selection start_time must be before end_time.")

        object.__setattr__(self, "split", split)
        object.__setattr__(
            self,
            "input_event_classes",
            _validate_names(self.input_event_classes, "input event classes"),
        )
        object.__setattr__(
            self,
            "target_event_classes",
            _validate_names(self.target_event_classes, "target event classes"),
        )


@dataclass(frozen=True, slots=True)
class TemporalSplitRule:
    """Half-open ``[start, end)`` interval assigned to one named split."""

    name: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("Split rule name must not be empty.")
        if self.start >= self.end:
            raise ValueError("Split rule start must be before end.")
        object.__setattr__(self, "name", name)


def _readonly_array(array: Array, *, ndim: int, name: str) -> Array:
    value = np.asarray(array)
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}.")
    if any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} must not contain empty dimensions.")
    view = value.view()
    view.setflags(write=False)
    return view


def _validate_feature_names(
    names: tuple[str, ...],
    *,
    expected_count: int,
    name: str,
) -> tuple[str, ...]:
    values = _validate_names(names, name)
    if len(values) != expected_count:
        raise ValueError(
            f"{name} must contain {expected_count} entries, got {len(values)}."
        )
    return values


def _validate_names(names: tuple[str, ...], name: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in names)
    if any(not value for value in values):
        raise ValueError(f"{name} must not contain empty names.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")
    return values


def _validate_strictly_increasing(
    values: tuple[datetime, ...],
    name: str,
) -> None:
    if any(current >= following for current, following in pairwise(values)):
        raise ValueError(f"{name} must be strictly increasing.")
