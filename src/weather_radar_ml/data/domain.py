"""Framework-independent domain types for radar data."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import xarray as xr

from weather_radar_ml.data.quality import ObservationQuality

PRECIPITATION_VARIABLE = "precipitation_rate"
QUALITY_VARIABLE = "quality"
SOURCE_FLAGS_VARIABLE = "source_flags"


@dataclass(frozen=True)
class SpatialGrid:
    """A regular projected spatial grid represented by x/y cell coordinates."""

    x: np.ndarray
    y: np.ndarray
    crs: str

    def __post_init__(self) -> None:
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("Grid coordinates must be one-dimensional.")
        if self.x.size == 0 or self.y.size == 0:
            raise ValueError("Grid coordinates must not be empty.")
        if not self.crs.strip():
            raise ValueError("Grid CRS must not be empty.")


@dataclass(frozen=True)
class RadarFrame:
    """One canonical precipitation observation on a spatial grid."""

    timestamp: datetime
    precipitation_rate: np.ndarray
    quality: np.ndarray
    grid: SpatialGrid
    source_flags: np.ndarray | None = None

    def __post_init__(self) -> None:
        expected_shape = (self.grid.y.size, self.grid.x.size)
        if self.precipitation_rate.shape != expected_shape:
            raise ValueError("Precipitation shape does not match the spatial grid.")
        if self.quality.shape != expected_shape:
            raise ValueError("Quality shape does not match the spatial grid.")
        if self.source_flags is not None and self.source_flags.shape != expected_shape:
            raise ValueError("Source flags shape does not match the spatial grid.")
        valid_codes = np.array([item.value for item in ObservationQuality])
        if not np.isin(self.quality, valid_codes).all():
            raise ValueError("Quality contains an unknown provenance code.")

    def to_dataset(self) -> xr.Dataset:
        """Convert the frame to the canonical xarray representation."""
        return xr.Dataset(
            data_vars={
                PRECIPITATION_VARIABLE: (
                    ("time", "y", "x"),
                    self.precipitation_rate[np.newaxis, ...],
                    {"units": "mm h-1"},
                ),
                QUALITY_VARIABLE: (
                    ("time", "y", "x"),
                    self.quality[np.newaxis, ...].astype(np.uint8, copy=False),
                ),
                SOURCE_FLAGS_VARIABLE: (
                    ("time", "y", "x"),
                    (
                        self.source_flags
                        if self.source_flags is not None
                        else np.zeros_like(self.quality, dtype=np.uint8)
                    )[np.newaxis, ...],
                    {"flag_masks": [1, 2], "flag_meanings": "secondary clutter"},
                ),
            },
            coords={
                "time": [np.datetime64(_naive_utc(self.timestamp), "ns")],
                "y": self.grid.y,
                "x": self.grid.x,
            },
            attrs={"crs": self.grid.crs, "schema_version": "1"},
        )


@dataclass(frozen=True)
class RadarSequence:
    """Chronological radar frames sharing one grid and a fixed time step."""

    frames: tuple[RadarFrame, ...]
    step: timedelta

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("RadarSequence must contain at least one frame.")
        if self.step <= timedelta(0):
            raise ValueError("RadarSequence step must be positive.")
        first_grid = self.frames[0].grid
        for previous, current in zip(self.frames, self.frames[1:], strict=False):
            if current.timestamp <= previous.timestamp:
                raise ValueError("Radar frames must be strictly chronological.")
            if current.timestamp - previous.timestamp != self.step:
                raise ValueError("RadarSequence contains a temporal gap.")
        for frame in self.frames[1:]:
            if not _same_grid(first_grid, frame.grid):
                raise ValueError("All radar frames must use the same spatial grid.")

    @classmethod
    def from_frames(
        cls, frames: Sequence[RadarFrame], *, step: timedelta
    ) -> "RadarSequence":
        """Create an immutable sequence from any frame sequence."""
        return cls(tuple(frames), step)

    def to_dataset(self) -> xr.Dataset:
        """Combine all frames into one canonical xarray dataset."""
        return xr.concat(
            [frame.to_dataset() for frame in self.frames],
            dim="time",
            combine_attrs="identical",
        )


def _same_grid(left: SpatialGrid, right: SpatialGrid) -> bool:
    return (
        left.crs == right.crs
        and np.array_equal(left.x, right.x)
        and np.array_equal(left.y, right.y)
    )


def _naive_utc(value: datetime) -> datetime:
    """Normalize aware timestamps to UTC for xarray/NumPy storage."""
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
