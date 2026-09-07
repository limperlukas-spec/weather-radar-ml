"""Persistence adapters for prepared radar data."""

from pathlib import Path
from typing import cast

import xarray as xr

from weather_radar_ml.data.domain import RadarSequence


def write_zarr(sequence: RadarSequence, path: Path) -> None:
    """Write a radar sequence to a Zarr store, replacing an existing store."""
    sequence.to_dataset().to_zarr(
        path,
        mode="w",
        zarr_format=2,
    )


def open_zarr(path: Path) -> xr.Dataset:
    """Open prepared radar data lazily from a Zarr store."""
    return cast(xr.Dataset, xr.open_zarr(path))
