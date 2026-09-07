from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from weather_radar_ml.data.domain import RadarFrame, RadarSequence, SpatialGrid
from weather_radar_ml.data.quality import ObservationQuality
from weather_radar_ml.data.store import open_zarr, write_zarr


def test_zarr_round_trip(tmp_path: Path) -> None:
    grid = SpatialGrid(
        x=np.array([0.0, 1000.0]),
        y=np.array([0.0, 1000.0]),
        crs="EPSG:3034",
    )
    frame = RadarFrame(
        timestamp=datetime(2026, 1, 1, 12, 0),
        precipitation_rate=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        quality=np.full((2, 2), ObservationQuality.OBSERVED, dtype=np.uint8),
        grid=grid,
    )
    sequence = RadarSequence((frame,), step=timedelta(minutes=5))
    path = tmp_path / "prepared.zarr"

    write_zarr(sequence, path)
    actual = open_zarr(path)

    xr.testing.assert_identical(actual.load(), sequence.to_dataset())
    actual.close()
