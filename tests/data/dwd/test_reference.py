from datetime import UTC, datetime

import numpy as np
import pytest

from weather_radar_ml.data.domain import RadarFrame, SpatialGrid
from weather_radar_ml.data.dwd.reference import RadarRoi, crop_frame
from weather_radar_ml.data.quality import ObservationQuality


def _frame() -> RadarFrame:
    grid = SpatialGrid(
        x=np.arange(6, dtype=np.float64) * 1000.0,
        y=np.arange(5, dtype=np.float64) * 1000.0,
        crs="test-crs",
    )
    values = np.arange(30, dtype=np.float32).reshape(5, 6)
    quality = np.full((5, 6), ObservationQuality.OBSERVED, dtype=np.uint8)
    flags = np.arange(30, dtype=np.uint8).reshape(5, 6)
    return RadarFrame(
        timestamp=datetime(2023, 9, 1, tzinfo=UTC),
        precipitation_rate=values,
        quality=quality,
        source_flags=flags,
        grid=grid,
    )


def test_crop_frame_preserves_native_coordinates_and_values() -> None:
    cropped = crop_frame(_frame(), RadarRoi(x=2, y=1, width=3, height=2))

    assert cropped.precipitation_rate.shape == (2, 3)
    np.testing.assert_array_equal(
        cropped.precipitation_rate,
        np.array([[8.0, 9.0, 10.0], [14.0, 15.0, 16.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(cropped.grid.x, np.array([2000.0, 3000.0, 4000.0]))
    np.testing.assert_array_equal(cropped.grid.y, np.array([1000.0, 2000.0]))
    assert cropped.grid.crs == "test-crs"
    assert cropped.source_flags is not None
    np.testing.assert_array_equal(
        cropped.source_flags, np.array([[8, 9, 10], [14, 15, 16]])
    )


def test_crop_frame_rejects_roi_outside_grid() -> None:
    with pytest.raises(ValueError, match="fit inside"):
        crop_frame(_frame(), RadarRoi(x=4, y=1, width=3, height=2))
