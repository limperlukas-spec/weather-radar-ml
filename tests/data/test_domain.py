from datetime import datetime, timedelta

import numpy as np
import pytest

from weather_radar_ml.data.domain import RadarFrame, RadarSequence, SpatialGrid
from weather_radar_ml.data.quality import ObservationQuality


def _grid() -> SpatialGrid:
    return SpatialGrid(
        x=np.array([0.0, 1000.0]),
        y=np.array([0.0, 1000.0]),
        crs="EPSG:3034",
    )


def _frame(minute: int, grid: SpatialGrid | None = None) -> RadarFrame:
    return RadarFrame(
        timestamp=datetime(2026, 1, 1, 12, minute),
        precipitation_rate=np.ones((2, 2), dtype=np.float32),
        quality=np.full((2, 2), ObservationQuality.OBSERVED, dtype=np.uint8),
        grid=grid or _grid(),
    )


def test_frame_converts_to_named_xarray_dimensions() -> None:
    dataset = _frame(0).to_dataset()

    assert dataset["precipitation_rate"].dims == ("time", "y", "x")
    assert dataset["quality"].dtype == np.uint8
    assert dataset.attrs["schema_version"] == "1"


def test_frame_rejects_unknown_quality_code() -> None:
    quality = np.full((2, 2), 99, dtype=np.uint8)

    with pytest.raises(ValueError, match="unknown provenance"):
        RadarFrame(
            timestamp=datetime(2026, 1, 1),
            precipitation_rate=np.zeros((2, 2)),
            quality=quality,
            grid=_grid(),
        )


def test_sequence_rejects_temporal_gap() -> None:
    with pytest.raises(ValueError, match="temporal gap"):
        RadarSequence((_frame(0), _frame(10)), step=timedelta(minutes=5))


def test_sequence_rejects_incompatible_grid() -> None:
    other = SpatialGrid(
        x=np.array([0.0, 2000.0]),
        y=np.array([0.0, 1000.0]),
        crs="EPSG:3034",
    )

    with pytest.raises(ValueError, match="same spatial grid"):
        RadarSequence(
            (_frame(0), _frame(5, other)),
            step=timedelta(minutes=5),
        )


def test_quality_states_keep_imputation_distinguishable() -> None:
    assert ObservationQuality.OBSERVED != ObservationQuality.INTERPOLATED
    assert ObservationQuality.INTERPOLATED != ObservationQuality.EXTRAPOLATED
    assert ObservationQuality.MISSING != ObservationQuality.OBSERVED
