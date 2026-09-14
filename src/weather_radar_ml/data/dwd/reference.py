"""Reference-month preparation helpers for the RADKLIM-YW 0.6 benchmark."""

from __future__ import annotations

import calendar
import itertools
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from weather_radar_ml.data.domain import RadarFrame, SpatialGrid
from weather_radar_ml.data.dwd.archive import extract_day
from weather_radar_ml.data.dwd.radklim_yw import RadklimYwDecoder


@dataclass(frozen=True, slots=True)
class RadarRoi:
    """One fixed pixel window on the native RADKLIM grid."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI origin must be non-negative.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be greater than zero.")


@dataclass(frozen=True, slots=True)
class PreparedMonthSummary:
    """Auditable temporal summary of one prepared monthly ROI artifact."""

    frames: int
    first_timestamp: datetime
    last_timestamp: datetime
    missing_frame_slots: int
    rejected_incomplete_frames: int

    def __post_init__(self) -> None:
        if self.frames <= 0:
            raise ValueError("frames must be greater than zero.")
        if self.first_timestamp.tzinfo is None or self.last_timestamp.tzinfo is None:
            raise ValueError("Prepared timestamps must be timezone-aware.")
        if self.first_timestamp > self.last_timestamp:
            raise ValueError("first_timestamp must not be after last_timestamp.")
        if self.missing_frame_slots < 0:
            raise ValueError("missing_frame_slots must not be negative.")
        if self.rejected_incomplete_frames < 0:
            raise ValueError("rejected_incomplete_frames must not be negative.")


# Dortmund city centre (approximately 51.5136 N, 7.4653 E) maps to native
# RADKLIM pixel (x=260, y=613). A 128 x 128 window starting at (196, 549)
# therefore places Dortmund close to the centre while preserving the native 1 km grid.
DORTMUND_REFERENCE_ROI = RadarRoi(x=196, y=549, width=128, height=128)


def crop_frame(frame: RadarFrame, roi: RadarRoi) -> RadarFrame:
    """Crop one canonical radar frame to a fixed native-grid ROI."""
    x_end = roi.x + roi.width
    y_end = roi.y + roi.height
    if x_end > frame.grid.x.size or y_end > frame.grid.y.size:
        raise ValueError("ROI must fit inside the radar frame grid.")

    source_flags = (
        None
        if frame.source_flags is None
        else frame.source_flags[roi.y : y_end, roi.x : x_end].copy()
    )
    return RadarFrame(
        timestamp=frame.timestamp,
        precipitation_rate=frame.precipitation_rate[
            roi.y : y_end,
            roi.x : x_end,
        ].copy(),
        quality=frame.quality[roi.y : y_end, roi.x : x_end].copy(),
        source_flags=source_flags,
        grid=SpatialGrid(
            x=frame.grid.x[roi.x : x_end].copy(),
            y=frame.grid.y[roi.y : y_end].copy(),
            crs=frame.grid.crs,
        ),
    )


def prepare_monthly_roi(
    archive: Path,
    destination: Path,
    *,
    year: int,
    month: int,
    roi: RadarRoi,
) -> PreparedMonthSummary:
    """Decode one monthly archive into one atomic cropped prepared Zarr store.

    Daily nested archives are extracted into temporary storage. Full native frames are
    decoded one at a time and immediately cropped before they are retained, keeping the
    working set bounded while preserving missing timestamps as gaps in the time axis.
    """
    if destination.exists():
        raise FileExistsError(f"Prepared destination already exists: {destination}")
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    shutil.rmtree(staging)
    decoder = RadklimYwDecoder()
    timestamps: list[datetime] = []
    rejected_incomplete = 0
    first_chunk = True

    try:
        with tempfile.TemporaryDirectory(prefix="radklim-reference-extract-") as tmp:
            extract_root = Path(tmp)
            days = calendar.monthrange(year, month)[1]
            for day_number in range(1, days + 1):
                day = date(year, month, day_number)
                day_root = extract_root / day.isoformat()
                frame_paths = extract_day(archive, day, day_root)
                frames: list[RadarFrame] = []
                for path in frame_paths:
                    frame = crop_frame(decoder.decode_frame(path), roi)
                    values = frame.precipitation_rate
                    if not bool(np.isfinite(values).all()) or bool(
                        (values < 0.0).any()
                    ):
                        rejected_incomplete += 1
                        continue
                    frames.append(frame)
                frames.sort(key=lambda frame: frame.timestamp)
                if not frames:
                    shutil.rmtree(day_root, ignore_errors=True)
                    continue
                day_dataset = xr.concat(
                    [frame.to_dataset() for frame in frames],
                    dim="time",
                    combine_attrs="identical",
                )
                _validate_chunk_times(day_dataset, timestamps)
                timestamps.extend(frame.timestamp.astimezone(UTC) for frame in frames)
                if first_chunk:
                    day_dataset.to_zarr(
                        staging,
                        mode="w",
                        encoding=_reference_encoding(day_dataset),
                        consolidated=False,
                        zarr_format=2,
                    )
                    first_chunk = False
                else:
                    day_dataset.to_zarr(
                        staging,
                        mode="a",
                        append_dim="time",
                        consolidated=False,
                        zarr_format=2,
                    )
                day_dataset.close()
                shutil.rmtree(day_root, ignore_errors=True)

        if not timestamps:
            raise ValueError("Monthly archive produced no RADKLIM-YW frames.")
        missing = _missing_frame_slots(timestamps, step=timedelta(minutes=5))
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return PreparedMonthSummary(
        frames=len(timestamps),
        first_timestamp=timestamps[0],
        last_timestamp=timestamps[-1],
        missing_frame_slots=missing,
        rejected_incomplete_frames=rejected_incomplete,
    )


def _validate_chunk_times(
    dataset: xr.Dataset,
    previous: list[datetime],
) -> None:
    raw = np.asarray(dataset.coords["time"].values)
    if raw.size == 0:
        raise ValueError("Prepared Zarr chunk must contain at least one timestamp.")
    values = tuple(_datetime64_to_utc(value) for value in raw)
    if any(current >= following for current, following in itertools.pairwise(values)):
        raise ValueError("Prepared timestamps must be strictly increasing.")
    if previous and values[0] <= previous[-1]:
        raise ValueError("Prepared monthly chunks must be strictly chronological.")


def _reference_encoding(dataset: xr.Dataset) -> dict[str, dict[str, object]]:
    time_chunk = min(12, int(dataset.sizes["time"]))
    y_chunk = min(64, int(dataset.sizes["y"]))
    x_chunk = min(64, int(dataset.sizes["x"]))
    chunks = (time_chunk, y_chunk, x_chunk)
    return {str(name): {"chunks": chunks} for name in dataset.data_vars}


def _missing_frame_slots(values: list[datetime], *, step: timedelta) -> int:
    missing = 0
    for current, following in itertools.pairwise(values):
        delta = following - current
        if delta <= timedelta(0):
            raise ValueError("Prepared timestamps must be strictly increasing.")
        slots, remainder = divmod(delta, step)
        if remainder != timedelta(0):
            raise ValueError("Prepared timestamps must stay on the five-minute grid.")
        missing += max(int(slots) - 1, 0)
    return missing


def _datetime64_to_utc(value: np.datetime64) -> datetime:
    if np.isnat(value):
        raise ValueError("Prepared time coordinate contains NaT.")
    nanoseconds = int(value.astype("datetime64[ns]").astype(np.int64))
    return datetime.fromtimestamp(nanoseconds / 1_000_000_000, tz=UTC)
