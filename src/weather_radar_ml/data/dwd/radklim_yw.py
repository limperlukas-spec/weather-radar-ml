"""Decoder for native binary RADKLIM-YW composites."""

import bz2
import gzip
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

import numpy as np

from weather_radar_ml.data.domain import RadarFrame, RadarSequence, SpatialGrid
from weather_radar_ml.data.quality import ObservationQuality


class BinaryReader(Protocol):
    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


RADKLIM_CRS = (
    "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=10 +k=0.93301270189 "
    "+x_0=0 +y_0=0 +a=6370040 +b=6370040 +units=m +no_defs"
)
_MISSING_BIT = np.uint16(0x2000)
_SECONDARY_BIT = np.uint16(0x1000)
_NEGATIVE_BIT = np.uint16(0x4000)
_CLUTTER_BIT = np.uint16(0x8000)
_VALUE_MASK = np.uint16(0x0FFF)


@dataclass(frozen=True)
class RadklimHeader:
    """Subset of RADKLIM metadata required for canonical decoding."""

    product: str
    timestamp: datetime
    rows: int
    cols: int
    precision: float
    interval_minutes: int


def _open_binary(path: Path) -> BinaryReader:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    if path.suffix == ".bz2":
        return bz2.open(path, "rb")
    return path.open("rb")


def _read_header(handle: BinaryReader) -> str:
    data = bytearray()
    while True:
        byte = handle.read(1)
        if not byte:
            raise ValueError("RADKLIM header is missing the ETX terminator.")
        if byte == b"\x03":
            return data.decode("ascii")
        data.extend(byte)
        if len(data) > 64 * 1024:
            raise ValueError("RADKLIM header exceeds the safety limit.")


def parse_header(header: str) -> RadklimHeader:
    """Parse and validate the RADKLIM header fields used by this adapter."""
    if len(header) < 17:
        raise ValueError("RADKLIM header is too short.")
    product = header[:2]
    if product != "YW":
        raise ValueError(f"Expected RADKLIM-YW product, got {product!r}.")

    try:
        timestamp = datetime(
            2000 + int(header[15:17]),
            int(header[13:15]),
            int(header[2:4]),
            int(header[4:6]),
            int(header[6:8]),
            tzinfo=UTC,
        )
    except ValueError as exc:
        raise ValueError("RADKLIM header contains an invalid timestamp.") from exc

    grid = re.search(r"GP\s*(\d+)\s*[xX]\s*(\d+)", header)
    precision = re.search(r"PR\s*E([+-]\d+)", header)
    interval = re.search(r"INT\s*(\d+)", header)
    if grid is None or precision is None or interval is None:
        raise ValueError("RADKLIM header lacks GP, PR, or INT metadata.")

    return RadklimHeader(
        product=product,
        timestamp=timestamp,
        rows=int(grid.group(1)),
        cols=int(grid.group(2)),
        precision=10.0 ** int(precision.group(1)),
        interval_minutes=int(interval.group(1)),
    )


def _radklim_grid(rows: int, cols: int) -> SpatialGrid:
    if (rows, cols) != (1100, 900):
        raise ValueError(f"Unsupported RADKLIM-YW grid {rows}x{cols}.")
    # DWD specifies the lower-left corner of the first pixel at these legacy
    # RADOLAN coordinates. Canonical coordinates represent pixel centres.
    x = -443_462.16692185594 + 500.0 + np.arange(cols) * 1000.0
    y = -4_758_644.724265573 + 500.0 + np.arange(rows) * 1000.0
    return SpatialGrid(x=x, y=y, crs=RADKLIM_CRS)


class RadklimYwDecoder:
    """Decode native YW composites into framework-independent radar sequences."""

    def decode_frame(self, artifact: Path) -> RadarFrame:
        """Decode one uncompressed, gzip, or bzip2 RADKLIM-YW composite."""
        with _open_binary(artifact) as handle:
            header = parse_header(_read_header(handle))
            payload = handle.read()

        expected_bytes = header.rows * header.cols * 2
        if len(payload) != expected_bytes:
            raise ValueError(
                f"RADKLIM payload has {len(payload)} bytes; expected {expected_bytes}."
            )

        encoded = np.frombuffer(payload, dtype="<u2").reshape(header.rows, header.cols)
        missing = (encoded & _MISSING_BIT) != 0
        negative = (encoded & _NEGATIVE_BIT) != 0
        precipitation_amount = (encoded & _VALUE_MASK).astype(
            np.float32
        ) * header.precision
        precipitation_amount[negative] *= -1.0

        # RADKLIM-YW stores precipitation for the interval declared by INT.
        # Normalize the native interval amount to the canonical precipitation
        # rate in mm/h.
        precipitation_rate = precipitation_amount * 60.0 / header.interval_minutes
        precipitation_rate[missing] = np.nan
        quality = np.full(encoded.shape, ObservationQuality.OBSERVED, dtype=np.uint8)
        quality[missing] = ObservationQuality.MISSING

        source_flags = np.zeros(encoded.shape, dtype=np.uint8)
        source_flags[(encoded & _SECONDARY_BIT) != 0] |= 1
        source_flags[(encoded & _CLUTTER_BIT) != 0] |= 2

        return RadarFrame(
            timestamp=header.timestamp,
            precipitation_rate=precipitation_rate,
            quality=quality,
            source_flags=source_flags,
            grid=_radklim_grid(header.rows, header.cols),
        )

    def decode(self, artifact: Path) -> RadarSequence:
        """Decode one composite as a one-frame sequence for the generic contract."""
        frame = self.decode_frame(artifact)
        return RadarSequence((frame,), step=timedelta(minutes=5))

    def decode_many(self, artifacts: tuple[Path, ...]) -> RadarSequence:
        """Decode multiple composites into a validated chronological sequence."""
        frames = sorted(
            (self.decode_frame(artifact) for artifact in artifacts),
            key=lambda frame: frame.timestamp,
        )
        return RadarSequence.from_frames(frames, step=timedelta(minutes=5))
