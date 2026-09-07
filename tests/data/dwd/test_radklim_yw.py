from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from weather_radar_ml.data.dwd.radklim_yw import RadklimYwDecoder, parse_header
from weather_radar_ml.data.quality import ObservationQuality


def _write_fixture(path: Path) -> None:
    header = "YW010005100000123PR E-02INT 5GP 1100x900"
    encoded = np.zeros((1100, 900), dtype="<u2")
    encoded[0, 0] = 343  # 3.43 mm / 5 min -> 41.16 mm/h
    encoded[0, 1] = 0x2000 | 2500  # DWD missing flag
    encoded[0, 2] = 0x1000 | 50  # secondary source flag
    encoded[0, 3] = 0x8000 | 25  # clutter source flag
    path.write_bytes(header.encode("ascii") + b"\x03" + encoded.tobytes())


def test_parse_header_reads_product_time_grid_and_scaling() -> None:
    parsed = parse_header("YW010005100000123PR E-02INT 5GP 1100x900")

    assert parsed.timestamp == datetime(2023, 1, 1, 0, 5, tzinfo=UTC)
    assert parsed.rows == 1100
    assert parsed.cols == 900
    assert parsed.precision == pytest.approx(0.01)
    assert parsed.interval_minutes == 5


def test_decoder_normalizes_yw_amount_to_rate_and_preserves_flags(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "yw.bin"
    _write_fixture(artifact)

    frame = RadklimYwDecoder().decode_frame(artifact)

    assert frame.precipitation_rate[0, 0] == pytest.approx(41.16)
    assert np.isnan(frame.precipitation_rate[0, 1])
    assert frame.quality[0, 1] == ObservationQuality.MISSING
    assert frame.source_flags is not None
    assert frame.source_flags[0, 2] == 1
    assert frame.source_flags[0, 3] == 2
    assert frame.grid.x.size == 900
    assert frame.grid.y.size == 1100
    assert frame.grid.x[1] - frame.grid.x[0] == pytest.approx(1000.0)


def test_decoder_rejects_truncated_payload(tmp_path: Path) -> None:
    artifact = tmp_path / "bad.bin"
    artifact.write_bytes(b"YW010005100000123PR E-02INT 5GP 1100x900\x03" + b"\x00\x00")

    with pytest.raises(ValueError, match="expected"):
        RadklimYwDecoder().decode_frame(artifact)


def test_decoder_rejects_non_yw_product() -> None:
    with pytest.raises(ValueError, match="Expected RADKLIM-YW"):
        parse_header("RW010005100000123PR E-02INT 5GP 1100x900")


def test_parse_header_rejects_missing_required_metadata() -> None:
    with pytest.raises(ValueError, match="lacks GP, PR, or INT"):
        parse_header("YW010005100000123")


def test_parse_header_rejects_too_short_header() -> None:
    with pytest.raises(ValueError, match="too short"):
        parse_header("YW123")


def test_parse_header_rejects_invalid_timestamp() -> None:
    with pytest.raises(ValueError, match="invalid timestamp"):
        parse_header("YW320005100000123PR E-02INT 5GP 1100x900")


def test_decoder_rejects_unsupported_grid(tmp_path: Path) -> None:
    artifact = tmp_path / "unsupported-grid.bin"
    header = "YW010005100000123PR E-02INT 5GP 100x100"
    payload = np.zeros((100, 100), dtype="<u2")

    artifact.write_bytes(header.encode("ascii") + b"\x03" + payload.tobytes())

    with pytest.raises(ValueError, match="Unsupported RADKLIM-YW grid"):
        RadklimYwDecoder().decode_frame(artifact)
