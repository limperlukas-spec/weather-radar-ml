from __future__ import annotations

import io
import tarfile
from datetime import date
from pathlib import Path

import pytest

from weather_radar_ml.data.dwd.archive import extract_day


def _add_bytes_to_tar(
    archive: tarfile.TarFile,
    name: str,
    content: bytes,
) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def _build_daily_archive(
    path: Path,
    frame_names: tuple[str, ...],
) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for frame_name in frame_names:
            _add_bytes_to_tar(
                archive,
                frame_name,
                b"YW-test-frame",
            )


def _build_monthly_archive(
    path: Path,
    daily_archive: Path,
) -> None:
    with tarfile.open(path, mode="w") as archive:
        archive.add(
            daily_archive,
            arcname=daily_archive.name,
        )


def test_extract_day_extracts_frames_from_nested_daily_archive(
    tmp_path: Path,
) -> None:
    frame_names = (
        "raa01-yw2017.002_10000-2309010000-dwd---bin",
        "raa01-yw2017.002_10000-2309010005-dwd---bin",
    )

    daily_archive = tmp_path / "YW2017.002_20230901.tar.gz"
    monthly_archive = tmp_path / "YW2017.002_202309.tar"
    destination = tmp_path / "extracted"

    _build_daily_archive(daily_archive, frame_names)
    _build_monthly_archive(monthly_archive, daily_archive)

    extracted = extract_day(
        monthly_archive,
        date(2023, 9, 1),
        destination,
    )

    assert tuple(path.name for path in extracted) == frame_names

    for path in extracted:
        assert path.is_file()
        assert path.read_bytes() == b"YW-test-frame"


def test_extract_day_does_not_return_nested_archive(
    tmp_path: Path,
) -> None:
    frame_name = "raa01-yw2017.002_10000-2309010000-dwd---bin"

    daily_archive = tmp_path / "YW2017.002_20230901.tar.gz"
    monthly_archive = tmp_path / "YW2017.002_202309.tar"
    destination = tmp_path / "extracted"

    _build_daily_archive(daily_archive, (frame_name,))
    _build_monthly_archive(monthly_archive, daily_archive)

    extracted = extract_day(
        monthly_archive,
        date(2023, 9, 1),
        destination,
    )

    assert len(extracted) == 1
    assert extracted[0].name == frame_name
    assert all(path.suffix != ".gz" for path in extracted)


def test_extract_day_rejects_missing_day(
    tmp_path: Path,
) -> None:
    daily_archive = tmp_path / "YW2017.002_20230902.tar.gz"
    monthly_archive = tmp_path / "YW2017.002_202309.tar"
    destination = tmp_path / "extracted"

    _build_daily_archive(
        daily_archive,
        ("raa01-yw2017.002_10000-2309020000-dwd---bin",),
    )
    _build_monthly_archive(monthly_archive, daily_archive)

    with pytest.raises(
        ValueError,
        match="contains no RADKLIM-YW archive",
    ):
        extract_day(
            monthly_archive,
            date(2023, 9, 1),
            destination,
        )
