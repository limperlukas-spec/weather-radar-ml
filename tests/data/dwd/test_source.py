from pathlib import Path

import pytest

from weather_radar_ml.data.dwd.source import DwdRadklimYwMonthlySource


def test_source_builds_authoritative_monthly_url() -> None:
    source = DwdRadklimYwMonthlySource(2023, 9)

    assert source.filename == "YW2017.002_202309.tar"
    assert source.url.endswith("/2023/YW2017.002_202309.tar")


def test_source_reuses_existing_immutable_artifact(tmp_path: Path) -> None:
    source = DwdRadklimYwMonthlySource(2023, 9)
    artifact = tmp_path / source.filename
    artifact.write_bytes(b"already downloaded")

    assert source.acquire(tmp_path) == (artifact,)
    assert artifact.read_bytes() == b"already downloaded"


def test_source_rejects_invalid_month() -> None:
    with pytest.raises(ValueError, match="Month"):
        DwdRadklimYwMonthlySource(2023, 13)


def test_source_downloads_via_partial_file_then_atomically_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response:
        def __init__(self) -> None:
            self._chunks = iter((b"abc", b"def", b""))

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return next(self._chunks)

    monkeypatch.setattr(
        "weather_radar_ml.data.dwd.source.urlopen", lambda _url: Response()
    )
    source = DwdRadklimYwMonthlySource(2023, 9)

    artifact = source.acquire(tmp_path)[0]

    assert artifact.read_bytes() == b"abcdef"
    assert not artifact.with_suffix(".tar.part").exists()
