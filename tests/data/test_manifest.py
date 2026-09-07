from pathlib import Path

import pytest

from weather_radar_ml.data.manifest import DatasetManifest, sha256_file


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="radklim-yw-sample-v1",
        source_product="RADKLIM-YW",
        source_version="example",
        pipeline_version="0.2.0",
        schema_version="1",
        source_checksums={"sample.bin": "abc123"},
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _manifest().write(path)

    assert DatasetManifest.read(path) == _manifest()


def test_manifest_requires_source_checksum() -> None:
    with pytest.raises(ValueError, match="source checksum"):
        DatasetManifest(
            dataset_id="dataset",
            source_product="product",
            source_version="version",
            pipeline_version="pipeline",
            schema_version="1",
            source_checksums={},
        )


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(b"weather-radar-ml")

    assert sha256_file(path) == (
        "4f324fa23e433a9c21e7a12633d497562474888b203730e85b64d9897f61c254"
    )
