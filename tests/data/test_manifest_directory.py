from pathlib import Path

import pytest

from weather_radar_ml.data.manifest import sha256_directory


def test_sha256_directory_is_order_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "b.txt").write_text("two", encoding="utf-8")
    (left / "a.txt").write_text("one", encoding="utf-8")
    (right / "a.txt").write_text("one", encoding="utf-8")
    (right / "b.txt").write_text("two", encoding="utf-8")

    assert sha256_directory(left) == sha256_directory(right)

    (right / "b.txt").write_text("changed", encoding="utf-8")
    assert sha256_directory(left) != sha256_directory(right)


def test_sha256_directory_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contains no files"):
        sha256_directory(tmp_path)
