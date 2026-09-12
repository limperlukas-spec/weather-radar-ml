import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import xarray as xr

from weather_radar_ml.config.ml_dataset import (
    FeatureConfig,
    FeatureKind,
    FeatureSchemaConfig,
    MLDatasetConfig,
    PreparedSourceConfig,
    SourceRole,
    SpatialSamplingConfig,
    SplitIntervalConfig,
    SplitName,
    TemporalSplitConfig,
    TemporalWindowConfig,
)
from weather_radar_ml.data.ml import artifact as artifact_module
from weather_radar_ml.data.ml.artifact import (
    build_ml_dataset_artifact,
    plan_ml_dataset_artifact,
)
from weather_radar_ml.data.ml.builder import BuildReport


def _config(source_path: Path | None) -> MLDatasetConfig:
    return MLDatasetConfig(
        name="hardening-test",
        sources=(
            PreparedSourceConfig(
                source_id="radar",
                artifact_id="prepared-v1",
                role=SourceRole.PRIMARY,
                path=source_path,
            ),
        ),
        features=FeatureSchemaConfig(
            dynamic_inputs=(
                FeatureConfig(
                    name="rain",
                    source_id="radar",
                    variable="rain",
                    kind=FeatureKind.DYNAMIC,
                ),
            ),
            targets=(
                FeatureConfig(
                    name="rain_target",
                    source_id="radar",
                    variable="rain",
                    kind=FeatureKind.DYNAMIC,
                ),
            ),
        ),
        temporal=TemporalWindowConfig(
            step_minutes=5,
            history_minutes=10,
            forecast_minutes=10,
        ),
        spatial=SpatialSamplingConfig(),
        splits=TemporalSplitConfig(
            intervals=(
                SplitIntervalConfig(
                    SplitName.TRAIN,
                    datetime(2023, 1, 1, tzinfo=UTC),
                    datetime(2023, 1, 2, tzinfo=UTC),
                ),
                SplitIntervalConfig(
                    SplitName.VALIDATION,
                    datetime(2023, 1, 3, tzinfo=UTC),
                    datetime(2023, 1, 4, tzinfo=UTC),
                ),
                SplitIntervalConfig(
                    SplitName.TEST,
                    datetime(2023, 1, 5, tzinfo=UTC),
                    datetime(2023, 1, 6, tzinfo=UTC),
                ),
            )
        ),
    )


@pytest.mark.parametrize(
    "checksums",
    ({}, {"radar": "abc123", "unexpected": "extra"}),
)
def test_plan_rejects_checksum_source_set_mismatch(
    tmp_path: Path,
    checksums: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="exactly match configured sources"):
        plan_ml_dataset_artifact(
            config=_config(tmp_path / "source.zarr"),
            source_checksums=checksums,
            output_root=tmp_path / "out",
        )


def test_plan_rejects_empty_checksum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        plan_ml_dataset_artifact(
            config=_config(tmp_path / "source.zarr"),
            source_checksums={"radar": "   "},
            output_root=tmp_path / "out",
        )


def test_build_rejects_missing_source_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no configured storage path"):
        build_ml_dataset_artifact(
            config=_config(None),
            source_checksums={"radar": "abc123"},
            output_root=tmp_path / "out",
        )


def test_build_rejects_nonexistent_source_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.zarr"
    with pytest.raises(FileNotFoundError):
        build_ml_dataset_artifact(
            config=_config(missing),
            source_checksums={"radar": "abc123"},
            output_root=tmp_path / "out",
        )


def _write_existing_artifact(plan, build_report: object) -> None:
    plan.paths.root.mkdir(parents=True)
    plan.paths.catalog.touch()
    plan.paths.resolved_config.write_text("{}\n", encoding="utf-8")
    plan.paths.manifest.write_text(
        json.dumps(
            {
                "dataset_build_id": plan.dataset_build_id,
                "build_report": build_report,
            }
        ),
        encoding="utf-8",
    )


def test_existing_manifest_with_wrong_build_id_is_rejected(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "source.zarr")
    plan = plan_ml_dataset_artifact(
        config=config,
        source_checksums={"radar": "abc123"},
        output_root=tmp_path / "out",
    )
    _write_existing_artifact(plan, {})
    data = json.loads(plan.paths.manifest.read_text(encoding="utf-8"))
    data["dataset_build_id"] = "wrong-id"
    plan.paths.manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match"):
        build_ml_dataset_artifact(
            config=config,
            source_checksums={"radar": "abc123"},
            output_root=tmp_path / "out",
        )


def test_existing_manifest_without_valid_build_report_is_rejected(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "source.zarr")
    plan = plan_ml_dataset_artifact(
        config=config,
        source_checksums={"radar": "abc123"},
        output_root=tmp_path / "out",
    )
    _write_existing_artifact(plan, "not-a-mapping")

    with pytest.raises(RuntimeError, match="no valid build_report"):
        build_ml_dataset_artifact(
            config=config,
            source_checksums={"radar": "abc123"},
            output_root=tmp_path / "out",
        )


def test_failed_build_removes_temporary_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.zarr"
    source.mkdir()
    output = tmp_path / "out"

    monkeypatch.setattr(
        artifact_module.xr,
        "open_zarr",
        lambda *args, **kwargs: xr.Dataset(),
    )

    class FailingBuilder:
        def __init__(self, **kwargs: object) -> None:
            pass

        def build(self) -> BuildReport:
            raise RuntimeError("synthetic build failure")

    monkeypatch.setattr(
        artifact_module,
        "SampleIndexBuilder",
        FailingBuilder,
    )

    with pytest.raises(RuntimeError, match="synthetic build failure"):
        build_ml_dataset_artifact(
            config=_config(source),
            source_checksums={"radar": "abc123"},
            output_root=output,
        )

    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_checksum_change_changes_artifact_identity(tmp_path: Path) -> None:
    config = _config(tmp_path / "source.zarr")

    first = plan_ml_dataset_artifact(
        config=config,
        source_checksums={"radar": "checksum-a"},
        output_root=tmp_path / "out",
    )
    second = plan_ml_dataset_artifact(
        config=config,
        source_checksums={"radar": "checksum-b"},
        output_root=tmp_path / "out",
    )

    assert first.dataset_build_id != second.dataset_build_id
    assert first.paths.root != second.paths.root
