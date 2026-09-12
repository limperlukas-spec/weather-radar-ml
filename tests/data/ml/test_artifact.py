from datetime import UTC, datetime
from pathlib import Path

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


def _config(source_path: Path) -> MLDatasetConfig:
    return MLDatasetConfig(
        name="artifact-test",
        sources=(
            PreparedSourceConfig(
                "radar", "prepared-v1", SourceRole.PRIMARY, source_path
            ),
        ),
        features=FeatureSchemaConfig(
            dynamic_inputs=(
                FeatureConfig("rain", "radar", "rain", FeatureKind.DYNAMIC),
            ),
            targets=(
                FeatureConfig("rain_target", "radar", "rain", FeatureKind.DYNAMIC),
            ),
        ),
        temporal=TemporalWindowConfig(
            step_minutes=5, history_minutes=10, forecast_minutes=10
        ),
        spatial=SpatialSamplingConfig(),
        splits=TemporalSplitConfig(
            (
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


def _report(build_id: str) -> BuildReport:
    return BuildReport(
        dataset_build_id=build_id,
        feature_schema_id="features",
        valid_temporal_windows=1,
        rejected_missing_frames=0,
        rejected_outside_splits=0,
        spatial_windows=1,
        samples=1,
        samples_by_split={"train": 1},
        input_events={"light": 1},
        target_events={"light": 1},
    )


def test_plan_is_deterministic_and_does_not_write(tmp_path: Path) -> None:
    config = _config(tmp_path / "source.zarr")
    checksums = {"radar": "abc123"}
    first = plan_ml_dataset_artifact(
        config=config, source_checksums=checksums, output_root=tmp_path / "out"
    )
    second = plan_ml_dataset_artifact(
        config=config, source_checksums=checksums, output_root=tmp_path / "out"
    )
    assert first == second
    assert not first.paths.root.exists()


def test_build_writes_and_reuses_artifact(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.zarr"
    source.mkdir()
    config = _config(source)
    checksums = {"radar": "abc123"}
    plan = plan_ml_dataset_artifact(
        config=config, source_checksums=checksums, output_root=tmp_path / "out"
    )

    monkeypatch.setattr(
        artifact_module.xr, "open_zarr", lambda *args, **kwargs: xr.Dataset()
    )

    class FakeBuilder:
        def __init__(self, **kwargs) -> None:
            self.catalog = kwargs["catalog"]

        def build(self) -> BuildReport:
            self.catalog.path.touch()
            return _report(plan.dataset_build_id)

    monkeypatch.setattr(artifact_module, "SampleIndexBuilder", FakeBuilder)

    first = build_ml_dataset_artifact(
        config=config, source_checksums=checksums, output_root=tmp_path / "out"
    )
    second = build_ml_dataset_artifact(
        config=config, source_checksums=checksums, output_root=tmp_path / "out"
    )

    assert first.plan.paths.catalog.is_file()
    assert first.plan.paths.manifest.is_file()
    assert first.plan.paths.resolved_config.is_file()
    assert first.report.samples == 1
    assert second.reused_existing is True
    assert second.report == first.report
