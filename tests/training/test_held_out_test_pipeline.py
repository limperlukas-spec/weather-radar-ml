from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from weather_radar_ml.config.experiment import ComponentConfig
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
from weather_radar_ml.config.schema import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    RunConfig,
    TrackingConfig,
    TrainingConfig,
)
from weather_radar_ml.data.ml.artifact import build_ml_dataset_artifact
from weather_radar_ml.training.pipeline import (
    evaluate_reference_checkpoint,
    run_experiment,
    write_reference_forecast_artifact,
)


def test_best_checkpoint_can_be_evaluated_and_materialized_on_test_split(
    tmp_path: Path,
) -> None:
    times = np.arange(
        np.datetime64("2023-09-01T00:00"),
        np.datetime64("2023-09-01T01:00"),
        np.timedelta64(5, "m"),
    )
    values = np.arange(times.size, dtype=np.float32)[:, None, None]
    values = np.broadcast_to(values, (times.size, 8, 8)).copy()
    source_path = tmp_path / "prepared.zarr"
    xr.Dataset(
        {"precipitation_rate": (("time", "y", "x"), values)},
        coords={"time": times, "y": np.arange(8), "x": np.arange(8)},
    ).to_zarr(source_path, mode="w", zarr_format=2)

    dataset = MLDatasetConfig(
        name="held-out-test",
        sources=(
            PreparedSourceConfig(
                source_id="radklim",
                artifact_id="prepared-test",
                role=SourceRole.PRIMARY,
                path=source_path,
            ),
        ),
        features=FeatureSchemaConfig(
            dynamic_inputs=(
                FeatureConfig(
                    "precipitation",
                    "radklim",
                    "precipitation_rate",
                    FeatureKind.DYNAMIC,
                ),
            ),
            targets=(
                FeatureConfig(
                    "precipitation_target",
                    "radklim",
                    "precipitation_rate",
                    FeatureKind.DYNAMIC,
                ),
            ),
        ),
        temporal=TemporalWindowConfig(
            step_minutes=5,
            history_minutes=10,
            forecast_minutes=5,
        ),
        spatial=SpatialSamplingConfig(),
        splits=TemporalSplitConfig(
            (
                SplitIntervalConfig(
                    SplitName.TRAIN,
                    datetime(2023, 9, 1, 0, 0, tzinfo=UTC),
                    datetime(2023, 9, 1, 0, 20, tzinfo=UTC),
                ),
                SplitIntervalConfig(
                    SplitName.VALIDATION,
                    datetime(2023, 9, 1, 0, 20, tzinfo=UTC),
                    datetime(2023, 9, 1, 0, 40, tzinfo=UTC),
                ),
                SplitIntervalConfig(
                    SplitName.TEST,
                    datetime(2023, 9, 1, 0, 40, tzinfo=UTC),
                    datetime(2023, 9, 1, 1, 0, tzinfo=UTC),
                ),
            )
        ),
    )
    built = build_ml_dataset_artifact(
        config=dataset,
        source_checksums={"radklim": "prepared-checksum"},
        output_root=tmp_path / "ml",
    )
    config = RunConfig(
        name="held-out-test",
        data=DataConfig(
            name="ml_dataset",
            catalog_path=built.plan.paths.catalog,
            dataset=dataset,
        ),
        model=ModelConfig(name="unet", parameters={"base_channels": 2}),
        training=TrainingConfig(
            seed=17,
            epochs=1,
            batch_size=2,
            optimizer=ComponentConfig("adam", {"lr": 0.01}),
        ),
        tracking=TrackingConfig(
            uri="./mlruns",
            experiment_name="tests",
            enabled=False,
        ),
        output=OutputConfig(runs_root=tmp_path / "runs"),
    )

    run = run_experiment(config)
    test_report = evaluate_reference_checkpoint(config, run, split="test")
    artifact = write_reference_forecast_artifact(
        config,
        run,
        split="test",
        research=test_report,
        qualitative_count=1,
    )

    assert test_report.all.learned.lead_minutes == (5,)
    assert test_report.all.learned.overall.count == 2 * 8 * 8
    assert artifact.split == "test"
    predictions = xr.open_zarr(artifact.predictions, consolidated=False)
    try:
        assert predictions.sizes["sample"] == 2
        assert predictions["lead"].values.tolist() == [5]
    finally:
        predictions.close()
