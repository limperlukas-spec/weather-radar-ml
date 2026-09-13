from datetime import UTC, datetime, timedelta, timezone

from weather_radar_ml.config.experiment import (
    ComponentConfig,
    DatasetReference,
    ExperimentConfig,
    TrainingSettings,
)
from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.training.run import create_run_metadata


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="run metadata",
        dataset=DatasetReference("dataset-build", "features-v1"),
        model_spec=ModelSpec(
            name="persistence",
            dynamic_input_features=("rain",),
            target_features=("rain",),
            history_steps=4,
            forecast_steps=2,
        ),
        model=ComponentConfig("persistence"),
        loss=ComponentConfig("mse"),
        optimizer=ComponentConfig("adam", {"lr": 0.001}),
        training=TrainingSettings(epochs=2, batch_size=4, seed=42),
    )


def test_create_run_metadata_binds_run_to_config_and_environment() -> None:
    local_time = datetime(
        2026,
        9,
        13,
        14,
        30,
        tzinfo=timezone(timedelta(hours=2)),
    )
    config = _config()

    metadata = create_run_metadata(
        config,
        run_id=" run-001 ",
        created_at=local_time,
        git_commit=" abcdef123 ",
    )

    assert metadata.run_id == "run-001"
    assert metadata.config_fingerprint == config.fingerprint
    assert metadata.created_at == datetime(2026, 9, 13, 12, 30, tzinfo=UTC)
    assert metadata.git_commit == "abcdef123"
    assert metadata.python_version
    assert metadata.numpy_version
    assert metadata.torch_version
    assert metadata.platform


def test_create_run_metadata_generates_run_id_and_utc_timestamp() -> None:
    metadata = create_run_metadata(_config())

    assert metadata.run_id
    assert metadata.created_at.tzinfo is UTC
