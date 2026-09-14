import json
from pathlib import Path

import pytest

from weather_radar_ml.config.schema import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    RunConfig,
    TrackingConfig,
    TrainingConfig,
)
from weather_radar_ml.training.pipeline import run_experiment


def _config(root: Path, *, epochs: int = 2) -> RunConfig:
    return RunConfig(
        name="pipeline-smoke",
        data=DataConfig(
            name="synthetic",
            train_samples=6,
            validation_samples=4,
            history_steps=2,
            forecast_steps=2,
            dynamic_input_features=("rain",),
            target_features=("rain",),
            height=3,
            width=4,
        ),
        model=ModelConfig(name="persistence"),
        training=TrainingConfig(seed=17, epochs=epochs, batch_size=2),
        tracking=TrackingConfig(
            uri="./mlruns",
            experiment_name="tests",
            enabled=False,
        ),
        output=OutputConfig(runs_root=root),
    )


def test_synthetic_pipeline_persists_complete_run_artifact(tmp_path: Path) -> None:
    result = run_experiment(_config(tmp_path / "runs"))

    assert result.tracking is None
    assert result.artifact.manifest.is_file()
    assert result.artifact.config.is_file()
    assert result.artifact.history.is_file()
    assert result.artifact.checkpoints == ()

    history = json.loads(result.artifact.history.read_text(encoding="utf-8"))
    persisted_config = json.loads(result.artifact.config.read_text(encoding="utf-8"))
    assert len(history["epochs"]) == 2
    assert persisted_config["training"]["device"] == "cpu"
    assert persisted_config["training"]["num_workers"] == 0
    assert history["epochs"][0]["train"]["samples"] == 6
    assert history["epochs"][0]["validation"]["samples"] == 4


def test_synthetic_pipeline_is_reproducible_for_same_config(tmp_path: Path) -> None:
    config = _config(tmp_path / "runs")

    first = run_experiment(config)
    second = run_experiment(config)

    first_config = first.artifact.config.read_text(encoding="utf-8")
    second_config = second.artifact.config.read_text(encoding="utf-8")
    first_history = first.artifact.history.read_text(encoding="utf-8")
    second_history = second.artifact.history.read_text(encoding="utf-8")

    assert first_config == second_config
    assert first_history == second_history


def test_pipeline_rejects_unsupported_model_before_publication(tmp_path: Path) -> None:
    config = _config(tmp_path / "runs")
    config = RunConfig(
        name=config.name,
        data=config.data,
        model=ModelConfig(name="unknown"),
        training=config.training,
        tracking=config.tracking,
        output=config.output,
    )

    with pytest.raises(ValueError, match="Unsupported model strategy"):
        run_experiment(config)

    assert not (tmp_path / "runs").exists()


def test_pipeline_rejects_optimizer_for_parameter_free_persistence(
    tmp_path: Path,
) -> None:
    from weather_radar_ml.config.experiment import ComponentConfig

    base = _config(tmp_path / "runs")
    training = TrainingConfig(
        seed=base.training.seed,
        epochs=1,
        batch_size=2,
        optimizer=ComponentConfig("adam", {"lr": 0.01}),
    )
    config = RunConfig(
        name=base.name,
        data=base.data,
        model=base.model,
        training=training,
        tracking=base.tracking,
        output=base.output,
    )

    with pytest.raises(ValueError, match="trainable parameters"):
        run_experiment(config)
