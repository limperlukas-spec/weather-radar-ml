import json
from pathlib import Path

import pytest

from weather_radar_ml.config.experiment import ComponentConfig
from weather_radar_ml.config.schema import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    RunConfig,
    TrackingConfig,
    TrainingConfig,
)
from weather_radar_ml.training.pipeline import run_experiment


def _learned_config(
    root: Path,
    *,
    epochs: int = 8,
    patience: int | None = None,
) -> RunConfig:
    return RunConfig(
        name="learned-pipeline-smoke",
        data=DataConfig(
            name="synthetic",
            train_samples=2,
            validation_samples=2,
            history_steps=2,
            forecast_steps=2,
            dynamic_input_features=("rain",),
            target_features=("rain",),
            height=8,
            width=8,
        ),
        model=ModelConfig(name="unet", parameters={"base_channels": 2}),
        training=TrainingConfig(
            seed=17,
            epochs=epochs,
            batch_size=2,
            early_stopping_patience=patience,
            optimizer=ComponentConfig("adam", {"lr": 0.01}),
        ),
        tracking=TrackingConfig(
            uri="./mlruns",
            experiment_name="tests",
            enabled=False,
        ),
        output=OutputConfig(runs_root=root),
    )


def test_learned_pipeline_trains_unet_and_persists_best_and_last(
    tmp_path: Path,
) -> None:
    result = run_experiment(_learned_config(tmp_path / "runs", epochs=4))

    assert len(result.history.train) == 4
    assert 1 <= result.history.best_epoch <= 4
    assert result.history.best_validation_loss >= 0.0
    assert result.history.stopped_early is False
    assert {path.name for path in result.artifact.checkpoints} == {
        "best.pt",
        "last.pt",
    }

    persisted = json.loads(result.artifact.history.read_text(encoding="utf-8"))
    assert persisted["best_epoch"] == result.history.best_epoch
    assert persisted["best_validation_loss"] == pytest.approx(
        result.history.best_validation_loss
    )
    assert persisted["stopped_early"] is False


def test_learned_pipeline_reduces_training_loss_on_tiny_dataset(
    tmp_path: Path,
) -> None:
    result = run_experiment(_learned_config(tmp_path / "runs", epochs=12))

    assert result.history.train[-1].loss < result.history.train[0].loss


def test_unet_pipeline_rejects_static_inputs_before_training(tmp_path: Path) -> None:
    base = _learned_config(tmp_path / "runs", epochs=1)
    config = RunConfig(
        name=base.name,
        data=DataConfig(
            name="synthetic",
            train_samples=2,
            validation_samples=2,
            history_steps=2,
            forecast_steps=2,
            dynamic_input_features=("rain",),
            static_input_features=("height",),
            target_features=("rain",),
            height=8,
            width=8,
        ),
        model=base.model,
        training=base.training,
        tracking=base.tracking,
        output=base.output,
    )

    with pytest.raises(ValueError, match="does not support static inputs"):
        run_experiment(config)
