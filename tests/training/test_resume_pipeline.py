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
from weather_radar_ml.training.domain import TrainingHistory
from weather_radar_ml.training.pipeline import run_experiment


def _config(root: Path) -> RunConfig:
    return RunConfig(
        name="resume-pipeline-smoke",
        data=DataConfig(
            name="synthetic",
            train_samples=4,
            validation_samples=2,
            history_steps=2,
            forecast_steps=1,
            dynamic_input_features=("rain",),
            target_features=("rain",),
            height=8,
            width=8,
        ),
        model=ModelConfig(name="unet", parameters={"base_channels": 2}),
        training=TrainingConfig(
            seed=17,
            epochs=3,
            batch_size=2,
            optimizer=ComponentConfig("adam", {"lr": 0.01}),
        ),
        tracking=TrackingConfig(
            uri="./mlruns",
            experiment_name="tests",
            enabled=False,
        ),
        output=OutputConfig(runs_root=root / "runs"),
    )


def test_pipeline_resume_continues_after_persisted_epoch(tmp_path: Path) -> None:
    baseline = run_experiment(_config(tmp_path / "baseline"))
    config = _config(tmp_path / "resumed")
    checkpoint_root = tmp_path / "resume"

    def interrupt_after_first(history: TrainingHistory) -> None:
        del history
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_experiment(
            config,
            checkpoint_root=checkpoint_root,
            on_epoch_end=interrupt_after_first,
        )

    assert (checkpoint_root / "best.pt").is_file()
    assert (checkpoint_root / "resume.pt").is_file()
    assert (checkpoint_root / "resume.json").is_file()

    completed: list[int] = []
    result = run_experiment(
        config,
        checkpoint_root=checkpoint_root,
        resume=True,
        on_epoch_end=lambda history: completed.append(len(history.train)),
    )

    assert completed == [2, 3]
    assert len(result.history.train) == 3
    assert result.history == baseline.history
    assert result.artifact.root.is_dir()
