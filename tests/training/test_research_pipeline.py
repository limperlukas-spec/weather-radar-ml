from pathlib import Path

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


def test_learned_pipeline_evaluates_best_checkpoint_against_persistence(
    tmp_path: Path,
) -> None:
    config = RunConfig(
        name="research-pipeline-smoke",
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
            epochs=2,
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

    result = run_experiment(config)

    assert result.validation_research is not None
    assert result.validation_research.all.learned.lead_minutes == (5, 10)
    assert result.validation_research.all.learned.overall.count == 256
    assert result.validation_research.all.persistence.overall.count == 256
