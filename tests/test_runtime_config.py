from pathlib import Path

import pytest

from weather_radar_ml.config.experiment import ComponentConfig
from weather_radar_ml.config.schema import (
    DataConfig,
    RunConfig,
    run_config_from_mapping,
)


def test_legacy_minimal_mapping_keeps_runnable_defaults() -> None:
    config = run_config_from_mapping(
        {
            "data": {"name": "synthetic"},
            "model": {"name": "persistence"},
            "training": {"seed": 42},
            "tracking": {
                "uri": "./mlruns",
                "experiment_name": "weather-radar-ml",
            },
        }
    )

    assert isinstance(config, RunConfig)
    assert config.data.name == "synthetic"
    assert config.training.epochs == 1
    assert config.training.batch_size == 4
    assert config.training.early_stopping_patience is None
    assert config.training.loss == ComponentConfig("mse")
    assert config.training.optimizer == ComponentConfig("none")
    assert config.output.runs_root == Path("runs")
    assert config.tracking.enabled is False


def test_full_synthetic_mapping_is_typed() -> None:
    config = run_config_from_mapping(
        {
            "name": "configured-run",
            "data": {
                "name": "synthetic",
                "train_samples": 12,
                "validation_samples": 6,
                "history_steps": 6,
                "forecast_steps": 3,
                "dynamic_input_features": ["rain", "temperature"],
                "static_input_features": ["height"],
                "target_features": ["rain"],
                "height": 16,
                "width": 20,
            },
            "model": {"name": "scaled_persistence", "parameters": {}},
            "training": {
                "seed": 7,
                "epochs": 2,
                "batch_size": 3,
                "device": "cpu",
                "num_workers": 0,
                "deterministic_algorithms": True,
                "early_stopping_patience": 7,
                "loss": {"name": "mse", "parameters": {}},
                "optimizer": {"name": "adam", "parameters": {"lr": 0.01}},
            },
            "tracking": {
                "enabled": True,
                "uri": "http://127.0.0.1:5000",
                "experiment_name": "tests",
                "artifact_path": "run-artifact",
            },
            "output": {"runs_root": "custom-runs"},
        }
    )

    assert config.name == "configured-run"
    assert config.data.dynamic_input_features == ("rain", "temperature")
    assert config.data.static_input_features == ("height",)
    assert config.training.optimizer.parameters["lr"] == 0.01
    assert config.training.early_stopping_patience == 7
    assert config.tracking.enabled is True
    assert config.output.runs_root == Path("custom-runs")


def test_ml_dataset_data_config_requires_catalog_and_dataset() -> None:
    with pytest.raises(ValueError, match="requires catalog_path and dataset"):
        DataConfig(name="ml_dataset")


def test_training_config_rejects_non_positive_early_stopping_patience() -> None:
    with pytest.raises(ValueError, match="early_stopping_patience"):
        run_config_from_mapping(
            {
                "data": {"name": "synthetic"},
                "model": {"name": "persistence"},
                "training": {"seed": 42, "early_stopping_patience": 0},
                "tracking": {
                    "uri": "./mlruns",
                    "experiment_name": "tests",
                },
            }
        )
