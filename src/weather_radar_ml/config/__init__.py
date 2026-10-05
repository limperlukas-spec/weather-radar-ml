"""Typed application configuration independent of Hydra."""

from weather_radar_ml.config.classical import (
    ClassicalNowcastConfig,
    ClassicalSelectionConfig,
    MotionCandidate,
    SProgConfig,
    classical_nowcast_config_from_mapping,
    classical_selection_config_from_mapping,
)
from weather_radar_ml.config.schema import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    RunConfig,
    TrackingConfig,
    TrainingConfig,
    run_config_from_mapping,
)

__all__ = [
    "ClassicalNowcastConfig",
    "ClassicalSelectionConfig",
    "DataConfig",
    "ModelConfig",
    "MotionCandidate",
    "OutputConfig",
    "RunConfig",
    "SProgConfig",
    "TrackingConfig",
    "TrainingConfig",
    "classical_nowcast_config_from_mapping",
    "classical_selection_config_from_mapping",
    "run_config_from_mapping",
]
