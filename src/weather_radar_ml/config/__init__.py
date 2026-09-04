"""Typed application configuration independent of Hydra."""

from weather_radar_ml.config.schema import RunConfig, run_config_from_mapping

__all__ = ["RunConfig", "run_config_from_mapping"]
