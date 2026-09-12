"""Model contracts shared by training and inference code."""

from weather_radar_ml.models.contracts import ForecastModel
from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.models.validation import (
    validate_forecast_batch,
    validate_prediction,
)

__all__ = [
    "ForecastModel",
    "ModelSpec",
    "validate_forecast_batch",
    "validate_prediction",
]
