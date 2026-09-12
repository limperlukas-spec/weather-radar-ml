"""Training contracts and reusable forecast losses."""

from weather_radar_ml.training.contracts import ForecastLoss
from weather_radar_ml.training.losses import MeanSquaredForecastLoss

__all__ = ["ForecastLoss", "MeanSquaredForecastLoss"]
