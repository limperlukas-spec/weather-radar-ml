"""Reference losses for radar forecasting."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.models.validation import validate_prediction


class MeanSquaredForecastLoss(nn.Module):
    """Mean squared error with the common forecast tensor contract enforced."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self._spec = spec

    @property
    def name(self) -> str:
        """Stable identifier used in experiment metadata."""
        return "mse"

    @property
    def spec(self) -> ModelSpec:
        """Return the tensor contract validated by this loss."""
        return self._spec

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Return scalar MSE after validating shape and numerical validity."""
        validate_prediction(self.spec, prediction, target)
        _require_finite(prediction, "prediction")
        _require_finite(target, "target")
        return torch.mean(torch.square(prediction - target))


def _require_finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values.")
