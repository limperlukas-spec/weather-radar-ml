"""Contracts shared by forecast training implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from weather_radar_ml.models.domain import ModelSpec


@runtime_checkable
class ForecastLoss(Protocol):
    """Loss interface used by the common forecast trainer."""

    @property
    def name(self) -> str: ...  # pragma: no cover

    @property
    def spec(self) -> ModelSpec: ...  # pragma: no cover

    def __call__(
        self, prediction: Tensor, target: Tensor
    ) -> Tensor: ...  # pragma: no cover
