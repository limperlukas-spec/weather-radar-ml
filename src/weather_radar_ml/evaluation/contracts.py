"""Contracts for stateful forecast metric accumulation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from weather_radar_ml.evaluation.domain import ForecastMetricReport
from weather_radar_ml.models.domain import ModelSpec


@runtime_checkable
class ForecastMetricAccumulator(Protocol):
    """Accumulate exact dataset-level metrics over arbitrary batch sizes."""

    @property
    def spec(self) -> ModelSpec: ...  # pragma: no cover

    def reset(self) -> None: ...  # pragma: no cover

    def update(
        self, prediction: Tensor, target: Tensor
    ) -> None: ...  # pragma: no cover

    def compute(self) -> ForecastMetricReport: ...  # pragma: no cover
