"""Typed batches and epoch results for forecast training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from weather_radar_ml.evaluation.domain import ForecastMetricReport


@dataclass(frozen=True, slots=True)
class ForecastBatch:
    """One batched training item using the common forecast tensor layout."""

    dynamic: Tensor
    target: Tensor
    static: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.dynamic.shape[0])

    def to(self, device: torch.device) -> ForecastBatch:
        """Move all tensors to a device without changing batch semantics."""
        return ForecastBatch(
            dynamic=self.dynamic.to(device),
            target=self.target.to(device),
            static=None if self.static is None else self.static.to(device),
        )


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Aggregated loss and metrics for one complete dataset pass."""

    loss: float
    metrics: ForecastMetricReport
    batches: int
    samples: int

    def __post_init__(self) -> None:
        if self.batches <= 0:
            raise ValueError("EpochResult batches must be greater than zero.")
        if self.samples <= 0:
            raise ValueError("EpochResult samples must be greater than zero.")


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Train and validation results in matching epoch order."""

    train: tuple[EpochResult, ...]
    validation: tuple[EpochResult, ...]

    def __post_init__(self) -> None:
        if not self.train:
            raise ValueError("TrainingHistory requires at least one epoch.")
        if len(self.train) != len(self.validation):
            raise ValueError("Train and validation histories must have equal length.")
