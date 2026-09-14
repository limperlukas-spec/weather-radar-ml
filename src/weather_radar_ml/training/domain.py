"""Typed batches and epoch results for forecast training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite

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
    """Train/validation history plus deterministic model-selection metadata."""

    train: tuple[EpochResult, ...]
    validation: tuple[EpochResult, ...]
    best_epoch: int = 0
    best_validation_loss: float = float("inf")
    stopped_early: bool = False

    def __post_init__(self) -> None:
        if not self.train:
            raise ValueError("TrainingHistory requires at least one epoch.")
        if len(self.train) != len(self.validation):
            raise ValueError("Train and validation histories must have equal length.")

        if self.best_epoch == 0:
            best_index = min(
                range(len(self.validation)),
                key=lambda index: self.validation[index].loss,
            )
            object.__setattr__(self, "best_epoch", best_index + 1)
            object.__setattr__(
                self,
                "best_validation_loss",
                self.validation[best_index].loss,
            )

        if not 1 <= self.best_epoch <= len(self.validation):
            raise ValueError("best_epoch must refer to a completed epoch.")
        if not isfinite(self.best_validation_loss):
            raise ValueError("best_validation_loss must be finite.")
        selected_loss = self.validation[self.best_epoch - 1].loss
        if not isclose(selected_loss, self.best_validation_loss):
            raise ValueError(
                "best_validation_loss must match the selected validation epoch."
            )
