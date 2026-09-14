"""Minimal reusable PyTorch training orchestration for radar forecasts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from typing import cast

import torch
from torch import nn
from torch.optim import Optimizer

from weather_radar_ml.evaluation.contracts import ForecastMetricAccumulator
from weather_radar_ml.models.contracts import ForecastModel
from weather_radar_ml.models.validation import validate_forecast_batch
from weather_radar_ml.training.contracts import ForecastLoss
from weather_radar_ml.training.domain import EpochResult, ForecastBatch, TrainingHistory

MetricFactory = Callable[[], ForecastMetricAccumulator]


class ForecastTrainer:
    """Run deterministic train/validation epochs over pre-batched tensors."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss: ForecastLoss,
        optimizer: Optimizer | None,
        metric_factory: MetricFactory,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(model, ForecastModel):
            raise TypeError("model must satisfy the ForecastModel protocol.")
        forecast_model = cast(ForecastModel, model)
        if forecast_model.spec != loss.spec:
            raise ValueError("Model and loss must use the same ModelSpec.")
        probe = metric_factory()
        if probe.spec != forecast_model.spec:
            raise ValueError("Model and metrics must use the same ModelSpec.")

        self.model = model
        self.forecast_model = forecast_model
        self.loss = loss
        self.optimizer = optimizer
        self.metric_factory = metric_factory
        self.device = torch.device(device)
        self.model.to(self.device)

    def fit(
        self,
        train_batches: Iterable[ForecastBatch],
        validation_batches: Iterable[ForecastBatch],
        *,
        epochs: int,
    ) -> TrainingHistory:
        """Run matching train and validation passes for a fixed epoch count."""
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        train_results: list[EpochResult] = []
        validation_results: list[EpochResult] = []
        for _ in range(epochs):
            train_results.append(self.run_train_epoch(train_batches))
            validation_results.append(self.run_validation_epoch(validation_batches))
        return TrainingHistory(tuple(train_results), tuple(validation_results))

    def run_train_epoch(self, batches: Iterable[ForecastBatch]) -> EpochResult:
        """Run one optimization pass and aggregate dataset-level metrics."""
        return self._run_epoch(batches, training=True)

    def run_validation_epoch(self, batches: Iterable[ForecastBatch]) -> EpochResult:
        """Run one gradient-free validation pass."""
        return self._run_epoch(batches, training=False)

    def _run_epoch(
        self,
        batches: Iterable[ForecastBatch],
        *,
        training: bool,
    ) -> EpochResult:
        optimizing = training and self.optimizer is not None
        self.model.train(optimizing)
        metrics = self.metric_factory()
        if metrics.spec != self.forecast_model.spec:
            raise ValueError("Metric factory returned a mismatching ModelSpec.")
        metrics.reset()

        weighted_loss = 0.0
        scalar_count = 0
        batch_count = 0
        sample_count = 0
        context: AbstractContextManager[object]
        context = nullcontext() if optimizing else torch.no_grad()

        with context:
            for raw_batch in batches:
                batch = raw_batch.to(self.device)
                validate_forecast_batch(
                    self.forecast_model.spec,
                    batch.dynamic,
                    batch.target,
                    batch.static,
                )

                if optimizing:
                    assert self.optimizer is not None
                    self.optimizer.zero_grad(set_to_none=True)
                prediction = self.forecast_model(batch.dynamic, batch.static)
                loss_value = self.loss(prediction, batch.target)
                if loss_value.ndim != 0:
                    raise ValueError("Forecast loss must return a scalar tensor.")
                if optimizing:
                    assert self.optimizer is not None
                    torch.autograd.backward(loss_value)
                    self.optimizer.step()

                metrics.update(prediction.detach(), batch.target.detach())
                count = batch.target.numel()
                weighted_loss += float(loss_value.detach().item()) * count
                scalar_count += count
                batch_count += 1
                sample_count += batch.batch_size

        if batch_count == 0:
            raise ValueError("Cannot run an epoch over zero batches.")
        return EpochResult(
            loss=weighted_loss / scalar_count,
            metrics=metrics.compute(),
            batches=batch_count,
            samples=sample_count,
        )
