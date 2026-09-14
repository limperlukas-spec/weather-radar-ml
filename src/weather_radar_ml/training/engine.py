"""Minimal reusable PyTorch training orchestration for radar forecasts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from typing import cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from weather_radar_ml.evaluation.contracts import ForecastMetricAccumulator
from weather_radar_ml.models.contracts import ForecastModel
from weather_radar_ml.models.validation import validate_forecast_batch
from weather_radar_ml.training.contracts import ForecastLoss
from weather_radar_ml.training.domain import EpochResult, ForecastBatch, TrainingHistory

MetricFactory = Callable[[], ForecastMetricAccumulator]
TensorTransform = Callable[[Tensor], Tensor]
BestValidationCallback = Callable[[int, EpochResult], None]


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
        input_transform: TensorTransform | None = None,
        target_transform: TensorTransform | None = None,
        metric_prediction_transform: TensorTransform | None = None,
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
        self.input_transform = input_transform or _identity
        self.target_transform = target_transform or _identity
        self.metric_prediction_transform = metric_prediction_transform or _identity
        self.model.to(self.device)

    def fit(
        self,
        train_batches: Iterable[ForecastBatch],
        validation_batches: Iterable[ForecastBatch],
        *,
        epochs: int,
        early_stopping_patience: int | None = None,
        on_best_validation: BestValidationCallback | None = None,
    ) -> TrainingHistory:
        """Run matching train/validation passes with optional early stopping."""
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")
        if early_stopping_patience is not None and early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be greater than zero.")

        train_results: list[EpochResult] = []
        validation_results: list[EpochResult] = []
        best_epoch = 0
        best_validation_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            train_results.append(self.run_train_epoch(train_batches))
            validation = self.run_validation_epoch(validation_batches)
            validation_results.append(validation)

            if not torch.isfinite(torch.tensor(validation.loss)).item():
                raise ValueError("Validation loss must be finite.")
            if validation.loss < best_validation_loss:
                best_epoch = epoch
                best_validation_loss = validation.loss
                epochs_without_improvement = 0
                if on_best_validation is not None:
                    on_best_validation(epoch, validation)
            else:
                epochs_without_improvement += 1

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                break

        return TrainingHistory(
            train=tuple(train_results),
            validation=tuple(validation_results),
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            stopped_early=len(train_results) < epochs,
        )

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
                model_input = self.input_transform(batch.dynamic)
                loss_target = self.target_transform(batch.target)
                prediction = self.forecast_model(model_input, batch.static)
                loss_value = self.loss(prediction, loss_target)
                if loss_value.ndim != 0:
                    raise ValueError("Forecast loss must return a scalar tensor.")
                if optimizing:
                    assert self.optimizer is not None
                    torch.autograd.backward(loss_value)
                    self.optimizer.step()

                metric_prediction = self.metric_prediction_transform(
                    prediction.detach()
                )
                metrics.update(metric_prediction, batch.target.detach())
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


def _identity(values: Tensor) -> Tensor:
    return values
