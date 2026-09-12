"""Continuous deterministic metrics for radar forecasts."""

from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from weather_radar_ml.evaluation.domain import ForecastMetricReport, MetricSummary
from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.models.validation import validate_prediction


class ContinuousForecastMetrics:
    """Accumulate MAE, RMSE, and signed bias without batch-averaging error.

    Sufficient statistics are accumulated over all scalar forecast values. This
    means results are invariant to how the same dataset is split into batches.
    The same statistics are retained independently for each forecast lead step.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self.reset()

    @property
    def spec(self) -> ModelSpec:
        """Return the forecast tensor contract used by the accumulator."""
        return self._spec

    def reset(self) -> None:
        """Discard all accumulated sufficient statistics."""
        self._sum_abs = 0.0
        self._sum_squared = 0.0
        self._sum_error = 0.0
        self._count = 0
        self._lead_sum_abs = [0.0] * self.spec.forecast_steps
        self._lead_sum_squared = [0.0] * self.spec.forecast_steps
        self._lead_sum_error = [0.0] * self.spec.forecast_steps
        self._lead_count = [0] * self.spec.forecast_steps

    def update(self, prediction: Tensor, target: Tensor) -> None:
        """Accumulate one batch after validating shape and finite values."""
        validate_prediction(self.spec, prediction, target)
        _require_finite(prediction, "prediction")
        _require_finite(target, "target")

        error = (prediction.detach() - target.detach()).to(dtype=torch.float64)
        absolute = error.abs()
        squared = error.square()

        self._sum_abs += float(absolute.sum().item())
        self._sum_squared += float(squared.sum().item())
        self._sum_error += float(error.sum().item())
        self._count += error.numel()

        lead_reduce_dims = (0, 2, 3, 4)
        lead_abs = absolute.sum(dim=lead_reduce_dims).cpu().tolist()
        lead_squared = squared.sum(dim=lead_reduce_dims).cpu().tolist()
        lead_error = error.sum(dim=lead_reduce_dims).cpu().tolist()
        values_per_lead = (
            error.shape[0] * error.shape[2] * error.shape[3] * error.shape[4]
        )

        for lead in range(self.spec.forecast_steps):
            self._lead_sum_abs[lead] += float(lead_abs[lead])
            self._lead_sum_squared[lead] += float(lead_squared[lead])
            self._lead_sum_error[lead] += float(lead_error[lead])
            self._lead_count[lead] += values_per_lead

    def compute(self) -> ForecastMetricReport:
        """Return exact overall and per-lead metrics for all observed batches."""
        if self._count == 0:
            raise RuntimeError("No forecast batches have been accumulated.")

        overall = _summary(
            sum_abs=self._sum_abs,
            sum_squared=self._sum_squared,
            sum_error=self._sum_error,
            count=self._count,
        )
        per_lead = tuple(
            _summary(
                sum_abs=self._lead_sum_abs[lead],
                sum_squared=self._lead_sum_squared[lead],
                sum_error=self._lead_sum_error[lead],
                count=self._lead_count[lead],
            )
            for lead in range(self.spec.forecast_steps)
        )
        return ForecastMetricReport(overall=overall, per_lead=per_lead)


def _summary(
    *,
    sum_abs: float,
    sum_squared: float,
    sum_error: float,
    count: int,
) -> MetricSummary:
    return MetricSummary(
        values={
            "mae": sum_abs / count,
            "rmse": sqrt(sum_squared / count),
            "bias": sum_error / count,
        },
        count=count,
    )


def _require_finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values.")
