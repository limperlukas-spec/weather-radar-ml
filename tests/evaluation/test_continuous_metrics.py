import pytest
import torch

from weather_radar_ml.evaluation.continuous import ContinuousForecastMetrics
from weather_radar_ml.evaluation.contracts import ForecastMetricAccumulator
from weather_radar_ml.models.domain import ModelSpec


def _spec(*, forecast_steps: int = 2) -> ModelSpec:
    return ModelSpec(
        name="metric-test",
        dynamic_input_features=("rain",),
        target_features=("rain",),
        history_steps=2,
        forecast_steps=forecast_steps,
    )


def test_continuous_metrics_satisfy_accumulator_protocol() -> None:
    metrics = ContinuousForecastMetrics(_spec())

    assert isinstance(metrics, ForecastMetricAccumulator)


def test_continuous_metrics_are_zero_for_perfect_prediction() -> None:
    metrics = ContinuousForecastMetrics(_spec())
    target = torch.ones(2, 2, 1, 3, 4)

    metrics.update(target.clone(), target)
    report = metrics.compute()

    assert report.overall.values == {"mae": 0.0, "rmse": 0.0, "bias": 0.0}
    assert report.overall.count == 48
    assert tuple(summary.count for summary in report.per_lead) == (24, 24)


def test_continuous_metrics_compute_known_overall_and_per_lead_values() -> None:
    metrics = ContinuousForecastMetrics(_spec())
    prediction = torch.tensor([[[[[1.0]]], [[[3.0]]]]])
    target = torch.zeros_like(prediction)

    metrics.update(prediction, target)
    report = metrics.compute()

    assert report.overall.values["mae"] == pytest.approx(2.0)
    assert report.overall.values["rmse"] == pytest.approx(5.0**0.5)
    assert report.overall.values["bias"] == pytest.approx(2.0)
    assert report.per_lead[0].values["mae"] == pytest.approx(1.0)
    assert report.per_lead[1].values["mae"] == pytest.approx(3.0)


def test_continuous_metrics_weight_batches_by_scalar_count() -> None:
    metrics = ContinuousForecastMetrics(_spec(forecast_steps=1))

    metrics.update(torch.tensor([[[[[1.0]]]]]), torch.zeros(1, 1, 1, 1, 1))
    metrics.update(
        torch.full((3, 1, 1, 1, 1), 3.0),
        torch.zeros(3, 1, 1, 1, 1),
    )
    report = metrics.compute()

    assert report.overall.values["mae"] == pytest.approx(2.5)
    assert report.overall.values["rmse"] == pytest.approx(7.0**0.5)
    assert report.overall.values["bias"] == pytest.approx(2.5)
    assert report.overall.count == 4


def test_continuous_metrics_use_signed_bias() -> None:
    metrics = ContinuousForecastMetrics(_spec())
    prediction = torch.tensor([[[[[-2.0]]], [[[1.0]]]]])
    target = torch.zeros_like(prediction)

    metrics.update(prediction, target)
    report = metrics.compute()

    assert report.overall.values["bias"] == pytest.approx(-0.5)


def test_continuous_metrics_reset_discards_previous_batches() -> None:
    metrics = ContinuousForecastMetrics(_spec(forecast_steps=1))
    metrics.update(torch.ones(1, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 1))

    metrics.reset()

    with pytest.raises(RuntimeError, match="No forecast batches"):
        metrics.compute()


def test_continuous_metrics_reject_non_finite_values() -> None:
    metrics = ContinuousForecastMetrics(_spec())
    target = torch.zeros(1, 2, 1, 1, 1)
    target[0, 1, 0, 0, 0] = torch.inf

    with pytest.raises(ValueError, match="target contains non-finite"):
        metrics.update(torch.zeros_like(target), target)


def test_continuous_metrics_validate_prediction_shape() -> None:
    metrics = ContinuousForecastMetrics(_spec())

    with pytest.raises(ValueError, match="Prediction shape must match target shape"):
        metrics.update(torch.zeros(1, 1, 1, 1, 1), torch.zeros(1, 2, 1, 1, 1))
