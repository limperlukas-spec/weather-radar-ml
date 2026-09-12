import pytest
import torch

from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.training.contracts import ForecastLoss
from weather_radar_ml.training.losses import MeanSquaredForecastLoss


def _spec() -> ModelSpec:
    return ModelSpec(
        name="loss-test",
        dynamic_input_features=("rain",),
        target_features=("rain",),
        history_steps=2,
        forecast_steps=2,
    )


def test_mse_loss_satisfies_forecast_loss_protocol() -> None:
    loss = MeanSquaredForecastLoss(_spec())

    assert isinstance(loss, ForecastLoss)
    assert loss.name == "mse"
    assert loss.spec.forecast_steps == 2


def test_mse_loss_returns_known_value() -> None:
    loss = MeanSquaredForecastLoss(_spec())
    prediction = torch.tensor([[[[[0.0]]], [[[2.0]]]]])
    target = torch.zeros_like(prediction)

    value = loss(prediction, target)

    assert value.item() == pytest.approx(2.0)


def test_mse_loss_is_differentiable() -> None:
    loss = MeanSquaredForecastLoss(_spec())
    prediction = torch.tensor(
        [[[[[1.0]]], [[[3.0]]]]],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)

    value = loss(prediction, target)
    value.backward()

    assert prediction.grad is not None
    torch.testing.assert_close(
        prediction.grad,
        torch.tensor([[[[[1.0]]], [[[3.0]]]]]),
    )


def test_mse_loss_rejects_prediction_shape_mismatch() -> None:
    loss = MeanSquaredForecastLoss(_spec())

    with pytest.raises(ValueError, match="Prediction shape must match target shape"):
        loss(torch.zeros(1, 1, 1, 2, 2), torch.zeros(1, 2, 1, 2, 2))


def test_mse_loss_rejects_non_finite_values() -> None:
    loss = MeanSquaredForecastLoss(_spec())
    prediction = torch.zeros(1, 2, 1, 1, 1)
    prediction[0, 0, 0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="prediction contains non-finite"):
        loss(prediction, torch.zeros_like(prediction))
