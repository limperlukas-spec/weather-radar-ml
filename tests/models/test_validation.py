import pytest
import torch

from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.models.validation import (
    validate_forecast_batch,
    validate_prediction,
)


def _spec(*, static: bool = False) -> ModelSpec:
    return ModelSpec(
        name="baseline",
        dynamic_input_features=("rain", "quality"),
        static_input_features=("elevation",) if static else (),
        target_features=("rain",),
        history_steps=4,
        forecast_steps=2,
    )


def test_validate_forecast_batch_accepts_matching_tensors() -> None:
    validate_forecast_batch(
        _spec(static=True),
        dynamic=torch.zeros(3, 4, 2, 8, 9),
        static=torch.zeros(3, 1, 8, 9),
        target=torch.zeros(3, 2, 1, 8, 9),
    )


def test_validate_forecast_batch_requires_declared_static_inputs() -> None:
    with pytest.raises(ValueError, match="requires static inputs"):
        validate_forecast_batch(
            _spec(static=True),
            dynamic=torch.zeros(1, 4, 2, 8, 9),
            target=torch.zeros(1, 2, 1, 8, 9),
        )


def test_validate_forecast_batch_rejects_wrong_history() -> None:
    with pytest.raises(ValueError, match="expected 4, got 3"):
        validate_forecast_batch(
            _spec(),
            dynamic=torch.zeros(1, 3, 2, 8, 9),
            target=torch.zeros(1, 2, 1, 8, 9),
        )


def test_validate_forecast_batch_rejects_spatial_mismatch() -> None:
    with pytest.raises(ValueError, match="same spatial shape"):
        validate_forecast_batch(
            _spec(),
            dynamic=torch.zeros(1, 4, 2, 8, 9),
            target=torch.zeros(1, 2, 1, 7, 9),
        )


def test_validate_prediction_accepts_matching_output() -> None:
    target = torch.zeros(2, 2, 1, 8, 9)
    validate_prediction(_spec(), target.clone(), target)


def test_validate_prediction_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="Prediction shape must match target shape"):
        validate_prediction(
            _spec(),
            torch.zeros(2, 1, 1, 8, 9),
            torch.zeros(2, 2, 1, 8, 9),
        )
