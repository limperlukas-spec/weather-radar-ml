import pytest
import torch

from weather_radar_ml.models.baseline import PersistenceForecast
from weather_radar_ml.models.contracts import ForecastModel
from weather_radar_ml.models.domain import ModelSpec


def _spec(
    *,
    dynamic_features: tuple[str, ...] = ("rain", "quality"),
    target_features: tuple[str, ...] = ("rain",),
    static_features: tuple[str, ...] = (),
    forecast_steps: int = 3,
) -> ModelSpec:
    return ModelSpec(
        name="persistence",
        dynamic_input_features=dynamic_features,
        static_input_features=static_features,
        target_features=target_features,
        history_steps=4,
        forecast_steps=forecast_steps,
    )


def test_persistence_satisfies_forecast_model_protocol() -> None:
    model = PersistenceForecast(_spec())

    assert isinstance(model, ForecastModel)
    assert model.spec.name == "persistence"


def test_persistence_repeats_latest_observation() -> None:
    model = PersistenceForecast(_spec(forecast_steps=2))
    dynamic = torch.zeros(1, 4, 2, 2, 3)
    dynamic[:, -1, 0] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    prediction = model(dynamic)

    assert prediction.shape == (1, 2, 1, 2, 3)
    torch.testing.assert_close(prediction[:, 0, 0], dynamic[:, -1, 0])
    torch.testing.assert_close(prediction[:, 1, 0], dynamic[:, -1, 0])


def test_persistence_maps_target_features_by_name_and_order() -> None:
    model = PersistenceForecast(
        _spec(
            dynamic_features=("quality", "rain", "temperature"),
            target_features=("temperature", "rain"),
            forecast_steps=1,
        )
    )
    dynamic = torch.zeros(2, 4, 3, 2, 2)
    dynamic[:, -1, 0] = 10.0
    dynamic[:, -1, 1] = 20.0
    dynamic[:, -1, 2] = 30.0

    prediction = model(dynamic)

    assert prediction.shape == (2, 1, 2, 2, 2)
    torch.testing.assert_close(prediction[:, 0, 0], dynamic[:, -1, 2])
    torch.testing.assert_close(prediction[:, 0, 1], dynamic[:, -1, 1])


def test_persistence_preserves_dtype_and_device() -> None:
    model = PersistenceForecast(_spec()).to(dtype=torch.float64)
    dynamic = torch.zeros(2, 4, 2, 3, 4, dtype=torch.float64)

    prediction = model(dynamic)

    assert prediction.dtype == dynamic.dtype
    assert prediction.device == dynamic.device


def test_persistence_accepts_declared_static_inputs_without_using_them() -> None:
    model = PersistenceForecast(_spec(static_features=("elevation",)))
    dynamic = torch.zeros(2, 4, 2, 3, 4)
    static = torch.ones(2, 1, 3, 4)

    prediction = model(dynamic, static)

    assert prediction.shape == (2, 3, 1, 3, 4)


def test_persistence_rejects_target_not_present_in_dynamic_inputs() -> None:
    with pytest.raises(ValueError, match="must also be dynamic input features"):
        PersistenceForecast(
            _spec(
                dynamic_features=("quality",),
                target_features=("rain",),
            )
        )


def test_persistence_validates_runtime_history_shape() -> None:
    model = PersistenceForecast(_spec())

    with pytest.raises(ValueError, match="expected 4, got 3"):
        model(torch.zeros(1, 3, 2, 2, 2))
