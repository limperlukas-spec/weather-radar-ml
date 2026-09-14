"""Behavior tests for the first learned forecast U-Net."""

import pytest
import torch

from weather_radar_ml.models.unet import UNetForecast


def test_unet_maps_reference_contract_to_six_forecast_steps() -> None:
    model = UNetForecast(base_channels=4)
    inputs = torch.rand((2, 12, 1, 32, 40), dtype=torch.float32)

    prediction = model(inputs)

    assert prediction.shape == (2, 6, 1, 32, 40)
    assert torch.all(torch.isfinite(prediction))


@pytest.mark.parametrize("height,width", [(16, 16), (31, 37), (48, 64)])
def test_unet_is_not_bound_to_one_spatial_size(height: int, width: int) -> None:
    model = UNetForecast(base_channels=4)
    inputs = torch.rand((1, 12, 1, height, width), dtype=torch.float32)

    prediction = model(inputs)

    assert prediction.shape == (1, 6, 1, height, width)


def test_unet_supports_explicit_feature_dimensions() -> None:
    model = UNetForecast(
        input_steps=4,
        input_features=2,
        output_steps=3,
        output_features=2,
        base_channels=4,
    )

    prediction = model(torch.rand((2, 4, 2, 20, 24), dtype=torch.float32))

    assert prediction.shape == (2, 3, 2, 20, 24)


def test_unet_rejects_wrong_time_dimension() -> None:
    model = UNetForecast(base_channels=4)

    with pytest.raises(ValueError, match="12 input steps"):
        model(torch.rand((1, 11, 1, 16, 16), dtype=torch.float32))


def test_unet_rejects_wrong_feature_dimension() -> None:
    model = UNetForecast(base_channels=4)

    with pytest.raises(ValueError, match="1 input features"):
        model(torch.rand((1, 12, 2, 16, 16), dtype=torch.float32))


def test_unet_rejects_non_contract_rank() -> None:
    model = UNetForecast(base_channels=4)

    with pytest.raises(ValueError, match=r"\[B, T, F, H, W\]"):
        model(torch.rand((1, 12, 16, 16), dtype=torch.float32))


def test_unet_propagates_gradients() -> None:
    model = UNetForecast(base_channels=4)
    inputs = torch.rand((1, 12, 1, 16, 16), dtype=torch.float32)

    loss = model(inputs).square().mean()
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert any(
        torch.any(gradient != 0).item()
        for gradient in gradients
        if gradient is not None
    )


def test_unet_initialization_is_reproducible_for_same_seed() -> None:
    torch.manual_seed(42)
    first = UNetForecast(base_channels=4)
    torch.manual_seed(42)
    second = UNetForecast(base_channels=4)

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_unet_rejects_too_small_spatial_dimensions() -> None:
    model = UNetForecast(base_channels=4)

    with pytest.raises(ValueError, match="at least 4"):
        model(torch.rand((1, 12, 1, 3, 16), dtype=torch.float32))


def test_unet_exposes_model_spec_for_common_training_pipeline() -> None:
    from weather_radar_ml.models.domain import ModelSpec

    spec = ModelSpec(
        name="unet",
        dynamic_input_features=("rain",),
        target_features=("rain",),
        history_steps=2,
        forecast_steps=1,
    )
    model = UNetForecast(spec, base_channels=4)

    assert model.spec == spec
    prediction = model(torch.rand((1, 2, 1, 8, 8)), None)
    assert prediction.shape == (1, 1, 1, 8, 8)


def test_unet_rejects_static_model_spec() -> None:
    from weather_radar_ml.models.domain import ModelSpec

    spec = ModelSpec(
        name="unet",
        dynamic_input_features=("rain",),
        static_input_features=("height",),
        target_features=("rain",),
        history_steps=2,
        forecast_steps=1,
    )

    with pytest.raises(ValueError, match="does not support static inputs"):
        UNetForecast(spec, base_channels=4)
