import torch

from weather_radar_ml.models.baseline import PersistenceForecast
from weather_radar_ml.models.domain import ModelSpec


def test_persistence_supports_explicit_target_to_input_mapping() -> None:
    spec = ModelSpec(
        name="persistence",
        dynamic_input_features=("rain_input", "temperature"),
        target_features=("rain_target",),
        history_steps=2,
        forecast_steps=2,
    )
    model = PersistenceForecast(spec, target_input_features=("rain_input",))
    dynamic = torch.tensor([[[[[1.0]], [[10.0]]], [[[2.0]], [[20.0]]]]])

    prediction = model(dynamic)

    assert prediction.shape == (1, 2, 1, 1, 1)
    torch.testing.assert_close(prediction[:, 0], torch.tensor([[[[2.0]]]]))
    torch.testing.assert_close(prediction[:, 1], torch.tensor([[[[2.0]]]]))
