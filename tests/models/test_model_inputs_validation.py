import torch

from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.models.validation import validate_model_inputs


def test_validate_model_inputs_supports_inference_without_targets() -> None:
    spec = ModelSpec(
        name="baseline",
        dynamic_input_features=("rain", "quality"),
        static_input_features=("elevation",),
        target_features=("rain",),
        history_steps=4,
        forecast_steps=2,
    )

    validate_model_inputs(
        spec,
        dynamic=torch.zeros(3, 4, 2, 8, 9),
        static=torch.zeros(3, 1, 8, 9),
    )
