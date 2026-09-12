import pytest

from weather_radar_ml.models.domain import ModelSpec


def test_model_spec_normalizes_names() -> None:
    spec = ModelSpec(
        name=" baseline ",
        dynamic_input_features=(" rain ",),
        static_input_features=(" elevation ",),
        target_features=(" rain_target ",),
        history_steps=6,
        forecast_steps=3,
    )

    assert spec.name == "baseline"
    assert spec.dynamic_input_features == ("rain",)
    assert spec.static_input_features == ("elevation",)
    assert spec.target_features == ("rain_target",)


@pytest.mark.parametrize("field", ["history_steps", "forecast_steps"])
def test_model_spec_rejects_non_positive_steps(field: str) -> None:
    kwargs = {"history_steps": 6, "forecast_steps": 3}
    kwargs[field] = 0

    with pytest.raises(ValueError, match="greater than zero"):
        ModelSpec(
            name="baseline",
            dynamic_input_features=("rain",),
            target_features=("rain_target",),
            **kwargs,
        )


def test_model_spec_rejects_duplicate_features() -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        ModelSpec(
            name="baseline",
            dynamic_input_features=("rain", "rain"),
            target_features=("rain_target",),
            history_steps=6,
            forecast_steps=3,
        )
