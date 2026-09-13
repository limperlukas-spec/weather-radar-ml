from types import MappingProxyType

import pytest

from weather_radar_ml.config.experiment import (
    ComponentConfig,
    DatasetReference,
    ExperimentConfig,
    TrainingSettings,
    experiment_config_payload,
)
from weather_radar_ml.models.domain import ModelSpec


def _spec() -> ModelSpec:
    return ModelSpec(
        name="tiny-conv",
        dynamic_input_features=("rain",),
        target_features=("rain",),
        history_steps=4,
        forecast_steps=2,
    )


def _config(
    *,
    seed: int = 7,
    model_parameters: dict[str, object] | None = None,
) -> ExperimentConfig:
    parameters = (
        {"hidden_channels": 16} if model_parameters is None else model_parameters
    )
    return ExperimentConfig(
        name="smoke experiment",
        dataset=DatasetReference(
            dataset_build_id="dataset-build-001",
            feature_schema_id="features-v1",
            artifact_fingerprint="dvc:abc123",
        ),
        model_spec=_spec(),
        model=ComponentConfig("tiny-conv", parameters),
        loss=ComponentConfig("mse"),
        optimizer=ComponentConfig("adam", {"lr": 0.001}),
        training=TrainingSettings(epochs=3, batch_size=8, seed=seed),
    )


def test_component_parameters_are_normalized_and_immutable() -> None:
    component = ComponentConfig(" adam ", {" weight_decay ": 0.0, "lr": 0.001})

    assert component.name == "adam"
    assert isinstance(component.parameters, MappingProxyType)
    assert component.parameters == {"lr": 0.001, "weight_decay": 0.0}
    with pytest.raises(TypeError):
        component.parameters["lr"] = 0.5  # type: ignore[index]


def test_component_supports_nested_json_parameters_immutably() -> None:
    component = ComponentConfig(
        "model",
        {"layers": [16, 32], "scheduler": {"name": "cosine", "warmup": 2}},
    )

    assert component.parameters["layers"] == (16, 32)
    scheduler = component.parameters["scheduler"]
    assert isinstance(scheduler, MappingProxyType)
    assert scheduler["name"] == "cosine"


def test_component_rejects_non_finite_or_non_json_parameters() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ComponentConfig("adam", {"lr": float("nan")})
    with pytest.raises(TypeError, match="JSON-compatible"):
        ComponentConfig("model", {"unsupported": object()})


def test_experiment_fingerprint_is_independent_of_parameter_order() -> None:
    first = _config(model_parameters={"hidden_channels": 16, "dropout": 0.1})
    second = _config(model_parameters={"dropout": 0.1, "hidden_channels": 16})

    assert first.fingerprint == second.fingerprint


def test_experiment_fingerprint_changes_when_training_settings_change() -> None:
    assert _config(seed=7).fingerprint != _config(seed=8).fingerprint


def test_experiment_payload_is_json_ready_and_contains_model_spec() -> None:
    payload = experiment_config_payload(_config())

    assert payload["name"] == "smoke experiment"
    assert payload["model_spec"]["history_steps"] == 4  # type: ignore[index]
    dataset = payload["dataset"]
    assert isinstance(dataset, dict)
    assert dataset["dataset_build_id"] == "dataset-build-001"


def test_experiment_requires_model_component_to_match_model_spec() -> None:
    with pytest.raises(ValueError, match=r"must match ModelSpec\.name"):
        ExperimentConfig(
            name="bad",
            dataset=DatasetReference("build", "schema"),
            model_spec=_spec(),
            model=ComponentConfig("other-model"),
            loss=ComponentConfig("mse"),
            optimizer=ComponentConfig("adam"),
            training=TrainingSettings(epochs=1, batch_size=1, seed=0),
        )


def test_training_settings_validate_seed_range() -> None:
    with pytest.raises(ValueError, match="seed must be between"):
        TrainingSettings(epochs=1, batch_size=1, seed=2**32)
