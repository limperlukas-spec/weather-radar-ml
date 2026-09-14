"""Typed runtime configuration independent of Hydra.

Hydra is restricted to the composition boundary.  The application core receives
these dataclasses and can therefore be exercised directly from tests or another
frontend without depending on ``DictConfig`` objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from weather_radar_ml.config.experiment import ComponentConfig
from weather_radar_ml.config.ml_dataset import (
    MLDatasetConfig,
    ml_dataset_config_from_mapping,
)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Runtime data source and deterministic loader geometry."""

    name: str
    catalog_path: Path | None = None
    dataset: MLDatasetConfig | None = None
    source_paths: Mapping[str, Path] = field(default_factory=dict)
    train_samples: int = 8
    validation_samples: int = 4
    history_steps: int = 2
    forecast_steps: int = 1
    dynamic_input_features: tuple[str, ...] = ("rain",)
    static_input_features: tuple[str, ...] = ()
    target_features: tuple[str, ...] = ("rain",)
    height: int = 8
    width: int = 8

    def __post_init__(self) -> None:
        normalized_name = _non_empty(self.name, "data.name")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(
            self,
            "source_paths",
            MappingProxyType(
                {
                    _non_empty(key, "data.source_paths key"): Path(value)
                    for key, value in sorted(self.source_paths.items())
                }
            ),
        )
        for field_name in (
            "train_samples",
            "validation_samples",
            "history_steps",
            "forecast_steps",
            "height",
            "width",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"data.{field_name} must be greater than zero.")
        _require_unique_names(self.dynamic_input_features, "dynamic_input_features")
        _require_unique_names(
            self.static_input_features,
            "static_input_features",
            allow_empty=True,
        )
        _require_unique_names(self.target_features, "target_features")

        if normalized_name == "ml_dataset" and (
            self.catalog_path is None or self.dataset is None
        ):
            raise ValueError(
                "data.name='ml_dataset' requires catalog_path and dataset."
            )
        if normalized_name == "synthetic" and (
            self.catalog_path is not None or self.dataset is not None
        ):
            raise ValueError(
                "Synthetic data must not configure catalog_path or dataset."
            )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Forecast model strategy and constructor parameters."""

    name: str
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        component = ComponentConfig(self.name, self.parameters)
        object.__setattr__(self, "name", component.name)
        object.__setattr__(self, "parameters", component.parameters)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training/runtime settings consumed by the common forecast pipeline."""

    seed: int
    epochs: int = 1
    batch_size: int = 4
    device: str = "cpu"
    num_workers: int = 0
    deterministic_algorithms: bool = True
    early_stopping_patience: int | None = None
    loss: ComponentConfig = field(default_factory=lambda: ComponentConfig("mse"))
    optimizer: ComponentConfig = field(default_factory=lambda: ComponentConfig("none"))

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("training.seed must be between 0 and 2**32 - 1.")
        if self.epochs <= 0:
            raise ValueError("training.epochs must be greater than zero.")
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be greater than zero.")
        if self.num_workers < 0:
            raise ValueError("training.num_workers must not be negative.")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError(
                "training.early_stopping_patience must be greater than zero."
            )
        object.__setattr__(self, "device", _non_empty(self.device, "training.device"))


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Optional MLflow publication settings."""

    uri: str
    experiment_name: str
    enabled: bool = False
    artifact_path: str = "run-artifact"

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _non_empty(self.uri, "tracking.uri"))
        object.__setattr__(
            self,
            "experiment_name",
            _non_empty(self.experiment_name, "tracking.experiment_name"),
        )
        object.__setattr__(
            self,
            "artifact_path",
            _non_empty(self.artifact_path, "tracking.artifact_path"),
        )


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Local canonical run-artifact destination."""

    runs_root: Path = field(default_factory=lambda: Path("runs"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs_root", Path(self.runs_root))


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Fully composed typed runtime configuration for one experiment run."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    tracking: TrackingConfig
    name: str = "weather-radar-ml"
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "name"))


def run_config_from_mapping(mapping: Mapping[str, Any]) -> RunConfig:
    """Convert a composed plain mapping into the typed runtime configuration."""
    data = _section(mapping, "data")
    model = _section(mapping, "model")
    training = _section(mapping, "training")
    tracking = _section(mapping, "tracking")
    output = _optional_section(mapping, "output")

    data_name = str(data["name"])
    ml_dataset_raw = data.get("dataset")
    ml_dataset = (
        None
        if ml_dataset_raw is None
        else ml_dataset_config_from_mapping(
            _require_mapping(ml_dataset_raw, "data.dataset")
        )
    )
    source_paths_raw = _optional_mapping(data.get("source_paths"), "data.source_paths")

    return RunConfig(
        name=str(mapping.get("name", "weather-radar-ml")),
        data=DataConfig(
            name=data_name,
            catalog_path=_optional_path(data.get("catalog_path")),
            dataset=ml_dataset,
            source_paths={
                str(key): Path(str(value)) for key, value in source_paths_raw.items()
            },
            train_samples=int(data.get("train_samples", 8)),
            validation_samples=int(data.get("validation_samples", 4)),
            history_steps=int(data.get("history_steps", 2)),
            forecast_steps=int(data.get("forecast_steps", 1)),
            dynamic_input_features=_string_tuple(
                data.get("dynamic_input_features", ("rain",)),
                "data.dynamic_input_features",
            ),
            static_input_features=_string_tuple(
                data.get("static_input_features", ()),
                "data.static_input_features",
                allow_empty=True,
            ),
            target_features=_string_tuple(
                data.get("target_features", ("rain",)),
                "data.target_features",
            ),
            height=int(data.get("height", 8)),
            width=int(data.get("width", 8)),
        ),
        model=ModelConfig(
            name=str(model["name"]),
            parameters=_optional_mapping(model.get("parameters"), "model.parameters"),
        ),
        training=TrainingConfig(
            seed=int(training["seed"]),
            epochs=int(training.get("epochs", 1)),
            batch_size=int(training.get("batch_size", 4)),
            device=str(training.get("device", "cpu")),
            num_workers=int(training.get("num_workers", 0)),
            deterministic_algorithms=bool(
                training.get("deterministic_algorithms", True)
            ),
            early_stopping_patience=_optional_int(
                training.get("early_stopping_patience"),
                "training.early_stopping_patience",
            ),
            loss=_component(training.get("loss"), "mse", "training.loss"),
            optimizer=_component(
                training.get("optimizer"), "none", "training.optimizer"
            ),
        ),
        tracking=TrackingConfig(
            uri=str(tracking["uri"]),
            experiment_name=str(tracking["experiment_name"]),
            enabled=bool(tracking.get("enabled", False)),
            artifact_path=str(tracking.get("artifact_path", "run-artifact")),
        ),
        output=OutputConfig(
            runs_root=Path(str(output.get("runs_root", "runs"))),
        ),
    )


def _component(value: Any, default_name: str, field_name: str) -> ComponentConfig:
    if value is None:
        return ComponentConfig(default_name)
    mapping = _require_mapping(value, field_name)
    return ComponentConfig(
        name=str(mapping.get("name", default_name)),
        parameters=_optional_mapping(
            mapping.get("parameters"), f"{field_name}.parameters"
        ),
    )


def _section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in mapping:
        raise KeyError(f"Missing configuration section {key!r}.")
    return _require_mapping(mapping[key], f"configuration section {key!r}")


def _optional_section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if value is None:
        return {}
    return _require_mapping(value, f"configuration section {key!r}")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return cast(Mapping[str, Any], value)


def _optional_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _require_mapping(value, name)


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or null.")
    return int(value)


def _optional_path(value: Any) -> Path | None:
    return None if value is None else Path(str(value))


def _string_tuple(
    value: Any, name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings.")
    result = tuple(str(item) for item in value)
    if not result and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    return result


def _require_unique_names(
    values: tuple[str, ...],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> None:
    normalized = tuple(_non_empty(value, f"data.{field_name}") for value in values)
    if not normalized and not allow_empty:
        raise ValueError(f"data.{field_name} must not be empty.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"data.{field_name} must not contain duplicates.")


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized
