"""Typed experiment configuration with stable reproducibility fingerprints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import isfinite
from types import MappingProxyType

from weather_radar_ml.models.domain import ModelSpec


@dataclass(frozen=True, slots=True)
class ComponentConfig:
    """Named model, loss, or optimizer component with immutable parameters."""

    name: str
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_name(self.name, "component name"))
        object.__setattr__(
            self,
            "parameters",
            _freeze_parameter_mapping(self.parameters),
        )


@dataclass(frozen=True, slots=True)
class DatasetReference:
    """Stable identity of the ML dataset consumed by one experiment."""

    dataset_build_id: str
    feature_schema_id: str
    artifact_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_build_id",
            _require_name(self.dataset_build_id, "dataset_build_id"),
        )
        object.__setattr__(
            self,
            "feature_schema_id",
            _require_name(self.feature_schema_id, "feature_schema_id"),
        )
        if self.artifact_fingerprint is not None:
            object.__setattr__(
                self,
                "artifact_fingerprint",
                _require_name(
                    self.artifact_fingerprint,
                    "artifact_fingerprint",
                ),
            )


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Run-level settings that materially affect training reproducibility."""

    epochs: int
    batch_size: int
    seed: int
    deterministic_algorithms: bool = True
    device: str = "cpu"
    num_workers: int = 0
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 2**32 - 1.")
        device = self.device.strip()
        if not device:
            raise ValueError("device must not be empty.")
        object.__setattr__(self, "device", device)
        if self.num_workers < 0:
            raise ValueError("num_workers must not be negative.")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be greater than zero.")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete logical description of one forecast-training experiment."""

    name: str
    dataset: DatasetReference
    model_spec: ModelSpec
    model: ComponentConfig
    loss: ComponentConfig
    optimizer: ComponentConfig
    training: TrainingSettings

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_name(self.name, "experiment name"))
        if self.model.name != self.model_spec.name:
            raise ValueError(
                "model component name must match ModelSpec.name: "
                f"{self.model.name!r} != {self.model_spec.name!r}."
            )

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 digest of the logical experiment config."""
        canonical = json.dumps(
            experiment_config_payload(self),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def experiment_config_payload(config: ExperimentConfig) -> dict[str, object]:
    """Return a JSON-serializable representation used for manifests and hashes."""
    return {
        "name": config.name,
        "dataset": {
            "dataset_build_id": config.dataset.dataset_build_id,
            "feature_schema_id": config.dataset.feature_schema_id,
            "artifact_fingerprint": config.dataset.artifact_fingerprint,
        },
        "model_spec": asdict(config.model_spec),
        "model": _component_payload(config.model),
        "loss": _component_payload(config.loss),
        "optimizer": _component_payload(config.optimizer),
        "training": asdict(config.training),
    }


def _component_payload(config: ComponentConfig) -> dict[str, object]:
    return {
        "name": config.name,
        "parameters": _json_parameter(config.parameters),
    }


def _freeze_parameter_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    normalized: dict[str, object] = {}
    for raw_key, value in values.items():
        key = _require_name(raw_key, "component parameter name")
        if key in normalized:
            raise ValueError(f"Duplicate component parameter name {key!r}.")
        normalized[key] = _freeze_parameter(value, f"component parameter {key!r}")
    return MappingProxyType(dict(sorted(normalized.items())))


def _freeze_parameter(value: object, field: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{field} must be finite.")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_parameter(item, field) for item in value)
    if isinstance(value, Mapping):
        return _freeze_parameter_mapping(value)
    raise TypeError(f"{field} must contain only JSON-compatible values.")


def _json_parameter(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_parameter(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_parameter(item) for item in value]
    return value


def _require_name(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty.")
    return normalized
