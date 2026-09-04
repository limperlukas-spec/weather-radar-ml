"""Typed configuration schemas used by the application core.

This module intentionally has no dependency on Hydra. Hydra is restricted to the
entrypoint/composition layer so that core code remains framework-independent.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Configuration identifying an input data strategy."""

    name: str


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration identifying a forecast model strategy."""

    name: str


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Minimal training configuration shared across model strategies."""

    seed: int


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Experiment tracking configuration."""

    uri: str
    experiment_name: str


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Fully composed, typed configuration for one experiment run."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    tracking: TrackingConfig


def _section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section '{key}' must be a mapping.")
    return cast(Mapping[str, Any], value)


def run_config_from_mapping(mapping: Mapping[str, Any]) -> RunConfig:
    """Convert an untyped composed mapping into the typed core configuration."""
    data = _section(mapping, "data")
    model = _section(mapping, "model")
    training = _section(mapping, "training")
    tracking = _section(mapping, "tracking")

    return RunConfig(
        data=DataConfig(name=str(data["name"])),
        model=ModelConfig(name=str(model["name"])),
        training=TrainingConfig(seed=int(training["seed"])),
        tracking=TrackingConfig(
            uri=str(tracking["uri"]),
            experiment_name=str(tracking["experiment_name"]),
        ),
    )
