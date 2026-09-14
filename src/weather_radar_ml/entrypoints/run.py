"""Hydra-backed experiment entrypoint.

Hydra is deliberately confined to this boundary module. Application components
receive the typed :class:`RunConfig` rather than Hydra ``DictConfig`` objects.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from weather_radar_ml.config import RunConfig, run_config_from_mapping
from weather_radar_ml.training.pipeline import PipelineResult, run_experiment

_CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


def to_run_config(cfg: DictConfig) -> RunConfig:
    """Adapt a Hydra configuration to the framework-independent core schema."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, Mapping):
        raise TypeError("The composed root configuration must be a mapping.")
    return run_config_from_mapping(cast(Mapping[str, Any], raw))


def describe_run(config: RunConfig) -> str:
    """Return a deterministic summary useful before execution."""
    return (
        f"data={config.data.name}, model={config.model.name}, "
        f"seed={config.training.seed}, tracking={config.tracking.experiment_name}"
    )


def execute_run(config: RunConfig) -> PipelineResult:
    """Execute the typed runtime configuration through the common pipeline."""
    return run_experiment(config)


def describe_result(result: PipelineResult) -> str:
    """Return a concise machine-readable summary for CLI users and CI logs."""
    parts = [
        f"run_id={result.artifact.root.name}",
        f"artifact={result.artifact.root}",
    ]
    if result.tracking is not None:
        parts.append(f"mlflow_run_id={result.tracking.mlflow_run_id}")
    return " ".join(parts)


@hydra.main(version_base="1.3", config_path=_CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Compose configuration, execute the run, and print its persisted identity."""
    config = to_run_config(cfg)
    result = execute_run(config)
    print(describe_result(result))


if __name__ == "__main__":
    main()
