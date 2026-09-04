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

_CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


def to_run_config(cfg: DictConfig) -> RunConfig:
    """Adapt a Hydra configuration to the framework-independent core schema."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, Mapping):
        raise TypeError("The composed root configuration must be a mapping.")
    return run_config_from_mapping(cast(Mapping[str, Any], raw))


def describe_run(config: RunConfig) -> str:
    """Return a deterministic summary useful for the foundation smoke test."""
    return (
        f"data={config.data.name}, model={config.model.name}, "
        f"seed={config.training.seed}, tracking={config.tracking.experiment_name}"
    )


@hydra.main(version_base="1.3", config_path=_CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Compose configuration and expose the first runnable project entrypoint."""
    config = to_run_config(cfg)
    print(describe_run(config))


if __name__ == "__main__":
    main()
