"""Tests for the framework-independent typed configuration schema."""

import pytest

from weather_radar_ml.config import run_config_from_mapping


def _valid_config() -> dict[str, object]:
    return {
        "data": {"name": "synthetic"},
        "model": {"name": "persistence"},
        "training": {"seed": 42},
        "tracking": {
            "uri": "./mlruns",
            "experiment_name": "weather-radar-ml",
        },
    }


def test_run_config_from_mapping_builds_typed_config() -> None:
    config = run_config_from_mapping(_valid_config())

    assert config.data.name == "synthetic"
    assert config.model.name == "persistence"
    assert config.training.seed == 42
    assert config.tracking.uri == "./mlruns"
    assert config.tracking.experiment_name == "weather-radar-ml"


def test_run_config_from_mapping_rejects_non_mapping_section() -> None:
    raw = _valid_config()
    raw["model"] = "persistence"

    with pytest.raises(TypeError, match="section 'model'"):
        run_config_from_mapping(raw)
