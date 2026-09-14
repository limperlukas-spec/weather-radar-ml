from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from weather_radar_ml.config.ml_dataset import ml_dataset_config_from_mapping


def test_real_reference_dataset_config_has_locked_06_geometry_and_splits() -> None:
    path = Path("configs/datasets/radklim_yw_2023_09_dortmund_reference.yaml")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(raw, Mapping)
    config = ml_dataset_config_from_mapping(cast(Mapping[str, Any], raw))

    assert config.temporal.step_minutes == 5
    assert config.temporal.history_steps == 12
    assert config.temporal.forecast_steps == 6
    assert config.spatial.mode.value == "full_frame"
    assert config.sources[0].artifact_id == "radklim-yw-2023-09-dortmund-128x128-v1"
    assert [interval.name.value for interval in config.splits.intervals] == [
        "train",
        "validation",
        "test",
    ]
