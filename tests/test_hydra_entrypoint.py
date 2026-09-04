"""Integration tests for Hydra composition at the application boundary."""

from pathlib import Path

from hydra import compose, initialize_config_dir

from weather_radar_ml.entrypoints.run import describe_run, to_run_config

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_default_hydra_configuration_composes_into_core_config() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        hydra_config = compose(config_name="config")

    config = to_run_config(hydra_config)

    assert describe_run(config) == (
        "data=synthetic, model=persistence, seed=42, tracking=weather-radar-ml"
    )


def test_hydra_cli_style_override_changes_composed_value() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        hydra_config = compose(
            config_name="config",
            overrides=["training.seed=7"],
        )

    assert to_run_config(hydra_config).training.seed == 7
