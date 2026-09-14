"""Execute the official 0.6 real RADKLIM-YW three-seed reference benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from weather_radar_ml.config.experiment import ComponentConfig
from weather_radar_ml.config.ml_dataset import (
    MLDatasetConfig,
    ml_dataset_config_from_mapping,
)
from weather_radar_ml.config.schema import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    RunConfig,
    TrackingConfig,
    TrainingConfig,
)
from weather_radar_ml.data.ml.artifact import plan_ml_dataset_artifact
from weather_radar_ml.training.multiseed import run_multiseed_experiment

_DEFAULT_DATASET_CONFIG = Path(
    "configs/datasets/radklim_yw_2023_09_dortmund_reference.yaml"
)
_DEFAULT_CHECKSUMS = Path(
    "data/prepared/radklim-yw-2023-09-dortmund-128x128-v1.checksums.json"
)


def main() -> None:
    """Load the immutable dataset build and run the held-out test benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--ml-datasets-root", type=Path, default=Path("data/ml"))
    parser.add_argument(
        "--tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlruns/mlflow.db",
    )
    parser.add_argument("--experiment-name", default="weather-radar-ml-0.6-reference")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must not be negative")

    dataset = _load_dataset_config(_DEFAULT_DATASET_CONFIG)
    checksums = _load_checksums(_DEFAULT_CHECKSUMS)
    plan = plan_ml_dataset_artifact(
        config=dataset,
        source_checksums=checksums,
        output_root=args.ml_datasets_root,
    )
    if not plan.paths.catalog.is_file():
        raise FileNotFoundError(
            "Reference ML dataset has not been built. Run "
            "'uv run python scripts/prepare_radklim_yw_reference.py' first. "
            f"Expected catalog: {plan.paths.catalog}"
        )

    config = RunConfig(
        name="radklim-yw-2023-09-dortmund-reference",
        data=DataConfig(
            name="ml_dataset",
            catalog_path=plan.paths.catalog,
            dataset=dataset,
        ),
        model=ModelConfig(name="unet", parameters={"base_channels": 32}),
        training=TrainingConfig(
            seed=17,
            epochs=50,
            batch_size=4,
            device=args.device,
            num_workers=args.num_workers,
            deterministic_algorithms=True,
            early_stopping_patience=7,
            loss=ComponentConfig("mse"),
            optimizer=ComponentConfig(
                "adam",
                {"lr": 0.001, "weight_decay": 0.0},
            ),
        ),
        tracking=TrackingConfig(
            uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            enabled=args.tracking,
        ),
        output=OutputConfig(runs_root=args.runs_root),
    )

    result = run_multiseed_experiment(config, benchmark_split="test")
    print(f"reference_seed={result.reference_seed}")
    print(f"summary={result.artifact.summary}")
    print(f"forecast_artifact={result.forecast_artifact.root}")
    print(f"forecast_split={result.forecast_artifact.split}")
    print(f"test_seed_count={len(result.test_research)}")


def _load_dataset_config(path: Path) -> MLDatasetConfig:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, Mapping):
        raise TypeError("Reference dataset configuration root must be a mapping.")
    return ml_dataset_config_from_mapping(cast(Mapping[str, Any], raw))


def _load_checksums(path: Path) -> dict[str, str]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
    ):
        raise TypeError("Reference checksum file must map strings to strings.")
    return cast(dict[str, str], raw)


if __name__ == "__main__":
    main()
