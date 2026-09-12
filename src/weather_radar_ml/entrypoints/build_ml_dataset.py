"""CLI for building a reproducible ML dataset artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from weather_radar_ml.config.ml_dataset import ml_dataset_config_from_mapping
from weather_radar_ml.data.ml.artifact import (
    build_ml_dataset_artifact,
    plan_ml_dataset_artifact,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build a versioned ML radar dataset artifact."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checksums", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    data = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(data, Mapping):
        raise TypeError("ML dataset config root must be a mapping.")
    config = ml_dataset_config_from_mapping(cast(Mapping[str, Any], data))
    raw = json.loads(args.checksums.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in raw.items()
    ):
        raise TypeError(
            "Checksum file must contain a JSON object of string keys and values."
        )
    checksums = cast(dict[str, str], raw)
    if args.dry_run:
        plan = plan_ml_dataset_artifact(
            config=config, source_checksums=checksums, output_root=args.output
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "dataset_build_id": plan.dataset_build_id,
                    "artifact_root": str(plan.paths.root),
                },
                sort_keys=True,
            )
        )
        return 0
    result = build_ml_dataset_artifact(
        config=config, source_checksums=checksums, output_root=args.output
    )
    print(
        json.dumps(
            {
                "dry_run": False,
                "dataset_build_id": result.plan.dataset_build_id,
                "artifact_root": str(result.plan.paths.root),
                "samples": result.report.samples,
                "reused_existing": result.reused_existing,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
