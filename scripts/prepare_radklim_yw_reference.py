"""Prepare and index the immutable real RADKLIM-YW 0.6 reference dataset."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from weather_radar_ml.config.ml_dataset import (
    MLDatasetConfig,
    ml_dataset_config_from_mapping,
)
from weather_radar_ml.data.dwd.reference import (
    DORTMUND_REFERENCE_ROI,
    prepare_monthly_roi,
)
from weather_radar_ml.data.dwd.source import DwdRadklimYwMonthlySource
from weather_radar_ml.data.manifest import (
    DatasetManifest,
    sha256_directory,
    sha256_file,
)
from weather_radar_ml.data.ml.artifact import build_ml_dataset_artifact

_YEAR = 2023
_MONTH = 9
_DATASET_ID = "radklim-yw-2023-09-dortmund-128x128-v1"
_DEFAULT_CONFIG = Path("configs/datasets/radklim_yw_2023_09_dortmund_reference.yaml")


def main() -> None:
    """Prepare the fixed real-data source and immutable ML sample catalog."""
    root = Path("data")
    raw_dir = root / "raw" / "radklim-yw" / f"{_YEAR}-{_MONTH:02d}"
    prepared = root / "prepared" / f"{_DATASET_ID}.zarr"
    manifest_path = root / "prepared" / f"{_DATASET_ID}.manifest.json"
    checksums_path = root / "prepared" / f"{_DATASET_ID}.checksums.json"
    reference_path = root / "prepared" / f"{_DATASET_ID}.reference.json"
    ml_root = root / "ml"

    source = DwdRadklimYwMonthlySource(_YEAR, _MONTH)
    archive = source.acquire(raw_dir)[0]
    raw_sha256 = sha256_file(archive)

    preparation_summary: dict[str, object] | None = None
    if not prepared.exists():
        summary = prepare_monthly_roi(
            archive,
            prepared,
            year=_YEAR,
            month=_MONTH,
            roi=DORTMUND_REFERENCE_ROI,
        )
        preparation_summary = {
            "frames": summary.frames,
            "first_timestamp": summary.first_timestamp.isoformat(),
            "last_timestamp": summary.last_timestamp.isoformat(),
            "missing_frame_slots": summary.missing_frame_slots,
            "rejected_incomplete_frames": summary.rejected_incomplete_frames,
        }
        print(
            "prepared_frames="
            f"{summary.frames} missing_frame_slots={summary.missing_frame_slots} "
            f"rejected_incomplete_frames={summary.rejected_incomplete_frames}"
        )
    else:
        print(f"prepared_reused={prepared}")
        if reference_path.is_file():
            previous = json.loads(reference_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict):
                candidate = previous.get("preparation")
                if isinstance(candidate, dict):
                    preparation_summary = candidate

    prepared_sha256 = sha256_directory(prepared)
    roi = DORTMUND_REFERENCE_ROI
    DatasetManifest(
        dataset_id=_DATASET_ID,
        source_product="RADKLIM-YW",
        source_version="2017.002",
        pipeline_version="0.6.6",
        schema_version="1",
        source_checksums={archive.name: raw_sha256},
    ).write(manifest_path)
    reference_metadata: dict[str, object] = {
        "dataset_id": _DATASET_ID,
        "prepared_sha256": prepared_sha256,
        "reference_month": f"{_YEAR}-{_MONTH:02d}",
        "roi": {
            "name": "dortmund-centred-128km",
            "x": roi.x,
            "y": roi.y,
            "width": roi.width,
            "height": roi.height,
            "grid_resolution_m": 1000,
            "approximate_centre_lon": 7.4653,
            "approximate_centre_lat": 51.5136,
        },
    }
    if preparation_summary is not None:
        reference_metadata["preparation"] = preparation_summary
    reference_path.write_text(
        json.dumps(reference_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums_path.write_text(
        json.dumps({"radklim": prepared_sha256}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dataset_config = _load_dataset_config(_DEFAULT_CONFIG)
    expected_path = dataset_config.sources[0].path
    if expected_path != prepared:
        raise ValueError(
            "Reference dataset config must point to the prepared reference Zarr: "
            f"expected {prepared}, got {expected_path}."
        )
    result = build_ml_dataset_artifact(
        config=dataset_config,
        source_checksums={"radklim": prepared_sha256},
        output_root=ml_root,
    )

    print(f"raw={archive}")
    print(f"prepared={prepared}")
    print(f"manifest={manifest_path}")
    print(f"checksums={checksums_path}")
    print(f"reference_metadata={reference_path}")
    print(f"dataset_build_id={result.plan.dataset_build_id}")
    print(f"catalog={result.plan.paths.catalog}")
    print(f"samples={result.report.samples}")
    print(f"samples_by_split={dict(result.report.samples_by_split)}")


def _load_dataset_config(path: Path) -> MLDatasetConfig:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, Mapping):
        raise TypeError("Reference dataset configuration root must be a mapping.")
    return ml_dataset_config_from_mapping(cast(Mapping[str, Any], raw))


if __name__ == "__main__":
    main()
