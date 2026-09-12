"""Reproducible artifact build service for ML dataset indexes."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import xarray as xr

from weather_radar_ml.config.ml_dataset import MLDatasetConfig
from weather_radar_ml.data.ml.builder import (
    BuildReport,
    SampleIndexBuilder,
    dataset_build_id,
)
from weather_radar_ml.data.ml.catalog import SQLiteSampleCatalog


@dataclass(frozen=True, slots=True)
class DatasetArtifactPaths:
    root: Path
    catalog: Path
    manifest: Path
    resolved_config: Path


@dataclass(frozen=True, slots=True)
class DatasetArtifactPlan:
    dataset_build_id: str
    paths: DatasetArtifactPaths


@dataclass(frozen=True, slots=True)
class DatasetArtifactResult:
    plan: DatasetArtifactPlan
    report: BuildReport
    reused_existing: bool = False


def plan_ml_dataset_artifact(
    *,
    config: MLDatasetConfig,
    source_checksums: dict[str, str],
    output_root: str | Path,
) -> DatasetArtifactPlan:
    _validate_checksums(config, source_checksums)
    build_id = dataset_build_id(config, source_checksums)
    root = Path(output_root) / build_id
    return DatasetArtifactPlan(
        build_id,
        DatasetArtifactPaths(
            root,
            root / "samples.sqlite",
            root / "manifest.json",
            root / "config.resolved.yaml",
        ),
    )


def build_ml_dataset_artifact(
    *,
    config: MLDatasetConfig,
    source_checksums: dict[str, str],
    output_root: str | Path,
) -> DatasetArtifactResult:
    plan = plan_ml_dataset_artifact(
        config=config, source_checksums=source_checksums, output_root=output_root
    )
    _validate_source_paths(config)
    if plan.paths.root.exists():
        return _load_existing(plan)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    tmp = Path(
        tempfile.mkdtemp(prefix=f".{plan.dataset_build_id}-", dir=output_root_path)
    )
    datasets: dict[str, xr.Dataset] = {}
    try:
        for source in config.sources:
            assert source.path is not None
            datasets[source.source_id] = cast(
                xr.Dataset, xr.open_zarr(source.path, consolidated=False)
            )
        catalog = SQLiteSampleCatalog(tmp / "samples.sqlite")
        report = SampleIndexBuilder(
            config=config,
            source_datasets=datasets,
            source_checksums=source_checksums,
            catalog=catalog,
        ).build()
        (tmp / "config.resolved.yaml").write_text(
            json.dumps(config.to_resolved_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest: dict[str, Any] = {
            "artifact_schema_version": 1,
            "dataset_build_id": report.dataset_build_id,
            "feature_schema_id": report.feature_schema_id,
            "dataset_name": config.name,
            "sample_schema_version": config.sample_schema_version,
            "sources": [
                {
                    "source_id": s.source_id,
                    "artifact_id": s.artifact_id,
                    "role": s.role.value,
                    "checksum": source_checksums[s.source_id],
                }
                for s in config.sources
            ],
            "files": {
                "catalog": "samples.sqlite",
                "resolved_config": "config.resolved.yaml",
            },
            "build_report": asdict(report),
        }
        (tmp / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        for ds in datasets.values():
            ds.close()
    tmp.rename(plan.paths.root)
    return DatasetArtifactResult(plan=plan, report=report)


def _load_existing(plan: DatasetArtifactPlan) -> DatasetArtifactResult:
    required = (plan.paths.catalog, plan.paths.manifest, plan.paths.resolved_config)
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"Existing dataset artifact is incomplete; missing "
            f"{sorted(missing)!r}: {plan.paths.root}"
        )
    data = json.loads(plan.paths.manifest.read_text(encoding="utf-8"))
    if data.get("dataset_build_id") != plan.dataset_build_id:
        raise RuntimeError("Existing manifest does not match the planned build ID.")
    report_data = data.get("build_report")
    if not isinstance(report_data, dict):
        raise RuntimeError("Existing manifest has no valid build_report.")
    return DatasetArtifactResult(
        plan=plan, report=BuildReport(**report_data), reused_existing=True
    )


def _validate_checksums(
    config: MLDatasetConfig, source_checksums: dict[str, str]
) -> None:
    expected = {s.source_id for s in config.sources}
    if set(source_checksums) != expected:
        raise ValueError(
            f"Source checksum IDs must exactly match configured sources; "
            f"expected {sorted(expected)!r}, "
            f"got {sorted(source_checksums)!r}."
        )
    empty = sorted(k for k, v in source_checksums.items() if not v.strip())
    if empty:
        raise ValueError(f"Source checksums must not be empty: {empty!r}.")


def _validate_source_paths(config: MLDatasetConfig) -> None:
    for source in config.sources:
        if source.path is None:
            raise ValueError(
                f"Source {source.source_id!r} has no configured storage path."
            )
        if not source.path.exists():
            raise FileNotFoundError(source.path)
