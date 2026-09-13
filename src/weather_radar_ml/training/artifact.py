"""Immutable on-disk artifacts for completed forecast experiment runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from weather_radar_ml.config.experiment import (
    ExperimentConfig,
    experiment_config_payload,
)
from weather_radar_ml.evaluation.domain import ForecastMetricReport, MetricSummary
from weather_radar_ml.training.domain import EpochResult, TrainingHistory
from weather_radar_ml.training.run import RunMetadata

RUN_ARTIFACT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    """Checkpoint to package with a completed run artifact."""

    source: Path
    completed_epochs: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        if self.completed_epochs <= 0:
            raise ValueError("completed_epochs must be greater than zero.")


@dataclass(frozen=True, slots=True)
class RunArtifactResult:
    """Paths published for one immutable experiment run."""

    root: Path
    manifest: Path
    config: Path
    history: Path
    checkpoints: tuple[Path, ...]


def write_run_artifact(
    runs_root: str | Path,
    *,
    config: ExperimentConfig,
    metadata: RunMetadata,
    history: TrainingHistory,
    checkpoints: tuple[CheckpointArtifact, ...] = (),
) -> RunArtifactResult:
    """Atomically publish a completed run and its reproducibility metadata."""
    _validate_run(config, metadata, history, checkpoints)
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = _safe_run_id(metadata.run_id)
    destination = root / run_id
    if destination.exists():
        raise FileExistsError(f"Run artifact already exists: {destination}")

    staging = Path(tempfile.mkdtemp(dir=root, prefix=f".{run_id}.tmp-"))
    try:
        config_path = staging / "config.json"
        history_path = staging / "history.json"
        _write_json(config_path, experiment_config_payload(config))
        _write_json(history_path, _history_payload(history))

        checkpoint_records, checkpoint_paths = _copy_checkpoints(
            checkpoints,
            staging,
        )
        manifest_payload = {
            "format_version": RUN_ARTIFACT_FORMAT_VERSION,
            "run": _run_metadata_payload(metadata),
            "config_fingerprint": config.fingerprint,
            "files": {
                "config": _file_record(config_path, staging),
                "history": _file_record(history_path, staging),
                "checkpoints": checkpoint_records,
            },
            "results": {
                "completed_epochs": len(history.train),
                "final_train": _epoch_payload(history.train[-1]),
                "final_validation": _epoch_payload(history.validation[-1]),
            },
        }
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest_payload)

        if destination.exists():
            raise FileExistsError(f"Run artifact already exists: {destination}")
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return RunArtifactResult(
        root=destination,
        manifest=destination / "manifest.json",
        config=destination / "config.json",
        history=destination / "history.json",
        checkpoints=tuple(destination / path for path in checkpoint_paths),
    )


def _validate_run(
    config: ExperimentConfig,
    metadata: RunMetadata,
    history: TrainingHistory,
    checkpoints: tuple[CheckpointArtifact, ...],
) -> None:
    if metadata.config_fingerprint != config.fingerprint:
        raise ValueError("Run metadata does not match the experiment configuration.")
    if len(history.train) != config.training.epochs:
        raise ValueError(
            "Training history must contain exactly the configured number of epochs."
        )

    completed_epochs = [checkpoint.completed_epochs for checkpoint in checkpoints]
    if len(completed_epochs) != len(set(completed_epochs)):
        raise ValueError("Checkpoint completed_epochs values must be unique.")
    for checkpoint in checkpoints:
        if checkpoint.completed_epochs > len(history.train):
            raise ValueError(
                "Checkpoint completed_epochs cannot exceed the training history."
            )
        if not checkpoint.source.is_file():
            raise FileNotFoundError(checkpoint.source)


def _copy_checkpoints(
    checkpoints: tuple[CheckpointArtifact, ...],
    staging: Path,
) -> tuple[list[dict[str, object]], tuple[Path, ...]]:
    if not checkpoints:
        return [], ()

    checkpoint_dir = staging / "checkpoints"
    checkpoint_dir.mkdir()
    records: list[dict[str, object]] = []
    relative_paths: list[Path] = []
    for checkpoint in sorted(checkpoints, key=lambda item: item.completed_epochs):
        relative = Path("checkpoints") / (
            f"checkpoint-{checkpoint.completed_epochs:04d}.pt"
        )
        destination = staging / relative
        shutil.copyfile(checkpoint.source, destination)
        record = _file_record(destination, staging)
        record["completed_epochs"] = checkpoint.completed_epochs
        records.append(record)
        relative_paths.append(relative)
    return records, tuple(relative_paths)


def _history_payload(history: TrainingHistory) -> dict[str, object]:
    return {
        "epochs": [
            {
                "epoch": index,
                "train": _epoch_payload(train),
                "validation": _epoch_payload(validation),
            }
            for index, (train, validation) in enumerate(
                zip(history.train, history.validation, strict=True),
                start=1,
            )
        ]
    }


def _epoch_payload(result: EpochResult) -> dict[str, object]:
    if not isfinite(result.loss):
        raise ValueError("Epoch loss must be finite before artifact persistence.")
    return {
        "loss": result.loss,
        "batches": result.batches,
        "samples": result.samples,
        "metrics": _metric_report_payload(result.metrics),
    }


def _metric_report_payload(report: ForecastMetricReport) -> dict[str, object]:
    return {
        "overall": _metric_summary_payload(report.overall),
        "per_lead": [_metric_summary_payload(item) for item in report.per_lead],
    }


def _metric_summary_payload(summary: MetricSummary) -> dict[str, object]:
    return {
        "values": dict(sorted(summary.values.items())),
        "count": summary.count,
    }


def _run_metadata_payload(metadata: RunMetadata) -> dict[str, object]:
    return {
        "run_id": metadata.run_id,
        "created_at": metadata.created_at.isoformat(),
        "git_commit": metadata.git_commit,
        "python_version": metadata.python_version,
        "numpy_version": metadata.numpy_version,
        "torch_version": metadata.torch_version,
        "platform": metadata.platform,
    }


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(serialized + "\n", encoding="utf-8")


def _safe_run_id(run_id: str) -> str:
    if run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single safe path segment.")
    return run_id
