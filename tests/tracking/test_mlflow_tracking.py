from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from weather_radar_ml.tracking.mlflow import (
    MlflowTrackingSettings,
    track_run_artifact,
)


@dataclass
class FakeExperiment:
    experiment_id: str


@dataclass
class FakeRunInfo:
    run_id: str


@dataclass
class FakeRun:
    info: FakeRunInfo


class FakeClient:
    def __init__(self, *, existing_experiment: bool = False) -> None:
        self.experiment = FakeExperiment("existing") if existing_experiment else None
        self.created_experiments: list[str] = []
        self.created_runs: list[tuple[str, dict[str, object] | None, str | None]] = []
        self.params: list[tuple[str, str, object]] = []
        self.metrics: list[tuple[str, str, float, int | None]] = []
        self.artifacts: list[tuple[str, str, str | None]] = []
        self.terminated: list[tuple[str, str | None]] = []
        self.fail_on_param = False

    def get_experiment_by_name(self, name: str) -> FakeExperiment | None:
        return self.experiment

    def create_experiment(self, name: str) -> str:
        self.created_experiments.append(name)
        return "created"

    def create_run(
        self,
        experiment_id: str,
        *,
        tags: dict[str, object] | None = None,
        run_name: str | None = None,
    ) -> FakeRun:
        self.created_runs.append((experiment_id, tags, run_name))
        return FakeRun(FakeRunInfo("mlflow-run"))

    def log_param(self, run_id: str, key: str, value: object) -> object:
        if self.fail_on_param:
            raise RuntimeError("tracking unavailable")
        self.params.append((run_id, key, value))
        return value

    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        *,
        step: int | None = None,
    ) -> object:
        self.metrics.append((run_id, key, value, step))
        return None

    def log_artifact(
        self,
        run_id: str,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None:
        self.artifacts.append((run_id, local_path, artifact_path))

    def set_terminated(self, run_id: str, status: str | None = None) -> None:
        self.terminated.append((run_id, status))


def test_settings_reject_empty_values() -> None:
    with pytest.raises(ValueError, match="tracking_uri must be a non-empty string"):
        MlflowTrackingSettings(tracking_uri="  ")


def test_tracking_creates_experiment_and_logs_run(tmp_path: Path) -> None:
    root = _write_run(tmp_path)
    client = FakeClient()

    result = track_run_artifact(root, client=client)

    assert result.local_run_id == "local-run"
    assert result.experiment_id == "created"
    assert result.mlflow_run_id == "mlflow-run"
    assert client.created_experiments == ["weather-radar-ml"]
    assert client.created_runs[0][0] == "created"
    assert client.created_runs[0][2] == "local-run"
    assert client.created_runs[0][1] == {
        "weather_radar_ml.run_id": "local-run",
        "weather_radar_ml.config_fingerprint": "fingerprint",
        "weather_radar_ml.artifact_format_version": "1",
        "weather_radar_ml.git_commit": "abc123",
        "weather_radar_ml.python_version": "3.11.16",
        "weather_radar_ml.numpy_version": "2.0.0",
        "weather_radar_ml.torch_version": "2.8.0",
        "weather_radar_ml.platform": "test-platform",
    }
    assert client.terminated == [("mlflow-run", "FINISHED")]


def test_tracking_reuses_existing_experiment(tmp_path: Path) -> None:
    client = FakeClient(existing_experiment=True)

    result = track_run_artifact(_write_run(tmp_path), client=client)

    assert result.experiment_id == "existing"
    assert client.created_experiments == []
    assert client.created_runs[0][0] == "existing"


def test_tracking_logs_flattened_config_and_epoch_metrics(tmp_path: Path) -> None:
    client = FakeClient()

    track_run_artifact(_write_run(tmp_path), client=client)

    params = {(key, value) for _, key, value in client.params}
    assert ("config.dataset.dataset_build_id", "dataset-v1") in params
    assert ("config.model.parameters.channels", "[16,32]") in params
    assert ("config.training.deterministic_algorithms", "true") in params

    metrics = {(key, value, step) for _, key, value, step in client.metrics}
    assert ("train/loss", 3.0, 1) in metrics
    assert ("validation/rmse", 2.0, 1) in metrics
    assert ("validation/lead_001/mae", 1.5, 1) in metrics
    assert ("validation/loss", 2.0, 2) in metrics


def test_tracking_logs_only_small_canonical_artifacts(tmp_path: Path) -> None:
    client = FakeClient()
    root = _write_run(tmp_path)
    checkpoint = root / "checkpoints" / "checkpoint-0002.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"large-model-state")

    track_run_artifact(root, client=client)

    logged_names = {Path(local_path).name for _, local_path, _ in client.artifacts}
    assert logged_names == {"manifest.json", "config.json", "history.json"}
    assert all(
        artifact_path == "run-artifact" for _, _, artifact_path in client.artifacts
    )
    assert checkpoint.name not in logged_names


def test_tracking_failure_marks_mlflow_run_failed(tmp_path: Path) -> None:
    client = FakeClient()
    client.fail_on_param = True

    with pytest.raises(RuntimeError, match="tracking unavailable"):
        track_run_artifact(_write_run(tmp_path), client=client)

    assert client.terminated == [("mlflow-run", "FAILED")]


def test_missing_canonical_artifact_is_rejected(tmp_path: Path) -> None:
    root = _write_run(tmp_path)
    (root / "history.json").unlink()

    with pytest.raises(FileNotFoundError, match="history\.json"):
        track_run_artifact(root, client=FakeClient())


def _write_run(tmp_path: Path) -> Path:
    root = tmp_path / "local-run"
    root.mkdir()
    _write_json(
        root / "manifest.json",
        {
            "format_version": 1,
            "config_fingerprint": "fingerprint",
            "run": {
                "run_id": "local-run",
                "created_at": "2026-09-13T12:00:00+00:00",
                "git_commit": "abc123",
                "python_version": "3.11.16",
                "numpy_version": "2.0.0",
                "torch_version": "2.8.0",
                "platform": "test-platform",
            },
        },
    )
    _write_json(
        root / "config.json",
        {
            "name": "smoke",
            "dataset": {
                "dataset_build_id": "dataset-v1",
                "feature_schema_id": "radar-v1",
                "artifact_fingerprint": None,
            },
            "model": {
                "name": "toy",
                "parameters": {"channels": [16, 32]},
            },
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "seed": 42,
                "deterministic_algorithms": True,
            },
        },
    )
    _write_json(
        root / "history.json",
        {
            "epochs": [
                _epoch(1, train_loss=3.0, validation_loss=2.5),
                _epoch(2, train_loss=2.5, validation_loss=2.0),
            ]
        },
    )
    return root


def _epoch(
    epoch: int, *, train_loss: float, validation_loss: float
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "train": _result(train_loss, mae=2.0, rmse=2.5),
        "validation": _result(validation_loss, mae=1.5, rmse=2.0),
    }


def _result(loss: float, *, mae: float, rmse: float) -> dict[str, object]:
    return {
        "loss": loss,
        "batches": 1,
        "samples": 2,
        "metrics": {
            "overall": {
                "values": {"mae": mae, "rmse": rmse, "bias": 0.25},
                "count": 8,
            },
            "per_lead": [
                {
                    "values": {"mae": mae, "rmse": rmse, "bias": 0.25},
                    "count": 8,
                }
            ],
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
