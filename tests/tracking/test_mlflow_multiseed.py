import json
from dataclasses import dataclass
from pathlib import Path

from weather_radar_ml.tracking.mlflow import (
    MlflowTrackingSettings,
    finish_mlflow_parent_run,
    start_mlflow_parent_run,
    track_run_artifact,
)


@dataclass
class _Experiment:
    experiment_id: str


@dataclass
class _RunInfo:
    run_id: str


@dataclass
class _Run:
    info: _RunInfo


class _Client:
    def __init__(self) -> None:
        self.experiment = _Experiment("experiment")
        self.created_runs: list[tuple[str, dict[str, object] | None, str | None]] = []
        self.params: list[tuple[str, str, object]] = []
        self.metrics: list[tuple[str, str, float, int | None]] = []
        self.artifacts: list[tuple[str, str, str | None]] = []
        self.terminated: list[tuple[str, str | None]] = []

    def get_experiment_by_name(self, name: str) -> _Experiment | None:
        del name
        return self.experiment

    def create_experiment(self, name: str) -> str:
        del name
        return "created"

    def create_run(
        self,
        experiment_id: str,
        *,
        tags: dict[str, object] | None = None,
        run_name: str | None = None,
    ) -> _Run:
        self.created_runs.append((experiment_id, tags, run_name))
        return _Run(_RunInfo(f"run-{len(self.created_runs)}"))

    def log_param(self, run_id: str, key: str, value: object) -> object:
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


def test_parent_run_records_fixed_seeds_and_finishes_with_summary(
    tmp_path: Path,
) -> None:
    client = _Client()
    settings = MlflowTrackingSettings(artifact_path="group")
    parent = start_mlflow_parent_run(
        run_name="reference-multi-seed",
        seeds=(17, 42, 73),
        settings=settings,
        client=client,
    )
    summary = tmp_path / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")

    finish_mlflow_parent_run(
        parent.mlflow_run_id,
        status="FINISHED",
        settings=settings,
        summary_path=summary,
        metrics={"multi_seed/rmse/mean": 1.25},
        client=client,
    )

    assert client.created_runs[0][1] == {
        "weather_radar_ml.run_type": "multi_seed_parent"
    }
    assert ("run-1", "multi_seed.seeds", "[17,42,73]") in client.params
    assert ("run-1", "multi_seed.seed_count", 3) in client.params
    assert ("run-1", "multi_seed/rmse/mean", 1.25, None) in client.metrics
    assert client.artifacts == [("run-1", str(summary), "group")]
    assert client.terminated == [("run-1", "FINISHED")]


def test_child_tracking_uses_mlflow_parent_run_tag(tmp_path: Path) -> None:
    client = _Client()
    root = _write_run(tmp_path)

    track_run_artifact(
        root,
        settings=MlflowTrackingSettings(parent_run_id="parent-123"),
        client=client,
    )

    tags = client.created_runs[0][1]
    assert tags is not None
    assert tags["mlflow.parentRunId"] == "parent-123"


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
                "created_at": "2026-09-14T12:00:00+00:00",
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
            "model": {"name": "toy", "parameters": {}},
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "seed": 17,
                "deterministic_algorithms": True,
            },
        },
    )
    result = {
        "loss": 1.0,
        "batches": 1,
        "samples": 1,
        "metrics": {
            "overall": {
                "values": {"mae": 1.0, "rmse": 1.0, "bias": 0.0},
                "count": 1,
            },
            "per_lead": [
                {
                    "values": {"mae": 1.0, "rmse": 1.0, "bias": 0.0},
                    "count": 1,
                }
            ],
        },
    }
    _write_json(
        root / "history.json",
        {"epochs": [{"epoch": 1, "train": result, "validation": result}]},
    )
    return root


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
