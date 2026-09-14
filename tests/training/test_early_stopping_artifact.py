import json
from datetime import UTC, datetime
from pathlib import Path

from weather_radar_ml.config.experiment import (
    ComponentConfig,
    DatasetReference,
    ExperimentConfig,
    TrainingSettings,
)
from weather_radar_ml.evaluation.domain import ForecastMetricReport, MetricSummary
from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.training.artifact import (
    CheckpointArtifact,
    write_run_artifact,
)
from weather_radar_ml.training.domain import EpochResult, TrainingHistory
from weather_radar_ml.training.run import create_run_metadata


def _config(*, epochs: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        name="early-stopping-artifact-test",
        dataset=DatasetReference("dataset-build", "features-v1", "dvc:abc"),
        model_spec=ModelSpec(
            name="toy",
            dynamic_input_features=("rain",),
            target_features=("rain",),
            history_steps=2,
            forecast_steps=2,
        ),
        model=ComponentConfig("toy"),
        loss=ComponentConfig("mse"),
        optimizer=ComponentConfig("adam", {"lr": 0.001}),
        training=TrainingSettings(epochs=epochs, batch_size=2, seed=42),
    )


def _epoch(loss: float) -> EpochResult:
    summary = MetricSummary({"mae": 1.0, "rmse": 2.0, "bias": 0.0}, count=8)
    return EpochResult(
        loss=loss,
        metrics=ForecastMetricReport(overall=summary, per_lead=(summary, summary)),
        batches=1,
        samples=2,
    )


def _metadata(config: ExperimentConfig):
    return create_run_metadata(
        config,
        run_id="run-001",
        created_at=datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
        git_commit="abcdef123",
    )


def test_run_artifact_accepts_early_stopped_history(tmp_path: Path) -> None:
    config = _config(epochs=5)
    history = TrainingHistory(
        train=(_epoch(3.0), _epoch(2.0)),
        validation=(_epoch(4.0), _epoch(2.5)),
        best_epoch=2,
        best_validation_loss=2.5,
        stopped_early=True,
    )

    result = write_run_artifact(
        tmp_path,
        config=config,
        metadata=_metadata(config),
        history=history,
    )

    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["results"]["completed_epochs"] == 2
    assert manifest["results"]["best_epoch"] == 2
    assert manifest["results"]["stopped_early"] is True


def test_run_artifact_supports_named_best_and_last_checkpoints(
    tmp_path: Path,
) -> None:
    config = _config()
    history = TrainingHistory(
        train=(_epoch(3.0), _epoch(2.0)),
        validation=(_epoch(4.0), _epoch(2.5)),
    )
    best = tmp_path / "best-source.pt"
    last = tmp_path / "last-source.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")

    result = write_run_artifact(
        tmp_path / "runs",
        config=config,
        metadata=_metadata(config),
        history=history,
        checkpoints=(
            CheckpointArtifact(best, completed_epochs=2, role="best"),
            CheckpointArtifact(last, completed_epochs=2, role="last"),
        ),
    )

    assert {path.name for path in result.checkpoints} == {"best.pt", "last.pt"}
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    records = manifest["files"]["checkpoints"]
    assert {record["role"] for record in records} == {"best", "last"}
