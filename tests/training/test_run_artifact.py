import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

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


def _config(*, epochs: int = 2, seed: int = 42) -> ExperimentConfig:
    return ExperimentConfig(
        name="artifact test",
        dataset=DatasetReference("dataset-build", "features-v1", "dvc:abc"),
        model_spec=ModelSpec(
            name="toy",
            dynamic_input_features=("rain",),
            target_features=("rain",),
            history_steps=2,
            forecast_steps=2,
        ),
        model=ComponentConfig("toy", {"channels": [8, 16]}),
        loss=ComponentConfig("mse"),
        optimizer=ComponentConfig("adam", {"lr": 0.001}),
        training=TrainingSettings(epochs=epochs, batch_size=4, seed=seed),
    )


def _epoch(loss: float, offset: float = 0.0) -> EpochResult:
    overall = MetricSummary(
        {"mae": 1.0 + offset, "rmse": 2.0 + offset, "bias": -0.5},
        count=24,
    )
    per_lead = (
        MetricSummary({"mae": 0.5, "rmse": 1.0, "bias": -0.25}, count=12),
        MetricSummary({"mae": 1.5, "rmse": 3.0, "bias": -0.75}, count=12),
    )
    return EpochResult(
        loss=loss,
        metrics=ForecastMetricReport(overall=overall, per_lead=per_lead),
        batches=3,
        samples=12,
    )


def _history() -> TrainingHistory:
    return TrainingHistory(
        train=(_epoch(3.0), _epoch(2.0, 0.25)),
        validation=(_epoch(4.0), _epoch(2.5, 0.5)),
    )


def _metadata(config: ExperimentConfig, *, run_id: str = "run-001"):
    return create_run_metadata(
        config,
        run_id=run_id,
        created_at=datetime(2026, 9, 13, 18, 0, tzinfo=UTC),
        git_commit="abcdef123",
    )


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_artifact_persists_config_history_manifest_and_results(
    tmp_path: Path,
) -> None:
    config = _config()
    result = write_run_artifact(
        tmp_path / "runs",
        config=config,
        metadata=_metadata(config),
        history=_history(),
    )

    assert result.root == tmp_path / "runs" / "run-001"
    assert result.config.is_file()
    assert result.history.is_file()
    assert result.manifest.is_file()
    assert result.checkpoints == ()

    manifest = _load_json(result.manifest)
    assert manifest["format_version"] == 1
    assert manifest["config_fingerprint"] == config.fingerprint
    assert manifest["run"]["git_commit"] == "abcdef123"  # type: ignore[index]
    results = manifest["results"]
    assert isinstance(results, dict)
    assert results["completed_epochs"] == 2
    final_validation = results["final_validation"]
    assert isinstance(final_validation, dict)
    assert final_validation["loss"] == 2.5

    history = _load_json(result.history)
    epochs = history["epochs"]
    assert isinstance(epochs, list)
    assert epochs[1]["validation"]["metrics"]["per_lead"][1]["values"]["rmse"] == 3.0


def test_run_artifact_records_content_hashes(tmp_path: Path) -> None:
    config = _config()
    result = write_run_artifact(
        tmp_path,
        config=config,
        metadata=_metadata(config),
        history=_history(),
    )
    manifest = _load_json(result.manifest)
    files = manifest["files"]
    assert isinstance(files, dict)
    config_record = files["config"]
    assert isinstance(config_record, dict)
    expected = hashlib.sha256(result.config.read_bytes()).hexdigest()
    assert config_record["sha256"] == expected


def test_run_artifact_packages_checkpoint_with_epoch_and_hash(tmp_path: Path) -> None:
    config = _config()
    source = tmp_path / "resume.pt"
    source.write_bytes(b"checkpoint-bytes")

    result = write_run_artifact(
        tmp_path / "runs",
        config=config,
        metadata=_metadata(config),
        history=_history(),
        checkpoints=(CheckpointArtifact(source, completed_epochs=2),),
    )

    assert len(result.checkpoints) == 1
    assert result.checkpoints[0].read_bytes() == b"checkpoint-bytes"
    manifest = _load_json(result.manifest)
    records = manifest["files"]["checkpoints"]  # type: ignore[index]
    assert records[0]["completed_epochs"] == 2
    assert records[0]["path"] == "checkpoints/checkpoint-0002.pt"


def test_run_artifact_rejects_mismatching_config_metadata(tmp_path: Path) -> None:
    config = _config(seed=1)
    other = _config(seed=2)

    with pytest.raises(ValueError, match="metadata does not match"):
        write_run_artifact(
            tmp_path,
            config=config,
            metadata=_metadata(other),
            history=_history(),
        )

    assert list(tmp_path.iterdir()) == []


def test_run_artifact_requires_complete_training_history(tmp_path: Path) -> None:
    config = _config(epochs=3)

    with pytest.raises(ValueError, match="configured number of epochs"):
        write_run_artifact(
            tmp_path,
            config=config,
            metadata=_metadata(config),
            history=_history(),
        )


def test_run_artifact_does_not_overwrite_existing_run(tmp_path: Path) -> None:
    config = _config()
    metadata = _metadata(config)
    first = write_run_artifact(
        tmp_path,
        config=config,
        metadata=metadata,
        history=_history(),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        write_run_artifact(
            tmp_path,
            config=config,
            metadata=metadata,
            history=_history(),
        )

    assert first.manifest.is_file()


def test_run_artifact_rejects_unsafe_run_id(tmp_path: Path) -> None:
    config = _config()

    with pytest.raises(ValueError, match="safe path segment"):
        write_run_artifact(
            tmp_path,
            config=config,
            metadata=_metadata(config, run_id="../escape"),
            history=_history(),
        )


def test_run_artifact_validates_checkpoint_epochs_and_source(tmp_path: Path) -> None:
    config = _config()
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"valid")

    with pytest.raises(ValueError, match="cannot exceed"):
        write_run_artifact(
            tmp_path / "runs-a",
            config=config,
            metadata=_metadata(config),
            history=_history(),
            checkpoints=(CheckpointArtifact(source, completed_epochs=3),),
        )

    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        write_run_artifact(
            tmp_path / "runs-b",
            config=config,
            metadata=_metadata(config),
            history=_history(),
            checkpoints=(CheckpointArtifact(missing, completed_epochs=1),),
        )
