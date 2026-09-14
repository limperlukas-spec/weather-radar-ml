import json
from pathlib import Path

import pytest

from weather_radar_ml.config.schema import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    RunConfig,
    TrackingConfig,
    TrainingConfig,
)
from weather_radar_ml.training.pipeline import run_experiment


def _config(root: Path, *, epochs: int = 2) -> RunConfig:
    return RunConfig(
        name="pipeline-smoke",
        data=DataConfig(
            name="synthetic",
            train_samples=6,
            validation_samples=4,
            history_steps=2,
            forecast_steps=2,
            dynamic_input_features=("rain",),
            target_features=("rain",),
            height=3,
            width=4,
        ),
        model=ModelConfig(name="persistence"),
        training=TrainingConfig(seed=17, epochs=epochs, batch_size=2),
        tracking=TrackingConfig(
            uri="./mlruns",
            experiment_name="tests",
            enabled=False,
        ),
        output=OutputConfig(runs_root=root),
    )


def test_synthetic_pipeline_persists_complete_run_artifact(tmp_path: Path) -> None:
    result = run_experiment(_config(tmp_path / "runs"))

    assert result.tracking is None
    assert result.artifact.manifest.is_file()
    assert result.artifact.config.is_file()
    assert result.artifact.history.is_file()
    assert result.artifact.checkpoints == ()

    history = json.loads(result.artifact.history.read_text(encoding="utf-8"))
    persisted_config = json.loads(result.artifact.config.read_text(encoding="utf-8"))
    assert len(history["epochs"]) == 2
    assert persisted_config["training"]["device"] == "cpu"
    assert persisted_config["training"]["num_workers"] == 0
    assert history["epochs"][0]["train"]["samples"] == 6
    assert history["epochs"][0]["validation"]["samples"] == 4


def test_synthetic_pipeline_is_reproducible_for_same_config(tmp_path: Path) -> None:
    config = _config(tmp_path / "runs")

    first = run_experiment(config)
    second = run_experiment(config)

    first_config = first.artifact.config.read_text(encoding="utf-8")
    second_config = second.artifact.config.read_text(encoding="utf-8")
    first_history = first.artifact.history.read_text(encoding="utf-8")
    second_history = second.artifact.history.read_text(encoding="utf-8")

    assert first_config == second_config
    assert first_history == second_history


def test_pipeline_rejects_unsupported_model_before_publication(tmp_path: Path) -> None:
    config = _config(tmp_path / "runs")
    config = RunConfig(
        name=config.name,
        data=config.data,
        model=ModelConfig(name="unknown"),
        training=config.training,
        tracking=config.tracking,
        output=config.output,
    )

    with pytest.raises(ValueError, match="Unsupported model strategy"):
        run_experiment(config)

    assert not (tmp_path / "runs").exists()


def test_pipeline_rejects_optimizer_for_parameter_free_persistence(
    tmp_path: Path,
) -> None:
    from weather_radar_ml.config.experiment import ComponentConfig

    base = _config(tmp_path / "runs")
    training = TrainingConfig(
        seed=base.training.seed,
        epochs=1,
        batch_size=2,
        optimizer=ComponentConfig("adam", {"lr": 0.01}),
    )
    config = RunConfig(
        name=base.name,
        data=base.data,
        model=base.model,
        training=training,
        tracking=base.tracking,
        output=base.output,
    )

    with pytest.raises(ValueError, match="trainable parameters"):
        run_experiment(config)


def test_synthetic_pipeline_supports_static_inputs(tmp_path: Path) -> None:
    base = _config(tmp_path / "runs", epochs=1)
    config = RunConfig(
        name=base.name,
        data=DataConfig(
            name="synthetic",
            train_samples=4,
            validation_samples=2,
            history_steps=2,
            forecast_steps=1,
            dynamic_input_features=("rain",),
            static_input_features=("height",),
            target_features=("rain",),
            height=2,
            width=3,
        ),
        model=base.model,
        training=base.training,
        tracking=base.tracking,
        output=base.output,
    )

    result = run_experiment(config)

    history = json.loads(result.artifact.history.read_text(encoding="utf-8"))
    assert history["epochs"][0]["train"]["samples"] == 4
    assert history["epochs"][0]["validation"]["samples"] == 2


def test_synthetic_pipeline_supports_explicit_persistence_mapping(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path / "runs", epochs=1)
    config = RunConfig(
        name=base.name,
        data=DataConfig(
            name="synthetic",
            train_samples=4,
            validation_samples=2,
            history_steps=2,
            forecast_steps=1,
            dynamic_input_features=("rain_input",),
            target_features=("rain_target",),
            height=2,
            width=2,
        ),
        model=ModelConfig(
            name="persistence",
            parameters={"target_input_features": ("rain_input",)},
        ),
        training=base.training,
        tracking=base.tracking,
        output=base.output,
    )

    result = run_experiment(config)

    assert result.artifact.manifest.is_file()


def test_pipeline_rejects_unsupported_data_strategy(tmp_path: Path) -> None:
    base = _config(tmp_path / "runs")
    config = RunConfig(
        name=base.name,
        data=DataConfig(name="unknown"),
        model=base.model,
        training=base.training,
        tracking=base.tracking,
        output=base.output,
    )

    with pytest.raises(ValueError, match="Unsupported data strategy"):
        run_experiment(config)

    assert not (tmp_path / "runs").exists()


def test_pipeline_rejects_unknown_persistence_parameter(tmp_path: Path) -> None:
    base = _config(tmp_path / "runs")
    config = RunConfig(
        name=base.name,
        data=base.data,
        model=ModelConfig(name="persistence", parameters={"unknown": 1}),
        training=base.training,
        tracking=base.tracking,
        output=base.output,
    )

    with pytest.raises(ValueError, match="Unsupported persistence parameters"):
        run_experiment(config)

    assert not (tmp_path / "runs").exists()


def test_pipeline_closes_data_bundle_when_execution_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weather_radar_ml.training.pipeline as pipeline

    closed = False

    def close_bundle(self: object) -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(pipeline._DataBundle, "close", close_bundle)
    base = _config(tmp_path / "runs")
    config = RunConfig(
        name=base.name,
        data=base.data,
        model=ModelConfig(name="unknown"),
        training=base.training,
        tracking=base.tracking,
        output=base.output,
    )

    with pytest.raises(ValueError, match="Unsupported model strategy"):
        run_experiment(config)

    assert closed


def test_tracking_failure_does_not_remove_canonical_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weather_radar_ml.training.pipeline as pipeline

    def fail_tracking(*_: object, **__: object) -> None:
        raise RuntimeError("tracking unavailable")

    monkeypatch.setattr(pipeline, "track_run_artifact", fail_tracking)
    base = _config(tmp_path / "runs", epochs=1)
    config = RunConfig(
        name=base.name,
        data=base.data,
        model=base.model,
        training=base.training,
        tracking=TrackingConfig(
            enabled=True,
            uri="http://127.0.0.1:5000",
            experiment_name="tests",
        ),
        output=base.output,
    )

    with pytest.raises(RuntimeError, match="tracking unavailable"):
        run_experiment(config)

    run_dirs = tuple((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "manifest.json").is_file()
    assert (run_dirs[0] / "config.json").is_file()
    assert (run_dirs[0] / "history.json").is_file()


def test_tracking_result_is_returned_after_local_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weather_radar_ml.training.pipeline as pipeline
    from weather_radar_ml.tracking.mlflow import (
        MlflowTrackingResult,
        MlflowTrackingSettings,
    )

    captured_root: Path | None = None

    def fake_tracking(
        run_root: str | Path,
        *,
        settings: MlflowTrackingSettings,
    ) -> MlflowTrackingResult:
        nonlocal captured_root
        captured_root = Path(run_root)
        assert settings.experiment_name == "tests"
        return MlflowTrackingResult(
            local_run_id=captured_root.name,
            experiment_id="experiment-1",
            mlflow_run_id="mlflow-run-1",
        )

    monkeypatch.setattr(pipeline, "track_run_artifact", fake_tracking)
    base = _config(tmp_path / "runs", epochs=1)
    config = RunConfig(
        name=base.name,
        data=base.data,
        model=base.model,
        training=base.training,
        tracking=TrackingConfig(
            enabled=True,
            uri="http://127.0.0.1:5000",
            experiment_name="tests",
        ),
        output=base.output,
    )

    result = run_experiment(config)

    assert captured_root == result.artifact.root
    assert result.tracking is not None
    assert result.tracking.mlflow_run_id == "mlflow-run-1"
