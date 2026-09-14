import json
from pathlib import Path

import pytest

from weather_radar_ml.config.experiment import ComponentConfig
from weather_radar_ml.config.schema import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    RunConfig,
    TrackingConfig,
    TrainingConfig,
)
from weather_radar_ml.evaluation.domain import ForecastMetricReport, MetricSummary
from weather_radar_ml.evaluation.research import (
    CategoricalScore,
    ConditionComparison,
    ErrorSkillSummary,
    ResearchEvaluationReport,
    ResearchForecastReport,
    ResearchMetricSummary,
    ResearchSkillReport,
)
from weather_radar_ml.training.artifact import RunArtifactResult
from weather_radar_ml.training.domain import EpochResult, TrainingHistory
from weather_radar_ml.training.forecast_artifact import ForecastArtifactResult
from weather_radar_ml.training.multiseed import (
    OFFICIAL_MULTI_SEEDS,
    run_multiseed_experiment,
)
from weather_radar_ml.training.pipeline import PipelineResult


def _config(root: Path) -> RunConfig:
    return RunConfig(
        name="multi-seed-test",
        data=DataConfig(
            name="synthetic",
            train_samples=2,
            validation_samples=2,
            history_steps=2,
            forecast_steps=1,
            dynamic_input_features=("rain",),
            target_features=("rain",),
            height=8,
            width=8,
        ),
        model=ModelConfig(name="unet", parameters={"base_channels": 2}),
        training=TrainingConfig(
            seed=999,
            epochs=1,
            batch_size=2,
            optimizer=ComponentConfig("adam", {"lr": 0.01}),
        ),
        tracking=TrackingConfig(
            uri="./mlruns",
            experiment_name="tests",
            enabled=False,
        ),
        output=OutputConfig(runs_root=root),
    )


def _epoch(loss: float) -> EpochResult:
    summary = MetricSummary({"mae": loss, "rmse": loss, "bias": 0.0}, count=1)
    return EpochResult(
        loss=loss,
        metrics=ForecastMetricReport(overall=summary, per_lead=(summary,)),
        batches=1,
        samples=1,
    )


def _research(mae: float) -> ResearchEvaluationReport:
    categorical = (
        CategoricalScore(
            threshold_mm_h=1.0,
            hits=0,
            false_alarms=0,
            misses=0,
            csi=None,
            precision=None,
            recall=None,
        ),
    )
    learned_summary = ResearchMetricSummary(
        count=1,
        mae=mae,
        rmse=mae,
        categorical=categorical,
    )
    persistence_summary = ResearchMetricSummary(
        count=1,
        mae=4.0,
        rmse=4.0,
        categorical=categorical,
    )
    learned = ResearchForecastReport(
        overall=learned_summary,
        per_lead=(learned_summary,),
        lead_minutes=(5,),
    )
    persistence = ResearchForecastReport(
        overall=persistence_summary,
        per_lead=(persistence_summary,),
        lead_minutes=(5,),
    )
    skill_summary = ErrorSkillSummary(mae=1.0 - mae / 4.0, rmse=1.0 - mae / 4.0)
    skill = ResearchSkillReport(
        overall=skill_summary,
        per_lead=(skill_summary,),
        lead_minutes=(5,),
    )
    comparison = ConditionComparison(
        learned=learned,
        persistence=persistence,
        skill=skill,
    )
    return ResearchEvaluationReport(
        all=comparison,
        wet_target=None,
        thresholds_mm_h=(1.0,),
        wet_threshold_mm_h=0.1,
    )


def test_multiseed_runs_fixed_seeds_selects_median_and_writes_aggregates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weather_radar_ml.training.multiseed as multiseed

    losses = {17: 3.0, 42: 1.0, 73: 2.0}
    maes = {17: 1.0, 42: 2.0, 73: 3.0}
    seen: list[int] = []

    def fake_run(
        config: RunConfig,
        *,
        tracking_parent_run_id: str | None = None,
    ) -> PipelineResult:
        assert tracking_parent_run_id is None
        seed = config.training.seed
        seen.append(seed)
        loss = losses[seed]
        history = TrainingHistory(
            train=(_epoch(loss),),
            validation=(_epoch(loss),),
        )
        root = tmp_path / f"run-{seed}"
        artifact = RunArtifactResult(
            root=root,
            manifest=root / "manifest.json",
            config=root / "config.json",
            history=root / "history.json",
            checkpoints=(),
        )
        return PipelineResult(
            artifact=artifact,
            history=history,
            validation_research=_research(maes[seed]),
        )

    forecast_seeds: list[int] = []

    def fake_forecast(
        config: RunConfig,
        result: PipelineResult,
        *,
        qualitative_count: int = 8,
    ) -> ForecastArtifactResult:
        assert qualitative_count == 8
        forecast_seeds.append(config.training.seed)
        root = tmp_path / "forecast-reference"
        return ForecastArtifactResult(
            root=root,
            predictions=root / "predictions.zarr",
            metadata=root / "metadata.json",
            qualitative_samples=root / "qualitative-samples.json",
            manifest=root / "manifest.json",
            fingerprint="forecast-fingerprint",
            sample_count=2,
            split="validation",
        )

    monkeypatch.setattr(multiseed, "run_experiment", fake_run)
    monkeypatch.setattr(multiseed, "write_reference_forecast_artifact", fake_forecast)

    result = run_multiseed_experiment(_config(tmp_path / "runs"))

    assert seen == list(OFFICIAL_MULTI_SEEDS)
    assert result.reference_seed == 73
    assert forecast_seeds == [73]
    assert result.forecast_artifact.fingerprint == "forecast-fingerprint"
    best_loss = result.aggregates["best_validation_loss"]
    assert best_loss.mean == pytest.approx(2.0)
    assert best_loss.std == pytest.approx(1.0)
    assert best_loss.defined_runs == 3

    research = result.aggregates["validation_research.all.learned.overall.mae"]
    assert research.mean == pytest.approx(2.0)
    assert research.std == pytest.approx(1.0)
    undefined = result.aggregates[
        "validation_research.all.learned.overall.threshold_1.csi"
    ]
    assert undefined.mean is None
    assert undefined.std is None
    assert undefined.defined_runs == 0

    payload = json.loads(result.artifact.summary.read_text(encoding="utf-8"))
    assert payload["seeds"] == [17, 42, 73]
    assert payload["reference_seed"] == 73
    assert payload["reference_local_run_id"] == "run-73"
    assert payload["reference_forecast_artifact"] == {
        "artifact_id": "forecast-reference",
        "artifact_fingerprint": "forecast-fingerprint",
        "sample_count": 2,
        "split": "validation",
    }
    assert result.artifact.manifest.is_file()


@pytest.mark.parametrize(
    "seeds",
    [
        (1, 2),
        (1, 1, 2),
        (1, 2, 2**32),
    ],
)
def test_multiseed_rejects_invalid_seed_sets(
    tmp_path: Path,
    seeds: tuple[int, ...],
) -> None:
    with pytest.raises((ValueError, TypeError)):
        run_multiseed_experiment(
            _config(tmp_path / "runs"),
            seeds=seeds,  # type: ignore[arg-type]
        )


def test_multiseed_rejects_non_unet_model(tmp_path: Path) -> None:
    base = _config(tmp_path / "runs")
    config = RunConfig(
        name=base.name,
        data=base.data,
        model=ModelConfig(name="persistence"),
        training=base.training,
        tracking=base.tracking,
        output=base.output,
    )

    with pytest.raises(ValueError, match="model='unet'"):
        run_multiseed_experiment(config)
