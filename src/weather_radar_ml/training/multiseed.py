"""Three-seed experiment orchestration and aggregate research reporting."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from math import isfinite
from pathlib import Path
from statistics import mean, stdev
from types import MappingProxyType
from uuid import uuid4

from weather_radar_ml.config.schema import RunConfig
from weather_radar_ml.evaluation.research import (
    ConditionComparison,
    ResearchEvaluationReport,
    ResearchForecastReport,
    ResearchMetricSummary,
    ResearchSkillReport,
)
from weather_radar_ml.tracking.mlflow import (
    MlflowParentRunResult,
    MlflowTrackingSettings,
    finish_mlflow_parent_run,
    start_mlflow_parent_run,
)
from weather_radar_ml.training.pipeline import PipelineResult, run_experiment

OFFICIAL_MULTI_SEEDS = (17, 42, 73)
MULTI_SEED_ARTIFACT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class AggregateStatistic:
    """Mean/sample-standard-deviation with explicit defined-run support."""

    mean: float | None
    std: float | None
    defined_runs: int

    def __post_init__(self) -> None:
        if self.defined_runs < 0:
            raise ValueError("defined_runs must not be negative.")
        for name, value in (("mean", self.mean), ("std", self.std)):
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be None or finite.")
        if self.defined_runs == 0 and (self.mean is not None or self.std is not None):
            raise ValueError("Undefined aggregates must not contain numeric values.")
        if self.defined_runs == 1 and self.std is not None:
            raise ValueError("Sample standard deviation needs at least two values.")


@dataclass(frozen=True, slots=True)
class SeedRun:
    """One completed child run in a multi-seed experiment."""

    seed: int
    result: PipelineResult

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        if self.result.validation_research is None:
            raise ValueError(
                "Multi-seed learned runs require research evaluation results."
            )


@dataclass(frozen=True, slots=True)
class MultiSeedArtifact:
    """Canonical local summary artifact for a multi-seed experiment."""

    root: Path
    summary: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class MultiSeedResult:
    """Completed child runs, selected reference seed, and aggregate artifact."""

    runs: tuple[SeedRun, ...]
    reference_seed: int
    artifact: MultiSeedArtifact
    aggregates: Mapping[str, AggregateStatistic]
    tracking_parent: MlflowParentRunResult | None = None

    def __post_init__(self) -> None:
        if len(self.runs) != 3:
            raise ValueError("MultiSeedResult requires exactly three child runs.")
        seeds = tuple(run.seed for run in self.runs)
        if self.reference_seed not in seeds:
            raise ValueError("reference_seed must identify one child run.")
        object.__setattr__(
            self,
            "aggregates",
            MappingProxyType(dict(sorted(self.aggregates.items()))),
        )


def run_multiseed_experiment(
    config: RunConfig,
    *,
    seeds: tuple[int, int, int] = OFFICIAL_MULTI_SEEDS,
) -> MultiSeedResult:
    """Run exactly three fixed-seed learned experiments and aggregate the results."""
    normalized_seeds = _validate_seeds(seeds)
    if config.model.name != "unet":
        raise ValueError("Multi-seed 0.6 benchmarking currently requires model='unet'.")

    tracking_settings = MlflowTrackingSettings(
        tracking_uri=config.tracking.uri,
        experiment_name=config.tracking.experiment_name,
        artifact_path=config.tracking.artifact_path,
    )
    parent: MlflowParentRunResult | None = None
    if config.tracking.enabled:
        parent = start_mlflow_parent_run(
            run_name=f"{config.name}-multi-seed",
            seeds=normalized_seeds,
            settings=tracking_settings,
        )

    completed: list[SeedRun] = []
    try:
        for seed in normalized_seeds:
            child_config = replace(
                config,
                training=replace(config.training, seed=seed),
            )
            result = run_experiment(
                child_config,
                tracking_parent_run_id=(
                    None if parent is None else parent.mlflow_run_id
                ),
            )
            completed.append(SeedRun(seed=seed, result=result))

        runs = tuple(completed)
        reference_seed = _reference_seed(runs)
        aggregates = _aggregate_runs(runs)
        artifact = _write_multiseed_artifact(
            config.output.runs_root,
            runs=runs,
            seeds=normalized_seeds,
            reference_seed=reference_seed,
            aggregates=aggregates,
        )

        if parent is not None:
            finish_mlflow_parent_run(
                parent.mlflow_run_id,
                status="FINISHED",
                settings=tracking_settings,
                summary_path=artifact.summary,
                metrics=_parent_metrics(aggregates),
            )

        return MultiSeedResult(
            runs=runs,
            reference_seed=reference_seed,
            artifact=artifact,
            aggregates=aggregates,
            tracking_parent=parent,
        )
    except Exception:
        if parent is not None:
            with suppress(Exception):
                finish_mlflow_parent_run(
                    parent.mlflow_run_id,
                    status="FAILED",
                    settings=tracking_settings,
                )
        raise


def _validate_seeds(seeds: tuple[int, ...]) -> tuple[int, int, int]:
    if len(seeds) != 3:
        raise ValueError("Multi-seed benchmarking requires exactly three seeds.")
    normalized = tuple(_validate_seed(seed) for seed in seeds)
    if len(set(normalized)) != 3:
        raise ValueError("Multi-seed benchmarking requires three unique seeds.")
    return normalized[0], normalized[1], normalized[2]


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1.")
    return seed


def _reference_seed(runs: tuple[SeedRun, ...]) -> int:
    if len(runs) != 3:
        raise ValueError("Reference selection requires exactly three runs.")
    ordered = sorted(
        runs,
        key=lambda run: (run.result.history.best_validation_loss, run.seed),
    )
    return ordered[1].seed


def _aggregate_runs(runs: tuple[SeedRun, ...]) -> dict[str, AggregateStatistic]:
    values: dict[str, list[float | None]] = {
        "best_validation_loss": [
            run.result.history.best_validation_loss for run in runs
        ]
    }
    flattened = [_flatten_research(_require_research(run)) for run in runs]
    key_sets = [set(payload) for payload in flattened]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("Research metric structure differs between seed runs.")
    keys = sorted(key_sets[0])
    for key in keys:
        values[f"validation_research.{key}"] = [
            payload.get(key) for payload in flattened
        ]
    return {key: _aggregate(items) for key, items in values.items()}


def _require_research(run: SeedRun) -> ResearchEvaluationReport:
    report = run.result.validation_research
    if report is None:
        raise ValueError("Multi-seed learned runs require research evaluation results.")
    return report


def _aggregate(values: list[float | None]) -> AggregateStatistic:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return AggregateStatistic(mean=None, std=None, defined_runs=0)
    return AggregateStatistic(
        mean=mean(defined),
        std=stdev(defined) if len(defined) >= 2 else None,
        defined_runs=len(defined),
    )


def _flatten_research(report: ResearchEvaluationReport) -> dict[str, float | None]:
    payload: dict[str, float | None] = {}
    _flatten_condition(payload, "all", report.all)
    if report.wet_target is not None:
        _flatten_condition(payload, "wet_target", report.wet_target)
    return payload


def _flatten_condition(
    payload: dict[str, float | None],
    prefix: str,
    condition: ConditionComparison,
) -> None:
    _flatten_forecast(payload, f"{prefix}.learned", condition.learned)
    _flatten_forecast(payload, f"{prefix}.persistence", condition.persistence)
    _flatten_skill(payload, f"{prefix}.skill", condition.skill)


def _flatten_forecast(
    payload: dict[str, float | None],
    prefix: str,
    report: ResearchForecastReport,
) -> None:
    _flatten_metric_summary(payload, f"{prefix}.overall", report.overall)
    for lead, summary in zip(report.lead_minutes, report.per_lead, strict=True):
        _flatten_metric_summary(payload, f"{prefix}.lead_{lead:03d}", summary)


def _flatten_metric_summary(
    payload: dict[str, float | None],
    prefix: str,
    summary: ResearchMetricSummary,
) -> None:
    payload[f"{prefix}.mae"] = summary.mae
    payload[f"{prefix}.rmse"] = summary.rmse
    for score in summary.categorical:
        threshold = f"{score.threshold_mm_h:g}"
        score_prefix = f"{prefix}.threshold_{threshold}"
        payload[f"{score_prefix}.csi"] = score.csi
        payload[f"{score_prefix}.precision"] = score.precision
        payload[f"{score_prefix}.recall"] = score.recall


def _flatten_skill(
    payload: dict[str, float | None],
    prefix: str,
    report: ResearchSkillReport,
) -> None:
    payload[f"{prefix}.overall.mae"] = report.overall.mae
    payload[f"{prefix}.overall.rmse"] = report.overall.rmse
    for lead, summary in zip(report.lead_minutes, report.per_lead, strict=True):
        payload[f"{prefix}.lead_{lead:03d}.mae"] = summary.mae
        payload[f"{prefix}.lead_{lead:03d}.rmse"] = summary.rmse


def _write_multiseed_artifact(
    runs_root: str | Path,
    *,
    runs: tuple[SeedRun, ...],
    seeds: tuple[int, int, int],
    reference_seed: int,
    aggregates: Mapping[str, AggregateStatistic],
) -> MultiSeedArtifact:
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    group_id = f"multiseed-{uuid4().hex}"
    destination = root / group_id
    staging = Path(tempfile.mkdtemp(dir=root, prefix=f".{group_id}.tmp-"))
    try:
        summary_path = staging / "summary.json"
        _write_json(
            summary_path,
            _summary_payload(
                group_id=group_id,
                runs=runs,
                seeds=seeds,
                reference_seed=reference_seed,
                aggregates=aggregates,
            ),
        )
        manifest_path = staging / "manifest.json"
        _write_json(
            manifest_path,
            {
                "format_version": MULTI_SEED_ARTIFACT_FORMAT_VERSION,
                "group_id": group_id,
                "summary": _file_record(summary_path, staging),
            },
        )
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return MultiSeedArtifact(
        root=destination,
        summary=destination / "summary.json",
        manifest=destination / "manifest.json",
    )


def _summary_payload(
    *,
    group_id: str,
    runs: tuple[SeedRun, ...],
    seeds: tuple[int, int, int],
    reference_seed: int,
    aggregates: Mapping[str, AggregateStatistic],
) -> dict[str, object]:
    reference = next(run for run in runs if run.seed == reference_seed)
    return {
        "format_version": MULTI_SEED_ARTIFACT_FORMAT_VERSION,
        "group_id": group_id,
        "seeds": list(seeds),
        "reference_seed": reference_seed,
        "reference_local_run_id": reference.result.artifact.root.name,
        "runs": [
            {
                "seed": run.seed,
                "local_run_id": run.result.artifact.root.name,
                "best_epoch": run.result.history.best_epoch,
                "best_validation_loss": run.result.history.best_validation_loss,
                "validation_research": asdict(_require_research(run)),
            }
            for run in runs
        ],
        "aggregates": {key: asdict(value) for key, value in sorted(aggregates.items())},
    }


def _parent_metrics(
    aggregates: Mapping[str, AggregateStatistic],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, aggregate in aggregates.items():
        if aggregate.mean is not None:
            metrics[f"multi_seed/{key}/mean"] = aggregate.mean
        if aggregate.std is not None:
            metrics[f"multi_seed/{key}/std"] = aggregate.std
    return metrics


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
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
