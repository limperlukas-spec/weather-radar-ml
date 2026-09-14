"""End-to-end runtime assembly for configured forecast experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.optim import SGD, Adam, Optimizer
from torch.utils.data import DataLoader, Dataset

from weather_radar_ml.config.experiment import (
    ComponentConfig,
    DatasetReference,
    ExperimentConfig,
    TrainingSettings,
)
from weather_radar_ml.config.schema import DataConfig, RunConfig
from weather_radar_ml.data.ml.catalog import SampleRecord, SQLiteSampleCatalog
from weather_radar_ml.data.ml.dataset import LazyRadarDataset
from weather_radar_ml.data.ml.domain import SampleSelection
from weather_radar_ml.data.ml.torch_adapter import TorchRadarDataset, TorchSample
from weather_radar_ml.evaluation.continuous import ContinuousForecastMetrics
from weather_radar_ml.evaluation.research import (
    ResearchEvaluationReport,
    ResearchForecastEvaluator,
)
from weather_radar_ml.models.baseline import PersistenceForecast
from weather_radar_ml.models.contracts import ForecastModel
from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.models.unet import UNetForecast
from weather_radar_ml.tracking.mlflow import (
    MlflowTrackingResult,
    MlflowTrackingSettings,
    track_run_artifact,
)
from weather_radar_ml.training.artifact import (
    CheckpointArtifact,
    RunArtifactResult,
    write_run_artifact,
)
from weather_radar_ml.training.batching import make_forecast_dataloader
from weather_radar_ml.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from weather_radar_ml.training.domain import ForecastBatch, TrainingHistory
from weather_radar_ml.training.engine import ForecastTrainer
from weather_radar_ml.training.forecast_artifact import (
    ForecastArtifactBatch,
    ForecastArtifactResult,
    write_forecast_artifact,
)
from weather_radar_ml.training.losses import MeanSquaredForecastLoss
from weather_radar_ml.training.reproducibility import configure_reproducibility
from weather_radar_ml.training.run import create_run_metadata
from weather_radar_ml.transforms.precipitation import (
    inverse_log1p_precipitation,
    log1p_precipitation,
)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Published local artifact and optional MLflow identity for one run."""

    artifact: RunArtifactResult
    history: TrainingHistory
    validation_research: ResearchEvaluationReport | None = None
    tracking: MlflowTrackingResult | None = None


@dataclass(slots=True)
class _DataBundle:
    train: Dataset[TorchSample]
    validation: Dataset[TorchSample]
    reference: DatasetReference
    spec: ModelSpec
    closers: tuple[Callable[[], None], ...] = ()

    def close(self) -> None:
        for closer in self.closers:
            closer()


class _SyntheticForecastDataset(Dataset[TorchSample]):
    def __init__(self, config: DataConfig, *, samples: int, offset: int) -> None:
        self._config = config
        self._samples = samples
        self._offset = offset

    def __len__(self) -> int:
        return self._samples

    def __getitem__(self, index: int) -> TorchSample:
        if index < 0 or index >= self._samples:
            raise IndexError(index)
        config = self._config
        base = float(index + self._offset)
        dynamic = torch.empty(
            (
                config.history_steps,
                len(config.dynamic_input_features),
                config.height,
                config.width,
            ),
            dtype=torch.float32,
        )
        for time_index in range(config.history_steps):
            for channel_index in range(len(config.dynamic_input_features)):
                dynamic[time_index, channel_index].fill_(
                    base + float(time_index) + channel_index / 10.0
                )

        target = torch.empty(
            (
                config.forecast_steps,
                len(config.target_features),
                config.height,
                config.width,
            ),
            dtype=torch.float32,
        )
        for target_index, feature in enumerate(config.target_features):
            if feature not in config.dynamic_input_features:
                target[:, target_index].fill_(base + target_index)
                continue
            source_index = config.dynamic_input_features.index(feature)
            latest = dynamic[-1, source_index]
            for lead_index in range(config.forecast_steps):
                target[lead_index, target_index].copy_(
                    latest + 0.25 * float(lead_index + 1)
                )

        item: TorchSample = {
            "dynamic_inputs": dynamic,
            "targets": target,
            "sample_id": f"synthetic-{self._offset + index:08d}",
        }
        if config.static_input_features:
            item["static_inputs"] = torch.full(
                (
                    len(config.static_input_features),
                    config.height,
                    config.width,
                ),
                base,
                dtype=torch.float32,
            )
        return item


def run_experiment(
    config: RunConfig,
    *,
    tracking_parent_run_id: str | None = None,
) -> PipelineResult:
    """Execute one configured experiment and publish its canonical run artifact."""
    data = _build_data_bundle(config)
    try:
        experiment = _experiment_config(config, data)
        configure_reproducibility(experiment.training)
        model = _build_model(config, data.spec)
        loss = _build_loss(config, data.spec)
        optimizer = _build_optimizer(config, model)
        input_transform, target_transform, metric_prediction_transform = (
            _training_transforms(config)
        )
        trainer = ForecastTrainer(
            model=model,
            loss=loss,
            optimizer=optimizer,
            metric_factory=lambda: ContinuousForecastMetrics(data.spec),
            device=config.training.device,
            input_transform=input_transform,
            target_transform=target_transform,
            metric_prediction_transform=metric_prediction_transform,
        )
        train_loader = make_forecast_dataloader(
            data.train,
            batch_size=config.training.batch_size,
            shuffle=True,
            seed=config.training.seed,
            num_workers=config.training.num_workers,
        )
        validation_loader = make_forecast_dataloader(
            data.validation,
            batch_size=config.training.batch_size,
            shuffle=False,
            seed=config.training.seed,
            num_workers=config.training.num_workers,
        )
        metadata = create_run_metadata(
            experiment,
            git_commit=_git_commit(),
        )

        with tempfile.TemporaryDirectory(prefix="weather-radar-ml-checkpoint-") as tmp:
            temporary_root = Path(tmp)
            best_checkpoint = temporary_root / "best.pt"

            def save_best(epoch: int, _: object) -> None:
                if optimizer is None:
                    return
                save_training_checkpoint(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    completed_epochs=epoch,
                )

            history = trainer.fit(
                train_loader,
                validation_loader,
                epochs=config.training.epochs,
                early_stopping_patience=config.training.early_stopping_patience,
                on_best_validation=save_best if optimizer is not None else None,
            )

            checkpoints: tuple[CheckpointArtifact, ...] = ()
            if optimizer is not None:
                last_checkpoint = temporary_root / "last.pt"
                completed_epochs = len(history.train)
                save_training_checkpoint(
                    last_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    completed_epochs=completed_epochs,
                )
                checkpoints = (
                    CheckpointArtifact(
                        best_checkpoint,
                        completed_epochs=history.best_epoch,
                        role="best",
                    ),
                    CheckpointArtifact(
                        last_checkpoint,
                        completed_epochs=completed_epochs,
                        role="last",
                    ),
                )

            validation_research: ResearchEvaluationReport | None = None
            if config.model.name == "unet":
                if optimizer is None:
                    raise RuntimeError("The learned U-Net requires an optimizer.")
                load_training_checkpoint(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    restore_rng_state=False,
                )
                validation_research = _evaluate_research(
                    config,
                    spec=data.spec,
                    model=model,
                    batches=validation_loader,
                )

            artifact = write_run_artifact(
                config.output.runs_root,
                config=experiment,
                metadata=metadata,
                history=history,
                checkpoints=checkpoints,
            )

        tracking: MlflowTrackingResult | None = None
        if config.tracking.enabled:
            tracking = track_run_artifact(
                artifact.root,
                settings=MlflowTrackingSettings(
                    tracking_uri=config.tracking.uri,
                    experiment_name=config.tracking.experiment_name,
                    artifact_path=config.tracking.artifact_path,
                    parent_run_id=tracking_parent_run_id,
                ),
            )
        return PipelineResult(
            artifact=artifact,
            history=history,
            validation_research=validation_research,
            tracking=tracking,
        )
    finally:
        data.close()


def evaluate_reference_checkpoint(
    config: RunConfig,
    result: PipelineResult,
    *,
    split: str,
) -> ResearchEvaluationReport:
    """Evaluate one selected best checkpoint on validation or held-out test data."""
    normalized_split = _evaluation_split(split)
    if config.model.name != "unet":
        raise ValueError(
            "Reference research evaluation currently requires model='unet'."
        )
    if result.validation_research is None:
        raise ValueError("Reference evaluation requires validation research results.")

    best_checkpoint = _checkpoint_by_role(result.artifact, "best")
    data = _build_data_bundle(config)
    extra_closer: Callable[[], None] | None = None
    try:
        dataset, extra_closer = _evaluation_dataset(
            config,
            data=data,
            split=normalized_split,
        )
        model = _build_model(config, data.spec)
        optimizer = _build_optimizer(config, model)
        if optimizer is None:
            raise RuntimeError("The learned U-Net requires an optimizer.")
        load_training_checkpoint(
            best_checkpoint,
            model=model,
            optimizer=optimizer,
            restore_rng_state=False,
        )
        loader = make_forecast_dataloader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            seed=config.training.seed,
            num_workers=config.training.num_workers,
        )
        return _evaluate_research(
            config,
            spec=data.spec,
            model=model,
            batches=loader,
        )
    finally:
        if extra_closer is not None:
            extra_closer()
        data.close()


def write_reference_forecast_artifact(
    config: RunConfig,
    result: PipelineResult,
    *,
    split: str = "validation",
    research: ResearchEvaluationReport | None = None,
    qualitative_count: int = 8,
) -> ForecastArtifactResult:
    """Persist physical forecasts from one learned run's selected best checkpoint."""
    normalized_split = _evaluation_split(split)
    if config.model.name != "unet":
        raise ValueError("Reference forecast artifacts currently require model='unet'.")
    if result.validation_research is None:
        raise ValueError(
            "Reference forecast artifacts require validation research results."
        )

    evaluation_research = research
    if evaluation_research is None:
        if normalized_split == "validation":
            evaluation_research = result.validation_research
        else:
            evaluation_research = evaluate_reference_checkpoint(
                config,
                result,
                split=normalized_split,
            )
    if evaluation_research is None:
        raise RuntimeError("Reference evaluation report is unexpectedly unavailable.")

    best_checkpoint = _checkpoint_by_role(result.artifact, "best")
    data = _build_data_bundle(config)
    extra_closer: Callable[[], None] | None = None
    try:
        dataset, extra_closer = _evaluation_dataset(
            config,
            data=data,
            split=normalized_split,
        )
        model = _build_model(config, data.spec)
        optimizer = _build_optimizer(config, model)
        if optimizer is None:
            raise RuntimeError("The learned U-Net requires an optimizer.")
        load_training_checkpoint(
            best_checkpoint,
            model=model,
            optimizer=optimizer,
            restore_rng_state=False,
        )

        persistence = PersistenceForecast(
            data.spec,
            target_input_features=_infer_persistence_mapping(config.data),
        )
        device = torch.device(config.training.device)
        model.to(device)
        persistence.to(device)
        model.eval()
        persistence.eval()

        loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        )
        artifact_id = f"forecast-{result.artifact.root.name}-{normalized_split}"
        return write_forecast_artifact(
            config.output.runs_root,
            artifact_id=artifact_id,
            spec=data.spec,
            lead_minutes=tuple(
                _forecast_step_minutes(config) * (index + 1)
                for index in range(data.spec.forecast_steps)
            ),
            batches=_forecast_artifact_batches(
                model=cast(ForecastModel, model),
                persistence=persistence,
                loader=loader,
                device=device,
            ),
            split=normalized_split,
            provenance=_forecast_provenance(
                config,
                result,
                best_checkpoint=best_checkpoint,
                evaluation_split=normalized_split,
                evaluation_research=evaluation_research,
            ),
            qualitative_count=qualitative_count,
        )
    finally:
        if extra_closer is not None:
            extra_closer()
        data.close()


def _forecast_artifact_batches(
    *,
    model: ForecastModel,
    persistence: PersistenceForecast,
    loader: Iterable[object],
    device: torch.device,
) -> Iterable[ForecastArtifactBatch]:
    with torch.no_grad():
        for raw_batch in loader:
            batch = _require_mapping(raw_batch, "forecast artifact batch")
            dynamic = _require_tensor(batch.get("dynamic_inputs"), "dynamic_inputs").to(
                device
            )
            target = _require_tensor(batch.get("targets"), "targets").to(device)
            static_raw = batch.get("static_inputs")
            static = (
                None
                if static_raw is None
                else _require_tensor(static_raw, "static_inputs").to(device)
            )
            sample_ids = _require_sample_ids(batch.get("sample_id"))

            prediction_log1p = model(log1p_precipitation(dynamic), static)
            learned = inverse_log1p_precipitation(prediction_log1p).clamp_min(0.0)
            baseline = persistence(dynamic, static)
            valid = torch.isfinite(target)
            if bool((valid & (target < 0.0)).any()):
                raise ValueError(
                    "Target precipitation must not be negative when valid."
                )
            for name, values in (
                ("learned forecast", learned),
                ("persistence forecast", baseline),
            ):
                if bool((valid & ~torch.isfinite(values)).any()):
                    raise ValueError(f"{name} is non-finite on a valid target value.")
                if bool((valid & (values < 0.0)).any()):
                    raise ValueError(f"{name} must not be negative on valid targets.")

            yield ForecastArtifactBatch(
                sample_ids=sample_ids,
                learned_mm_h=learned.detach().cpu().numpy(),
                persistence_mm_h=baseline.detach().cpu().numpy(),
                target_mm_h=target.detach().cpu().numpy(),
                valid_mask=valid.detach().cpu().numpy(),
            )


def _forecast_provenance(
    config: RunConfig,
    result: PipelineResult,
    *,
    best_checkpoint: Path,
    evaluation_split: str,
    evaluation_research: ResearchEvaluationReport,
) -> dict[str, object]:
    manifest = _read_json_mapping(result.artifact.manifest)
    persisted_config = _read_json_mapping(result.artifact.config)
    run_payload = _require_mapping(manifest.get("run"), "run manifest metadata")
    fingerprint = manifest.get("config_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ValueError("Run manifest has no valid config_fingerprint.")

    validation_research = result.validation_research
    if validation_research is None:
        raise ValueError("Forecast provenance requires validation research results.")

    return {
        "source_local_run_id": result.artifact.root.name,
        "experiment_fingerprint": fingerprint,
        "dataset": persisted_config.get("dataset"),
        "model": persisted_config.get("model"),
        "loss": persisted_config.get("loss"),
        "optimizer": persisted_config.get("optimizer"),
        "training": persisted_config.get("training"),
        "seed": config.training.seed,
        "best_epoch": result.history.best_epoch,
        "best_validation_loss": result.history.best_validation_loss,
        "stopped_early": result.history.stopped_early,
        "run_environment": dict(run_payload),
        "device": config.training.device,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "checkpoint": {
            "role": "best",
            "path": best_checkpoint.relative_to(result.artifact.root).as_posix(),
            "sha256": _sha256(best_checkpoint),
        },
        "selection_policy": "best checkpoint selected by validation loss only",
        "validation_research": asdict(validation_research),
        "evaluation_split": evaluation_split,
        "evaluation_research": asdict(evaluation_research),
    }


def _checkpoint_by_role(artifact: RunArtifactResult, role: str) -> Path:
    expected = f"{role}.pt"
    matches = tuple(path for path in artifact.checkpoints if path.name == expected)
    if len(matches) != 1:
        raise ValueError(
            f"Run artifact must contain exactly one {expected!r} checkpoint."
        )
    return matches[0]


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    return _require_mapping(payload, str(path))


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping.")
    return cast(Mapping[str, object], value)


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor.")
    return value


def _require_sample_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError("sample_id batch must be a non-empty sequence of strings.")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise TypeError("sample_id batch must contain only non-empty strings.")
    return tuple(item.strip() for item in value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_research(
    config: RunConfig,
    *,
    spec: ModelSpec,
    model: nn.Module,
    batches: Iterable[ForecastBatch],
) -> ResearchEvaluationReport:
    forecast_model = cast(ForecastModel, model)
    evaluator = ResearchForecastEvaluator(
        spec,
        lead_step_minutes=_forecast_step_minutes(config),
        persistence_input_features=_infer_persistence_mapping(config.data),
    )
    device = torch.device(config.training.device)
    model.eval()
    with torch.no_grad():
        for raw_batch in batches:
            batch = raw_batch.to(device)
            prediction = forecast_model(
                log1p_precipitation(batch.dynamic),
                batch.static,
            )
            evaluator.update(
                dynamic_inputs=batch.dynamic,
                learned_prediction_log1p=prediction,
                target=batch.target,
            )
    return evaluator.compute()


def _forecast_step_minutes(config: RunConfig) -> int:
    if config.data.dataset is None:
        return 5
    return config.data.dataset.temporal.step_minutes


def _experiment_config(config: RunConfig, data: _DataBundle) -> ExperimentConfig:
    return ExperimentConfig(
        name=config.name,
        dataset=data.reference,
        model_spec=data.spec,
        model=ComponentConfig(config.model.name, config.model.parameters),
        loss=config.training.loss,
        optimizer=config.training.optimizer,
        training=TrainingSettings(
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            seed=config.training.seed,
            deterministic_algorithms=config.training.deterministic_algorithms,
            device=config.training.device,
            num_workers=config.training.num_workers,
            early_stopping_patience=config.training.early_stopping_patience,
        ),
    )


def _build_data_bundle(config: RunConfig) -> _DataBundle:
    if config.data.name == "synthetic":
        return _synthetic_bundle(config)
    if config.data.name == "ml_dataset":
        return _ml_dataset_bundle(config)
    raise ValueError(f"Unsupported data strategy: {config.data.name!r}.")


def _synthetic_bundle(config: RunConfig) -> _DataBundle:
    data = config.data
    spec = ModelSpec(
        name=config.model.name,
        dynamic_input_features=data.dynamic_input_features,
        static_input_features=data.static_input_features,
        target_features=data.target_features,
        history_steps=data.history_steps,
        forecast_steps=data.forecast_steps,
    )
    payload = {
        "train_samples": data.train_samples,
        "validation_samples": data.validation_samples,
        "history_steps": data.history_steps,
        "forecast_steps": data.forecast_steps,
        "dynamic_input_features": data.dynamic_input_features,
        "static_input_features": data.static_input_features,
        "target_features": data.target_features,
        "height": data.height,
        "width": data.width,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return _DataBundle(
        train=_SyntheticForecastDataset(data, samples=data.train_samples, offset=0),
        validation=_SyntheticForecastDataset(
            data,
            samples=data.validation_samples,
            offset=data.train_samples,
        ),
        reference=DatasetReference(
            dataset_build_id=f"synthetic-{digest}",
            feature_schema_id=f"synthetic-{digest}",
        ),
        spec=spec,
    )


def _ml_dataset_bundle(config: RunConfig) -> _DataBundle:
    data = config.data
    if data.catalog_path is None or data.dataset is None:
        raise ValueError("ml_dataset runtime requires catalog_path and dataset config.")
    if not data.catalog_path.is_file():
        raise FileNotFoundError(data.catalog_path)
    catalog = SQLiteSampleCatalog(data.catalog_path)
    train_selection = SampleSelection(split="train")
    validation_selection = SampleSelection(split="validation")
    train_records = catalog.query(train_selection)
    validation_records = catalog.query(validation_selection)
    if not train_records:
        raise ValueError("Configured training split contains no samples.")
    if not validation_records:
        raise ValueError("Configured validation split contains no samples.")
    reference = _catalog_reference((*train_records, *validation_records))
    dataset_config = data.dataset
    spec = ModelSpec(
        name=config.model.name,
        dynamic_input_features=tuple(
            feature.name for feature in dataset_config.features.dynamic_inputs
        ),
        static_input_features=tuple(
            feature.name for feature in dataset_config.features.static_inputs
        ),
        target_features=tuple(
            feature.name for feature in dataset_config.features.targets
        ),
        history_steps=dataset_config.temporal.history_steps,
        forecast_steps=dataset_config.temporal.forecast_steps,
    )
    train_dataset = LazyRadarDataset(
        catalog=catalog,
        config=dataset_config,
        selection=train_selection,
        source_paths=data.source_paths,
    )
    validation_dataset = LazyRadarDataset(
        catalog=catalog,
        config=dataset_config,
        selection=validation_selection,
        source_paths=data.source_paths,
    )
    return _DataBundle(
        train=TorchRadarDataset(train_dataset),
        validation=TorchRadarDataset(validation_dataset),
        reference=reference,
        spec=spec,
        closers=(train_dataset.close, validation_dataset.close),
    )


def _evaluation_dataset(
    config: RunConfig,
    *,
    data: _DataBundle,
    split: str,
) -> tuple[Dataset[TorchSample], Callable[[], None] | None]:
    if split == "validation":
        return data.validation, None
    if split != "test":
        raise ValueError(f"Unsupported evaluation split: {split!r}.")
    if config.data.name != "ml_dataset":
        raise ValueError("Held-out test evaluation requires data.name='ml_dataset'.")
    dataset_config = config.data.dataset
    catalog_path = config.data.catalog_path
    if dataset_config is None or catalog_path is None:
        raise ValueError("ml_dataset runtime requires catalog_path and dataset config.")
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)

    catalog = SQLiteSampleCatalog(catalog_path)
    selection = SampleSelection(split="test")
    records = catalog.query(selection)
    if not records:
        raise ValueError("Configured test split contains no samples.")
    if _catalog_reference(records) != data.reference:
        raise ValueError(
            "Test samples must share the training dataset build and feature schema."
        )

    lazy = LazyRadarDataset(
        catalog=catalog,
        config=dataset_config,
        selection=selection,
        source_paths=config.data.source_paths,
    )
    return TorchRadarDataset(lazy), lazy.close


def _evaluation_split(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"validation", "test"}:
        raise ValueError("Reference evaluation split must be 'validation' or 'test'.")
    return normalized


def _catalog_reference(records: tuple[SampleRecord, ...]) -> DatasetReference:
    if not records:
        raise ValueError("Configured train/validation selections contain no samples.")
    build_ids = {record.dataset_build_id for record in records}
    schema_ids = {record.feature_schema_id for record in records}
    if len(build_ids) != 1 or len(schema_ids) != 1:
        raise ValueError(
            "Train and validation samples must share one dataset build "
            "and feature schema."
        )
    return DatasetReference(
        dataset_build_id=next(iter(build_ids)),
        feature_schema_id=next(iter(schema_ids)),
    )


def _build_model(config: RunConfig, spec: ModelSpec) -> nn.Module:
    parameters = config.model.parameters
    if config.model.name == "persistence":
        unknown = set(parameters) - {"target_input_features"}
        if unknown:
            raise ValueError(
                f"Unsupported persistence parameters: {sorted(unknown)!r}."
            )
        raw_mapping = parameters.get("target_input_features")
        target_inputs = (
            _string_tuple(raw_mapping, "model.parameters.target_input_features")
            if raw_mapping is not None
            else _infer_persistence_mapping(config.data)
        )
        return PersistenceForecast(spec, target_input_features=target_inputs)

    if config.model.name == "unet":
        if len(spec.dynamic_input_features) != 1 or len(spec.target_features) != 1:
            raise ValueError(
                "The 0.6 U-Net reference requires exactly one dynamic input "
                "and one target precipitation feature."
            )
        unknown = set(parameters) - {"base_channels"}
        if unknown:
            raise ValueError(f"Unsupported unet parameters: {sorted(unknown)!r}.")
        return UNetForecast(
            spec,
            base_channels=_positive_int_parameter(
                parameters,
                "base_channels",
                32,
            ),
        )

    raise ValueError(f"Unsupported model strategy: {config.model.name!r}.")


def _infer_persistence_mapping(data: DataConfig) -> tuple[str, ...] | None:
    if data.dataset is None:
        return None
    dynamic = data.dataset.features.dynamic_inputs
    result: list[str] = []
    for target in data.dataset.features.targets:
        matches = tuple(
            feature.name
            for feature in dynamic
            if feature.source_id == target.source_id
            and feature.variable == target.variable
        )
        if len(matches) != 1:
            raise ValueError(
                "Persistence requires exactly one dynamic input matching each target "
                "source and variable, or an explicit target_input_features mapping."
            )
        result.append(matches[0])
    return tuple(result)


def _training_transforms(
    config: RunConfig,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor] | None,
    Callable[[torch.Tensor], torch.Tensor] | None,
    Callable[[torch.Tensor], torch.Tensor] | None,
]:
    if config.model.name == "unet":
        return (
            log1p_precipitation,
            log1p_precipitation,
            inverse_log1p_precipitation,
        )
    return None, None, None


def _build_loss(config: RunConfig, spec: ModelSpec) -> MeanSquaredForecastLoss:
    if config.training.loss.name != "mse":
        raise ValueError(f"Unsupported forecast loss: {config.training.loss.name!r}.")
    if config.training.loss.parameters:
        raise ValueError("The mse loss does not accept parameters.")
    return MeanSquaredForecastLoss(spec)


def _build_optimizer(config: RunConfig, model: nn.Module) -> Optimizer | None:
    component = config.training.optimizer
    if component.name == "none":
        if component.parameters:
            raise ValueError("optimizer='none' does not accept parameters.")
        return None

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError(
            f"Optimizer {component.name!r} requires a model with trainable parameters."
        )
    if component.name == "adam":
        _require_only(component.parameters, {"lr", "weight_decay"}, "adam")
        return Adam(
            parameters,
            lr=_float_parameter(component.parameters, "lr", 1e-3),
            weight_decay=_float_parameter(component.parameters, "weight_decay", 0.0),
        )
    if component.name == "sgd":
        _require_only(
            component.parameters,
            {"lr", "momentum", "weight_decay"},
            "sgd",
        )
        return SGD(
            parameters,
            lr=_float_parameter(component.parameters, "lr", 1e-2),
            momentum=_float_parameter(component.parameters, "momentum", 0.0),
            weight_decay=_float_parameter(component.parameters, "weight_decay", 0.0),
        )
    raise ValueError(f"Unsupported optimizer: {component.name!r}.")


def _positive_int_parameter(
    parameters: Mapping[str, object],
    name: str,
    default: int,
) -> int:
    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Model parameter {name!r} must be an integer.")
    if value <= 0:
        raise ValueError(f"Model parameter {name!r} must be greater than zero.")
    return value


def _float_parameter(
    parameters: Mapping[str, object],
    name: str,
    default: float,
) -> float:
    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Optimizer parameter {name!r} must be numeric.")
    return float(value)


def _require_only(
    parameters: Mapping[str, object],
    allowed: set[str],
    component: str,
) -> None:
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(
            f"Unsupported {component} optimizer parameters: {sorted(unknown)!r}."
        )


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise TypeError(f"{field} must be a non-empty sequence of strings.")
    return tuple(item.strip() for item in value)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None
