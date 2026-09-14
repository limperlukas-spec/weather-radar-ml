"""End-to-end runtime assembly for configured forecast experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD, Adam, Optimizer
from torch.utils.data import Dataset

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
from weather_radar_ml.models.baseline import PersistenceForecast
from weather_radar_ml.models.domain import ModelSpec
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
from weather_radar_ml.training.checkpoint import save_training_checkpoint
from weather_radar_ml.training.engine import ForecastTrainer
from weather_radar_ml.training.losses import MeanSquaredForecastLoss
from weather_radar_ml.training.reproducibility import configure_reproducibility
from weather_radar_ml.training.run import create_run_metadata


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Published local artifact and optional MLflow identity for one run."""

    artifact: RunArtifactResult
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


def run_experiment(config: RunConfig) -> PipelineResult:
    """Execute one configured experiment and publish its canonical run artifact."""
    data = _build_data_bundle(config)
    try:
        experiment = _experiment_config(config, data)
        configure_reproducibility(experiment.training)
        model = _build_model(config, data.spec)
        loss = _build_loss(config, data.spec)
        optimizer = _build_optimizer(config, model)
        trainer = ForecastTrainer(
            model=model,
            loss=loss,
            optimizer=optimizer,
            metric_factory=lambda: ContinuousForecastMetrics(data.spec),
            device=config.training.device,
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
        history = trainer.fit(
            train_loader,
            validation_loader,
            epochs=config.training.epochs,
        )
        metadata = create_run_metadata(
            experiment,
            git_commit=_git_commit(),
        )

        with tempfile.TemporaryDirectory(prefix="weather-radar-ml-checkpoint-") as tmp:
            checkpoints: tuple[CheckpointArtifact, ...] = ()
            if optimizer is not None:
                checkpoint_path = Path(tmp) / "final.pt"
                save_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    completed_epochs=config.training.epochs,
                )
                checkpoints = (
                    CheckpointArtifact(
                        checkpoint_path,
                        completed_epochs=config.training.epochs,
                    ),
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
                ),
            )
        return PipelineResult(artifact=artifact, tracking=tracking)
    finally:
        data.close()


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
    if config.model.name != "persistence":
        raise ValueError(f"Unsupported model strategy: {config.model.name!r}.")
    parameters = config.model.parameters
    unknown = set(parameters) - {"target_input_features"}
    if unknown:
        raise ValueError(f"Unsupported persistence parameters: {sorted(unknown)!r}.")
    raw_mapping = parameters.get("target_input_features")
    target_inputs = (
        _string_tuple(raw_mapping, "model.parameters.target_input_features")
        if raw_mapping is not None
        else _infer_persistence_mapping(config.data)
    )
    return PersistenceForecast(spec, target_input_features=target_inputs)


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
