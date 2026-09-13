"""MLflow adapter for immutable local experiment run artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast


@dataclass(frozen=True, slots=True)
class MlflowTrackingSettings:
    """Connection and grouping settings for MLflow experiment tracking."""

    tracking_uri: str = "http://127.0.0.1:5000"
    experiment_name: str = "weather-radar-ml"
    artifact_path: str = "run-artifact"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tracking_uri", _require_string(self.tracking_uri, "tracking_uri")
        )
        object.__setattr__(
            self,
            "experiment_name",
            _require_string(self.experiment_name, "experiment_name"),
        )
        object.__setattr__(
            self, "artifact_path", _require_string(self.artifact_path, "artifact_path")
        )


@dataclass(frozen=True, slots=True)
class MlflowTrackingResult:
    """Identity linking one canonical local run to its MLflow representation."""

    local_run_id: str
    experiment_id: str
    mlflow_run_id: str


class _Experiment(Protocol):
    @property
    def experiment_id(self) -> str: ...  # pragma: no cover


class _RunInfo(Protocol):
    @property
    def run_id(self) -> str: ...  # pragma: no cover


class _Run(Protocol):
    @property
    def info(self) -> _RunInfo: ...  # pragma: no cover


class MlflowClientProtocol(Protocol):
    """Small MLflow client surface required by the project adapter."""

    def get_experiment_by_name(self, name: str) -> _Experiment | None: ...

    def create_experiment(self, name: str) -> str: ...

    def create_run(
        self,
        experiment_id: str,
        *,
        tags: dict[str, object] | None = None,
        run_name: str | None = None,
    ) -> _Run: ...

    def log_param(self, run_id: str, key: str, value: object) -> object: ...

    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        *,
        step: int | None = None,
    ) -> object: ...

    def log_artifact(
        self,
        run_id: str,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None: ...

    def set_terminated(self, run_id: str, status: str | None = None) -> None: ...


class _MlflowClientFactory(Protocol):
    def __call__(self, *, tracking_uri: str) -> MlflowClientProtocol: ...


def create_mlflow_client(settings: MlflowTrackingSettings) -> MlflowClientProtocol:
    """Create the real MLflow client without making MLflow a core import."""
    try:
        mlflow_module: ModuleType = import_module("mlflow")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow tracking requires the optional 'mlflow-skinny' dependency."
        ) from exc

    factory = cast(_MlflowClientFactory, mlflow_module.__dict__["MlflowClient"])
    return factory(tracking_uri=settings.tracking_uri)


def track_run_artifact(
    run_root: str | Path,
    *,
    settings: MlflowTrackingSettings | None = None,
    client: MlflowClientProtocol | None = None,
) -> MlflowTrackingResult:
    """Publish one canonical local run artifact to MLflow for comparison."""
    tracking_settings = settings or MlflowTrackingSettings()
    root = Path(run_root)
    manifest_path = root / "manifest.json"
    config_path = root / "config.json"
    history_path = root / "history.json"
    for path in (manifest_path, config_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = _read_mapping(manifest_path)
    config = _read_mapping(config_path)
    history = _read_mapping(history_path)
    run_payload = _require_mapping(manifest.get("run"), "manifest.run")
    local_run_id = _require_string(run_payload.get("run_id"), "manifest.run.run_id")
    fingerprint = _require_string(
        manifest.get("config_fingerprint"),
        "manifest.config_fingerprint",
    )

    tracking_client = (
        client if client is not None else create_mlflow_client(tracking_settings)
    )
    experiment = tracking_client.get_experiment_by_name(
        tracking_settings.experiment_name
    )
    experiment_id = (
        tracking_client.create_experiment(tracking_settings.experiment_name)
        if experiment is None
        else experiment.experiment_id
    )
    mlflow_run = tracking_client.create_run(
        experiment_id,
        run_name=local_run_id,
        tags=_run_tags(run_payload, fingerprint, manifest),
    )
    mlflow_run_id = mlflow_run.info.run_id

    try:
        for key, value in _flatten_parameters("config", config).items():
            tracking_client.log_param(mlflow_run_id, key, value)
        _log_history_metrics(tracking_client, mlflow_run_id, history)
        for path in (manifest_path, config_path, history_path):
            tracking_client.log_artifact(
                mlflow_run_id,
                str(path),
                artifact_path=tracking_settings.artifact_path,
            )
    except Exception:
        with suppress(Exception):
            tracking_client.set_terminated(mlflow_run_id, status="FAILED")
        raise

    tracking_client.set_terminated(mlflow_run_id, status="FINISHED")
    return MlflowTrackingResult(
        local_run_id=local_run_id,
        experiment_id=experiment_id,
        mlflow_run_id=mlflow_run_id,
    )


def _run_tags(
    run_payload: Mapping[str, object],
    fingerprint: str,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    tags: dict[str, object] = {
        "weather_radar_ml.run_id": _require_string(
            run_payload.get("run_id"), "manifest.run.run_id"
        ),
        "weather_radar_ml.config_fingerprint": fingerprint,
        "weather_radar_ml.artifact_format_version": str(
            _require_int(manifest.get("format_version"), "manifest.format_version")
        ),
    }
    for source_key in (
        "git_commit",
        "python_version",
        "numpy_version",
        "torch_version",
        "platform",
    ):
        value = run_payload.get(source_key)
        if value is not None:
            tags[f"weather_radar_ml.{source_key}"] = _require_string(
                value, f"manifest.run.{source_key}"
            )
    return tags


def _flatten_parameters(prefix: str, value: object) -> dict[str, str]:
    if isinstance(value, Mapping):
        mapping = _require_mapping(value, prefix)
        flattened: dict[str, str] = {}
        for key in sorted(mapping):
            nested = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_parameters(nested, mapping[key]))
        return flattened
    if isinstance(value, list):
        return {prefix: json.dumps(value, sort_keys=True, separators=(",", ":"))}
    if value is None:
        return {prefix: "null"}
    if isinstance(value, bool):
        return {prefix: "true" if value else "false"}
    if isinstance(value, (str, int, float)):
        return {prefix: str(value)}
    raise ValueError(f"Unsupported MLflow parameter value at {prefix!r}.")


def _log_history_metrics(
    client: MlflowClientProtocol,
    run_id: str,
    history: Mapping[str, object],
) -> None:
    epochs = history.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("history.epochs must be a non-empty list.")
    for raw_epoch in epochs:
        epoch = _require_mapping(raw_epoch, "history epoch")
        step = _require_int(epoch.get("epoch"), "history epoch number")
        if step <= 0:
            raise ValueError("history epoch number must be greater than zero.")
        for split in ("train", "validation"):
            result = _require_mapping(epoch.get(split), f"history.{split}")
            client.log_metric(
                run_id,
                f"{split}/loss",
                _require_float(result.get("loss"), f"history.{split}.loss"),
                step=step,
            )
            metrics = _require_mapping(
                result.get("metrics"), f"history.{split}.metrics"
            )
            overall = _require_mapping(
                metrics.get("overall"), f"history.{split}.metrics.overall"
            )
            _log_metric_values(
                client,
                run_id,
                f"{split}",
                overall,
                step,
            )
            per_lead = metrics.get("per_lead")
            if not isinstance(per_lead, list) or not per_lead:
                raise ValueError(
                    f"history.{split}.metrics.per_lead must be a non-empty list."
                )
            for lead, summary in enumerate(per_lead, start=1):
                _log_metric_values(
                    client,
                    run_id,
                    f"{split}/lead_{lead:03d}",
                    _require_mapping(summary, f"history.{split}.metrics.per_lead"),
                    step,
                )


def _log_metric_values(
    client: MlflowClientProtocol,
    run_id: str,
    prefix: str,
    summary: Mapping[str, object],
    step: int,
) -> None:
    values = _require_mapping(summary.get("values"), f"{prefix}.values")
    for raw_name in sorted(values, key=str):
        name = _require_string(raw_name, f"{prefix} metric name")
        client.log_metric(
            run_id,
            f"{prefix}/{name}",
            _require_float(values[raw_name], f"{prefix}.{name}"),
            step=step,
        )


def _read_mapping(path: Path) -> Mapping[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    return _require_mapping(payload, str(path))


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping.")
    return cast(Mapping[str, object], value)


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _require_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized
