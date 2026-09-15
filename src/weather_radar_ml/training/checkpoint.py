"""Versioned and atomic training checkpoints for reproducible resume."""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from weather_radar_ml.evaluation.domain import ForecastMetricReport, MetricSummary
from weather_radar_ml.models.contracts import ForecastModel
from weather_radar_ml.training.domain import EpochResult, TrainingHistory

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainingCheckpointState:
    """Metadata returned after a checkpoint has been restored."""

    completed_epochs: int
    history: TrainingHistory | None = None
    dataloader_generator_state: Tensor | None = None
    format_version: int = CHECKPOINT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.completed_epochs < 0:
            raise ValueError("completed_epochs must not be negative.")


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    completed_epochs: int,
    history: TrainingHistory | None = None,
    dataloader_generator_state: Tensor | None = None,
) -> None:
    """Atomically persist model, optimizer, epoch, RNG, and optional resume state."""
    if history is not None and len(history.train) != completed_epochs:
        raise ValueError("Checkpoint history must match completed_epochs.")
    state = TrainingCheckpointState(
        completed_epochs=completed_epochs,
        history=history,
        dataloader_generator_state=dataloader_generator_state,
    )
    forecast_model = _forecast_model(model)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "format_version": state.format_version,
        "completed_epochs": state.completed_epochs,
        "model_type": _qualified_type(model),
        "optimizer_type": _qualified_type(optimizer),
        "model_spec": asdict(forecast_model.spec),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": _capture_rng_state(),
    }
    if history is not None:
        payload["training_history"] = _history_payload(history)
    if dataloader_generator_state is not None:
        payload["dataloader_generator_state"] = (
            dataloader_generator_state.detach().cpu().clone()
        )
    _atomic_torch_save(payload, destination)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    restore_rng_state: bool = True,
) -> TrainingCheckpointState:
    """Restore a compatible checkpoint into an existing model and optimizer."""
    forecast_model = _forecast_model(model)
    source = Path(path)
    raw = torch.load(source, map_location="cpu", weights_only=True)
    payload = _require_mapping(raw, "checkpoint")

    version = _require_int(payload, "format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format version "
            f"{version}; expected {CHECKPOINT_FORMAT_VERSION}."
        )

    completed_epochs = _require_int(payload, "completed_epochs")
    history_raw = payload.get("training_history")
    history = (
        None
        if history_raw is None
        else _history_from_payload(_require_mapping(history_raw, "training_history"))
    )
    if history is not None and len(history.train) != completed_epochs:
        raise ValueError("Checkpoint history does not match completed_epochs.")
    generator_state_raw = payload.get("dataloader_generator_state")
    if generator_state_raw is not None and not isinstance(generator_state_raw, Tensor):
        raise ValueError("dataloader_generator_state must be a tensor.")
    state = TrainingCheckpointState(
        completed_epochs=completed_epochs,
        history=history,
        dataloader_generator_state=(
            None
            if generator_state_raw is None
            else generator_state_raw.detach().cpu().clone()
        ),
        format_version=version,
    )

    _require_equal(payload, "model_type", _qualified_type(model))
    _require_equal(payload, "optimizer_type", _qualified_type(optimizer))
    _require_equal(payload, "model_spec", asdict(forecast_model.spec))

    model_state = cast(
        Mapping[str, Tensor],
        _require_mapping(payload.get("model_state"), "model_state"),
    )
    optimizer_state = cast(
        dict[str, Any],
        _require_mapping(payload.get("optimizer_state"), "optimizer_state"),
    )
    rng_state = _require_mapping(payload.get("rng_state"), "rng_state")

    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    if restore_rng_state:
        _restore_rng_state(rng_state)
    return state


def _forecast_model(model: nn.Module) -> ForecastModel:
    if not isinstance(model, ForecastModel):
        raise TypeError("model must satisfy the ForecastModel protocol.")
    return cast(ForecastModel, model)


def _qualified_type(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _capture_rng_state() -> dict[str, object]:
    python_version, python_internal, python_gauss = random.getstate()
    numpy_name, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached = (
        np.random.get_state()
    )
    numpy_keys_array = np.asarray(numpy_keys, dtype=np.uint32)
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    return {
        "python": {
            "version": python_version,
            "internal": list(python_internal),
            "gauss": python_gauss,
        },
        "numpy": {
            "name": numpy_name,
            "keys": torch.tensor(numpy_keys_array, dtype=torch.int64),
            "position": numpy_position,
            "has_gauss": numpy_has_gauss,
            "cached_gaussian": numpy_cached,
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "torch_mps": mps_state,
    }


def _restore_rng_state(payload: Mapping[str, object]) -> None:
    python_state = _require_mapping(payload.get("python"), "rng_state.python")
    python_version = _require_int(python_state, "version")
    python_internal_raw = python_state.get("internal")
    if not isinstance(python_internal_raw, list) or not all(
        isinstance(value, int) for value in python_internal_raw
    ):
        raise ValueError("rng_state.python.internal must be a list of integers.")
    python_gauss = python_state.get("gauss")
    if python_gauss is not None and not isinstance(python_gauss, float):
        raise ValueError("rng_state.python.gauss must be a float or None.")
    random.setstate((python_version, tuple(python_internal_raw), python_gauss))

    numpy_state = _require_mapping(payload.get("numpy"), "rng_state.numpy")
    numpy_name = numpy_state.get("name")
    if not isinstance(numpy_name, str):
        raise ValueError("rng_state.numpy.name must be a string.")
    numpy_keys = numpy_state.get("keys")
    if not isinstance(numpy_keys, Tensor) or numpy_keys.ndim != 1:
        raise ValueError("rng_state.numpy.keys must be a one-dimensional tensor.")
    numpy_position = _require_int(numpy_state, "position")
    numpy_has_gauss = _require_int(numpy_state, "has_gauss")
    numpy_cached = numpy_state.get("cached_gaussian")
    if not isinstance(numpy_cached, float):
        raise ValueError("rng_state.numpy.cached_gaussian must be a float.")
    np.random.set_state(
        (
            numpy_name,
            numpy_keys.cpu().numpy().astype(np.uint32),
            numpy_position,
            numpy_has_gauss,
            numpy_cached,
        )
    )

    torch_cpu = payload.get("torch_cpu")
    if not isinstance(torch_cpu, Tensor):
        raise ValueError("rng_state.torch_cpu must be a tensor.")
    torch.set_rng_state(torch_cpu.cpu())

    cuda_states = payload.get("torch_cuda")
    if not isinstance(cuda_states, list) or not all(
        isinstance(value, Tensor) for value in cuda_states
    ):
        raise ValueError("rng_state.torch_cuda must be a list of tensors.")
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA device count does not match the saved checkpoint RNG state."
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])

    mps_state = payload.get("torch_mps")
    if mps_state is not None:
        if not isinstance(mps_state, Tensor):
            raise ValueError("rng_state.torch_mps must be a tensor or None.")
        if torch.backends.mps.is_available():
            torch.mps.set_rng_state(mps_state.cpu())


def _history_payload(history: TrainingHistory) -> dict[str, object]:
    return {
        "best_epoch": history.best_epoch,
        "best_validation_loss": history.best_validation_loss,
        "stopped_early": history.stopped_early,
        "train": [_epoch_payload(item) for item in history.train],
        "validation": [_epoch_payload(item) for item in history.validation],
    }


def _history_from_payload(payload: Mapping[str, object]) -> TrainingHistory:
    train = _epoch_sequence(payload.get("train"), "training_history.train")
    validation = _epoch_sequence(
        payload.get("validation"),
        "training_history.validation",
    )
    best_epoch = _require_int(payload, "best_epoch")
    best_loss = payload.get("best_validation_loss")
    if isinstance(best_loss, bool) or not isinstance(best_loss, (int, float)):
        raise ValueError("best_validation_loss must be numeric.")
    stopped_early = payload.get("stopped_early")
    if not isinstance(stopped_early, bool):
        raise ValueError("stopped_early must be boolean.")
    return TrainingHistory(
        train=train,
        validation=validation,
        best_epoch=best_epoch,
        best_validation_loss=float(best_loss),
        stopped_early=stopped_early,
    )


def _epoch_sequence(value: object, name: str) -> tuple[EpochResult, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list.")
    return tuple(
        _epoch_from_payload(_require_mapping(item, f"{name} item")) for item in value
    )


def _epoch_payload(result: EpochResult) -> dict[str, object]:
    return {
        "loss": result.loss,
        "batches": result.batches,
        "samples": result.samples,
        "metrics": {
            "overall": _metric_summary_payload(result.metrics.overall),
            "per_lead": [
                _metric_summary_payload(item) for item in result.metrics.per_lead
            ],
        },
    }


def _epoch_from_payload(payload: Mapping[str, object]) -> EpochResult:
    loss = _require_float(payload.get("loss"), "epoch.loss")
    metrics = _require_mapping(payload.get("metrics"), "epoch.metrics")
    overall = _metric_summary_from_payload(
        _require_mapping(metrics.get("overall"), "epoch.metrics.overall")
    )
    per_lead_raw = metrics.get("per_lead")
    if not isinstance(per_lead_raw, list) or not per_lead_raw:
        raise ValueError("epoch.metrics.per_lead must be a non-empty list.")
    per_lead = tuple(
        _metric_summary_from_payload(
            _require_mapping(item, "epoch.metrics.per_lead item")
        )
        for item in per_lead_raw
    )
    return EpochResult(
        loss=loss,
        metrics=ForecastMetricReport(overall=overall, per_lead=per_lead),
        batches=_require_int(payload, "batches"),
        samples=_require_int(payload, "samples"),
    )


def _metric_summary_payload(summary: MetricSummary) -> dict[str, object]:
    return {"values": dict(summary.values), "count": summary.count}


def _metric_summary_from_payload(payload: Mapping[str, object]) -> MetricSummary:
    values = _require_mapping(payload.get("values"), "metric.values")
    numeric: dict[str, float] = {}
    for key, value in values.items():
        numeric[key] = _require_float(value, f"metric.values.{key}")
    return MetricSummary(values=numeric, count=_require_int(payload, "count"))


def _require_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    return float(value)


def _atomic_torch_save(payload: object, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping.")
    return cast(Mapping[str, object], value)


def _require_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer.")
    return value


def _require_equal(
    payload: Mapping[str, object],
    key: str,
    expected: object,
) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise ValueError(
            f"Checkpoint {key} mismatch: expected {expected!r}, got {actual!r}."
        )
