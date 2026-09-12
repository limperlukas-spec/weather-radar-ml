"""Runtime validation for the common forecasting tensor contract."""

from __future__ import annotations

from torch import Tensor

from weather_radar_ml.models.domain import ModelSpec


def validate_forecast_batch(
    spec: ModelSpec,
    dynamic: Tensor,
    target: Tensor,
    static: Tensor | None = None,
) -> None:
    """Validate one batched sample against a model specification."""
    _require_5d(dynamic, "dynamic inputs")
    _require_5d(target, "targets")

    batch, history, channels, height, width = dynamic.shape
    if history != spec.history_steps:
        raise ValueError(
            "Dynamic input history does not match model spec: "
            f"expected {spec.history_steps}, got {history}."
        )
    if channels != len(spec.dynamic_input_features):
        raise ValueError(
            "Dynamic input channels do not match model spec: "
            f"expected {len(spec.dynamic_input_features)}, got {channels}."
        )

    target_batch, forecast, target_channels, target_height, target_width = target.shape
    if target_batch != batch:
        raise ValueError("Dynamic inputs and targets must use the same batch size.")
    if forecast != spec.forecast_steps:
        raise ValueError(
            "Target forecast length does not match model spec: "
            f"expected {spec.forecast_steps}, got {forecast}."
        )
    if target_channels != len(spec.target_features):
        raise ValueError(
            "Target channels do not match model spec: "
            f"expected {len(spec.target_features)}, got {target_channels}."
        )
    if (target_height, target_width) != (height, width):
        raise ValueError("Dynamic inputs and targets must use the same spatial shape.")

    _validate_static(spec, static, batch=batch, height=height, width=width)


def validate_prediction(spec: ModelSpec, prediction: Tensor, target: Tensor) -> None:
    """Validate model output before loss and metric computation."""
    _require_5d(prediction, "prediction")
    _require_5d(target, "targets")

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction shape must match target shape: "
            f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}."
        )
    if prediction.shape[1] != spec.forecast_steps:
        raise ValueError(
            "Prediction forecast length does not match model spec: "
            f"expected {spec.forecast_steps}, got {prediction.shape[1]}."
        )
    if prediction.shape[2] != len(spec.target_features):
        raise ValueError(
            "Prediction channels do not match model spec: "
            f"expected {len(spec.target_features)}, got {prediction.shape[2]}."
        )


def _validate_static(
    spec: ModelSpec,
    static: Tensor | None,
    *,
    batch: int,
    height: int,
    width: int,
) -> None:
    expected_channels = len(spec.static_input_features)
    if expected_channels == 0:
        if static is not None:
            raise ValueError(
                "Model spec declares no static inputs, but static data was given."
            )
        return

    if static is None:
        raise ValueError("Model spec requires static inputs, but none were given.")
    if static.ndim != 4:
        raise ValueError(
            f"static inputs must have 4 dimensions [B, C, H, W], got {static.ndim}."
        )
    if static.shape[0] != batch:
        raise ValueError("Dynamic and static inputs must use the same batch size.")
    if static.shape[1] != expected_channels:
        raise ValueError(
            "Static input channels do not match model spec: "
            f"expected {expected_channels}, got {static.shape[1]}."
        )
    if tuple(static.shape[-2:]) != (height, width):
        raise ValueError("Dynamic and static inputs must use the same spatial shape.")


def _require_5d(value: Tensor, name: str) -> None:
    if value.ndim != 5:
        raise ValueError(
            f"{name} must have 5 dimensions [B, T, C, H, W], got {value.ndim}."
        )
