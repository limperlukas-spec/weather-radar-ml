"""Framework-neutral model metadata for radar forecasting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Stable input/output contract exposed by one forecasting model.

    The spec describes feature ordering and temporal lengths independently of
    the concrete model architecture. Spatial dimensions are intentionally not
    fixed here because the same model may be evaluated on different windows.
    """

    name: str
    dynamic_input_features: tuple[str, ...]
    target_features: tuple[str, ...]
    history_steps: int
    forecast_steps: int
    static_input_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_name(self.name, "model name"))
        object.__setattr__(
            self,
            "dynamic_input_features",
            _require_unique_names(
                self.dynamic_input_features,
                "dynamic input features",
            ),
        )
        object.__setattr__(
            self,
            "static_input_features",
            _require_unique_names(
                self.static_input_features,
                "static input features",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "target_features",
            _require_unique_names(self.target_features, "target features"),
        )
        if self.history_steps <= 0:
            raise ValueError("history_steps must be greater than zero.")
        if self.forecast_steps <= 0:
            raise ValueError("forecast_steps must be greater than zero.")


def _require_name(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty.")
    return normalized


def _require_unique_names(
    values: tuple[str, ...],
    field: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(_require_name(value, field) for value in values)
    if not normalized and not allow_empty:
        raise ValueError(f"{field} must contain at least one entry.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must not contain duplicates.")
    return normalized
