"""Deterministic benchmark models for radar forecasting."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.models.validation import validate_model_inputs


class PersistenceForecast(nn.Module):
    """Repeat the latest observed target feature for every forecast step.

    Persistence is intentionally simple. It provides a deterministic weather
    forecasting benchmark that learned models should outperform. Target
    features are selected from dynamic input channels by name unless an
    explicit target-to-input mapping is supplied.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        target_input_features: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        source_features = (
            spec.target_features
            if target_input_features is None
            else target_input_features
        )
        if len(source_features) != len(spec.target_features):
            raise ValueError(
                "target_input_features must match the number of target features."
            )
        missing = tuple(
            feature
            for feature in source_features
            if feature not in spec.dynamic_input_features
        )
        if missing:
            raise ValueError(
                "Persistence targets must also be dynamic input features: "
                f"missing={missing!r}."
            )
        self._spec = spec
        indices = [
            spec.dynamic_input_features.index(feature) for feature in source_features
        ]
        self.register_buffer(
            "_target_indices",
            torch.tensor(indices, dtype=torch.long),
            persistent=False,
        )

    @property
    def spec(self) -> ModelSpec:
        """Return the immutable tensor contract implemented by this model."""
        return self._spec

    def forward(
        self,
        dynamic: Tensor,
        static: Tensor | None = None,
    ) -> Tensor:
        """Return the last observed target channels over the full horizon."""
        validate_model_inputs(self.spec, dynamic, static)
        target_indices = self.get_buffer("_target_indices")
        latest = dynamic[:, -1:, :, :, :].index_select(2, target_indices)
        return latest.expand(-1, self.spec.forecast_steps, -1, -1, -1)
