"""Typed metric results shared by training and offline evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Named metric values computed from a known number of scalar elements."""

    values: Mapping[str, float]
    count: int

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("MetricSummary count must be greater than zero.")
        normalized = {name.strip(): float(value) for name, value in self.values.items()}
        if not normalized or any(not name for name in normalized):
            raise ValueError("MetricSummary requires non-empty metric names.")
        if any(not isfinite(value) for value in normalized.values()):
            raise ValueError("MetricSummary values must be finite.")
        object.__setattr__(self, "values", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class ForecastMetricReport:
    """Overall metrics plus the same metrics resolved by forecast lead step."""

    overall: MetricSummary
    per_lead: tuple[MetricSummary, ...]

    def __post_init__(self) -> None:
        if not self.per_lead:
            raise ValueError("ForecastMetricReport requires per-lead metrics.")
