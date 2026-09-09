"""Framework-neutral contracts for ML radar datasets and transforms."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from weather_radar_ml.data.ml.domain import (
    RadarSample,
    SampleSelection,
    TemporalSplitRule,
)


@runtime_checkable
class RadarDatasetProtocol(Protocol):
    """Indexable view over a deterministic selection of radar samples."""

    def __len__(self) -> int: ...  # pragma: no cover

    def __getitem__(self, index: int) -> RadarSample: ...  # pragma: no cover

    def get(self, sample_id: str) -> RadarSample: ...  # pragma: no cover


@runtime_checkable
class SampleCatalogProtocol(Protocol):
    """Storage-independent catalog interface used to create dataset views."""

    def query(
        self, selection: SampleSelection
    ) -> RadarDatasetProtocol: ...  # pragma: no cover


@runtime_checkable
class SplitStrategy(Protocol):
    """Provide temporal split rules without knowing catalog persistence."""

    def rules(self) -> tuple[TemporalSplitRule, ...]: ...  # pragma: no cover


@runtime_checkable
class SampleTransform(Protocol):
    """Return a transformed sample without mutating the supplied sample."""

    def transform(self, sample: RadarSample) -> RadarSample: ...  # pragma: no cover


@runtime_checkable
class FittableSampleTransform(SampleTransform, Protocol):
    """Transform whose state can be fitted from an arbitrary sample iterable."""

    def fit(self, samples: Iterable[RadarSample]) -> None: ...  # pragma: no cover
