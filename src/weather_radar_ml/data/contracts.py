"""Protocol boundaries for the data lifecycle."""

from pathlib import Path
from typing import Protocol

from weather_radar_ml.data.domain import RadarSequence


class RadarDataSource(Protocol):
    """Acquire immutable source artifacts without interpreting their payload."""

    def acquire(self, destination: Path) -> tuple[Path, ...]: ...


class RadarDecoder(Protocol):
    """Decode source-specific artifacts into canonical radar data."""

    def decode(self, artifact: Path) -> RadarSequence: ...


class RadarPreprocessor(Protocol):
    """Apply deterministic, experiment-independent preprocessing."""

    def prepare(self, sequence: RadarSequence) -> RadarSequence: ...
