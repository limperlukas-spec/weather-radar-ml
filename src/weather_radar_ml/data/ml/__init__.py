"""Public ML-dataset domain API."""

from weather_radar_ml.data.ml.contracts import (
    FittableSampleTransform,
    RadarDatasetProtocol,
    SampleCatalogProtocol,
    SampleTransform,
    SplitStrategy,
)
from weather_radar_ml.data.ml.domain import (
    RadarInputs,
    RadarSample,
    RadarTargets,
    SampleContext,
    SampleSelection,
    SpatialWindow,
    SpatialWindowMode,
    TemporalSplitRule,
)

__all__ = [
    "FittableSampleTransform",
    "RadarDatasetProtocol",
    "RadarInputs",
    "RadarSample",
    "RadarTargets",
    "SampleCatalogProtocol",
    "SampleContext",
    "SampleSelection",
    "SampleTransform",
    "SpatialWindow",
    "SpatialWindowMode",
    "SplitStrategy",
    "TemporalSplitRule",
]
