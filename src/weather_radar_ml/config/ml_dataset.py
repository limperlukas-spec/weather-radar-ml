"""Typed configuration models for reproducible ML dataset builds."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast


class SourceRole(StrEnum):
    """Role of a prepared source in an ML dataset build."""

    PRIMARY = "primary"
    AUXILIARY = "auxiliary"


class FeatureKind(StrEnum):
    """Temporal behaviour of an ML feature."""

    DYNAMIC = "dynamic"
    STATIC = "static"


class SpatialMode(StrEnum):
    """Spatial sampling mode."""

    FULL_FRAME = "full_frame"
    PATCH = "patch"


class SplitName(StrEnum):
    """Split names implemented by Data Foundation 0.4."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class EventClass(StrEnum):
    """Versioned precipitation-event labels."""

    DRY = "dry"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    EXTREME = "extreme"


@dataclass(frozen=True, slots=True)
class PreparedSourceConfig:
    """Reference to one versioned prepared source."""

    source_id: str
    artifact_id: str
    role: SourceRole
    path: Path | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.source_id, "source_id")
        _require_non_empty(self.artifact_id, "artifact_id")


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """One ordered feature exposed to an ML sample."""

    name: str
    source_id: str
    variable: str
    kind: FeatureKind

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "feature name")
        _require_non_empty(self.source_id, "feature source_id")
        _require_non_empty(self.variable, "feature variable")


@dataclass(frozen=True, slots=True)
class FeatureSchemaConfig:
    """Ordered dynamic/static input and target feature definitions."""

    dynamic_inputs: tuple[FeatureConfig, ...]
    static_inputs: tuple[FeatureConfig, ...] = ()
    targets: tuple[FeatureConfig, ...] = ()

    def __post_init__(self) -> None:
        _validate_feature_group(
            self.dynamic_inputs,
            FeatureKind.DYNAMIC,
            "dynamic_inputs",
        )
        _validate_feature_group(
            self.static_inputs,
            FeatureKind.STATIC,
            "static_inputs",
        )
        _validate_feature_group(self.targets, FeatureKind.DYNAMIC, "targets")

        all_names = [
            feature.name
            for group in (self.dynamic_inputs, self.static_inputs, self.targets)
            for feature in group
        ]
        if len(all_names) != len(set(all_names)):
            raise ValueError("Feature names must be unique across the feature schema.")
        if not self.dynamic_inputs:
            raise ValueError("At least one dynamic input feature is required.")
        if not self.targets:
            raise ValueError("At least one target feature is required.")


@dataclass(frozen=True, slots=True)
class TemporalWindowConfig:
    """Temporal sample geometry on one regular time grid."""

    step_minutes: int = 5
    history_minutes: int = 60
    forecast_minutes: int = 120

    def __post_init__(self) -> None:
        if self.step_minutes <= 0:
            raise ValueError("step_minutes must be greater than zero.")
        if self.history_minutes <= 0:
            raise ValueError("history_minutes must be greater than zero.")
        if self.forecast_minutes <= 0:
            raise ValueError("forecast_minutes must be greater than zero.")
        if self.history_minutes % self.step_minutes:
            raise ValueError("history_minutes must be divisible by step_minutes.")
        if self.forecast_minutes % self.step_minutes:
            raise ValueError("forecast_minutes must be divisible by step_minutes.")

    @property
    def history_steps(self) -> int:
        return self.history_minutes // self.step_minutes

    @property
    def forecast_steps(self) -> int:
        return self.forecast_minutes // self.step_minutes


@dataclass(frozen=True, slots=True)
class SpatialSamplingConfig:
    """Full-frame or fixed-size patch sampling configuration."""

    mode: SpatialMode = SpatialMode.FULL_FRAME
    patch_height: int | None = None
    patch_width: int | None = None
    stride_y: int | None = None
    stride_x: int | None = None

    def __post_init__(self) -> None:
        patch_values = (
            self.patch_height,
            self.patch_width,
            self.stride_y,
            self.stride_x,
        )
        if self.mode is SpatialMode.FULL_FRAME:
            if any(value is not None for value in patch_values):
                raise ValueError(
                    "Patch dimensions and strides must be omitted in full_frame mode."
                )
            return
        if any(value is None for value in patch_values):
            raise ValueError(
                "Patch mode requires patch_height, patch_width, stride_y, and stride_x."
            )
        if any(value is not None and value <= 0 for value in patch_values):
            raise ValueError("Patch dimensions and strides must be greater than zero.")


@dataclass(frozen=True, slots=True)
class SplitIntervalConfig:
    """One explicit UTC interval used by the first temporal split strategy."""

    name: SplitName
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        start = _as_utc(self.start, "split start")
        end = _as_utc(self.end, "split end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if start >= end:
            raise ValueError("Split start must be earlier than split end.")


@dataclass(frozen=True, slots=True)
class TemporalSplitConfig:
    """Explicit temporal blocks with optional intentional gaps."""

    intervals: tuple[SplitIntervalConfig, ...]

    def __post_init__(self) -> None:
        names = [interval.name for interval in self.intervals]
        if len(names) != len(set(names)):
            raise ValueError("Each split name may occur at most once.")
        required = {SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST}
        if set(names) != required:
            raise ValueError(
                "Exactly train, validation, and test intervals are required."
            )
        ordered = sorted(self.intervals, key=lambda interval: interval.start)
        for current, following in itertools.pairwise(ordered):
            if current.end > following.start:
                raise ValueError("Split intervals must not overlap.")


@dataclass(frozen=True, slots=True)
class EventThresholdConfig:
    """Upper p95 intensity threshold for one non-extreme event class."""

    event_class: EventClass
    upper_p95_mm_per_h: float

    def __post_init__(self) -> None:
        if self.event_class in {EventClass.DRY, EventClass.EXTREME}:
            raise ValueError(
                "Threshold entries are only valid for light/moderate/heavy."
            )
        if self.upper_p95_mm_per_h <= 0:
            raise ValueError("Event thresholds must be greater than zero.")


@dataclass(frozen=True, slots=True)
class PrecipEventProfileConfig:
    """Versioned precipitation-event classifier configuration.

    ``precip_event_v1`` contains technical defaults only. They are deliberately
    configurable and must be validated against the real RADKLIM distribution
    before being treated as meteorologically calibrated thresholds.
    """

    profile_id: str = "precip_event_v1"
    wet_pixel_threshold_mm_per_h: float = 0.1
    min_wet_fraction: float = 0.01
    thresholds: tuple[EventThresholdConfig, ...] = (
        EventThresholdConfig(EventClass.LIGHT, 2.5),
        EventThresholdConfig(EventClass.MODERATE, 10.0),
        EventThresholdConfig(EventClass.HEAVY, 25.0),
    )

    def __post_init__(self) -> None:
        _require_non_empty(self.profile_id, "profile_id")
        if self.wet_pixel_threshold_mm_per_h < 0:
            raise ValueError("wet_pixel_threshold_mm_per_h must not be negative.")
        if not 0.0 <= self.min_wet_fraction <= 1.0:
            raise ValueError("min_wet_fraction must be between zero and one.")
        expected = [EventClass.LIGHT, EventClass.MODERATE, EventClass.HEAVY]
        actual = [threshold.event_class for threshold in self.thresholds]
        if actual != expected:
            raise ValueError(
                "Event thresholds must be ordered as light, moderate, heavy."
            )
        values = [threshold.upper_p95_mm_per_h for threshold in self.thresholds]
        for current, following in itertools.pairwise(values):
            if current >= following:
                raise ValueError("Event thresholds must be strictly increasing.")


@dataclass(frozen=True, slots=True)
class MLDatasetConfig:
    """Fully typed configuration for one immutable ML dataset build."""

    name: str
    sources: tuple[PreparedSourceConfig, ...]
    features: FeatureSchemaConfig
    temporal: TemporalWindowConfig
    spatial: SpatialSamplingConfig
    splits: TemporalSplitConfig
    events: PrecipEventProfileConfig = field(default_factory=PrecipEventProfileConfig)
    sample_schema_version: int = 1

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "dataset name")
        if self.sample_schema_version <= 0:
            raise ValueError("sample_schema_version must be greater than zero.")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique.")
        primary = [
            source for source in self.sources if source.role is SourceRole.PRIMARY
        ]
        if len(primary) != 1:
            raise ValueError("Exactly one primary prepared source is required.")
        known_sources = set(source_ids)
        for feature in (
            *self.features.dynamic_inputs,
            *self.features.static_inputs,
            *self.features.targets,
        ):
            if feature.source_id not in known_sources:
                raise ValueError(
                    f"Feature {feature.name!r} references unknown source "
                    f"{feature.source_id!r}."
                )

    def to_resolved_dict(self) -> dict[str, Any]:
        """Return a serialization-ready representation with all defaults resolved."""

        return cast(dict[str, Any], _serialize(asdict(self)))


def ml_dataset_config_from_mapping(data: Mapping[str, Any]) -> MLDatasetConfig:
    """Build and strictly validate an ML dataset config from a mapping.

    Unknown keys are rejected at every level. Hydra/OmegaConf may resolve
    composition first and pass the resulting plain mapping here.
    """

    root = _strict_mapping(
        data,
        {
            "name",
            "sources",
            "features",
            "temporal",
            "spatial",
            "splits",
            "events",
            "sample_schema_version",
        },
        "dataset",
    )
    sources = tuple(
        _parse_source(item)
        for item in _require_sequence(root.get("sources"), "sources")
    )
    features = _parse_features(_require_mapping(root.get("features"), "features"))
    temporal = _parse_temporal(_require_mapping(root.get("temporal"), "temporal"))
    spatial = _parse_spatial(_require_mapping(root.get("spatial"), "spatial"))
    splits = _parse_splits(_require_mapping(root.get("splits"), "splits"))
    events_raw = root.get("events")
    events = (
        PrecipEventProfileConfig()
        if events_raw is None
        else _parse_events(_require_mapping(events_raw, "events"))
    )
    return MLDatasetConfig(
        name=_require_str(root.get("name"), "name"),
        sources=sources,
        features=features,
        temporal=temporal,
        spatial=spatial,
        splits=splits,
        events=events,
        sample_schema_version=int(root.get("sample_schema_version", 1)),
    )


def _parse_source(value: Any) -> PreparedSourceConfig:
    data = _strict_mapping(
        _require_mapping(value, "source"),
        {"source_id", "artifact_id", "role", "path"},
        "source",
    )
    path = data.get("path")
    return PreparedSourceConfig(
        source_id=_require_str(data.get("source_id"), "source.source_id"),
        artifact_id=_require_str(data.get("artifact_id"), "source.artifact_id"),
        role=SourceRole(_require_str(data.get("role"), "source.role")),
        path=None if path is None else Path(_require_str(path, "source.path")),
    )


def _parse_feature(value: Any, expected_kind: FeatureKind) -> FeatureConfig:
    data = _strict_mapping(
        _require_mapping(value, "feature"),
        {"name", "source_id", "variable", "kind"},
        "feature",
    )
    kind = FeatureKind(data.get("kind", expected_kind.value))
    return FeatureConfig(
        name=_require_str(data.get("name"), "feature.name"),
        source_id=_require_str(data.get("source_id"), "feature.source_id"),
        variable=_require_str(data.get("variable"), "feature.variable"),
        kind=kind,
    )


def _parse_features(data: Mapping[str, Any]) -> FeatureSchemaConfig:
    data = _strict_mapping(
        data,
        {"dynamic_inputs", "static_inputs", "targets"},
        "features",
    )
    dynamic = tuple(
        _parse_feature(item, FeatureKind.DYNAMIC)
        for item in _require_sequence(data.get("dynamic_inputs"), "dynamic_inputs")
    )
    static = tuple(
        _parse_feature(item, FeatureKind.STATIC)
        for item in _optional_sequence(data.get("static_inputs"))
    )
    targets = tuple(
        _parse_feature(item, FeatureKind.DYNAMIC)
        for item in _require_sequence(data.get("targets"), "targets")
    )
    return FeatureSchemaConfig(dynamic, static, targets)


def _parse_temporal(data: Mapping[str, Any]) -> TemporalWindowConfig:
    data = _strict_mapping(
        data,
        {"step_minutes", "history_minutes", "forecast_minutes"},
        "temporal",
    )
    return TemporalWindowConfig(
        step_minutes=int(data.get("step_minutes", 5)),
        history_minutes=int(data.get("history_minutes", 60)),
        forecast_minutes=int(data.get("forecast_minutes", 120)),
    )


def _parse_spatial(data: Mapping[str, Any]) -> SpatialSamplingConfig:
    data = _strict_mapping(
        data,
        {"mode", "patch_height", "patch_width", "stride_y", "stride_x"},
        "spatial",
    )
    return SpatialSamplingConfig(
        mode=SpatialMode(data.get("mode", SpatialMode.FULL_FRAME.value)),
        patch_height=_optional_int(data.get("patch_height")),
        patch_width=_optional_int(data.get("patch_width")),
        stride_y=_optional_int(data.get("stride_y")),
        stride_x=_optional_int(data.get("stride_x")),
    )


def _parse_splits(data: Mapping[str, Any]) -> TemporalSplitConfig:
    data = _strict_mapping(data, {"intervals"}, "splits")
    intervals = tuple(
        _parse_split_interval(item)
        for item in _require_sequence(data.get("intervals"), "splits.intervals")
    )
    return TemporalSplitConfig(intervals)


def _parse_split_interval(value: Any) -> SplitIntervalConfig:
    data = _strict_mapping(
        _require_mapping(value, "split interval"),
        {"name", "start", "end"},
        "split interval",
    )
    return SplitIntervalConfig(
        name=SplitName(_require_str(data.get("name"), "split.name")),
        start=_parse_datetime(data.get("start"), "split.start"),
        end=_parse_datetime(data.get("end"), "split.end"),
    )


def _parse_events(data: Mapping[str, Any]) -> PrecipEventProfileConfig:
    data = _strict_mapping(
        data,
        {
            "profile_id",
            "wet_pixel_threshold_mm_per_h",
            "min_wet_fraction",
            "thresholds",
        },
        "events",
    )
    thresholds_raw = data.get("thresholds")
    thresholds = (
        PrecipEventProfileConfig().thresholds
        if thresholds_raw is None
        else tuple(
            _parse_event_threshold(item)
            for item in _require_sequence(thresholds_raw, "events.thresholds")
        )
    )
    return PrecipEventProfileConfig(
        profile_id=str(data.get("profile_id", "precip_event_v1")),
        wet_pixel_threshold_mm_per_h=float(
            data.get("wet_pixel_threshold_mm_per_h", 0.1)
        ),
        min_wet_fraction=float(data.get("min_wet_fraction", 0.01)),
        thresholds=thresholds,
    )


def _parse_event_threshold(value: Any) -> EventThresholdConfig:
    data = _strict_mapping(
        _require_mapping(value, "event threshold"),
        {"event_class", "upper_p95_mm_per_h"},
        "event threshold",
    )
    return EventThresholdConfig(
        EventClass(_require_str(data.get("event_class"), "event_class")),
        float(data["upper_p95_mm_per_h"]),
    )


def _validate_feature_group(
    features: Sequence[FeatureConfig],
    expected: FeatureKind,
    name: str,
) -> None:
    if any(feature.kind is not expected for feature in features):
        raise ValueError(f"All {name} features must have kind={expected.value!r}.")


def _strict_mapping(
    value: Mapping[str, Any],
    allowed: set[str],
    name: str,
) -> Mapping[str, Any]:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} config keys: {sorted(unknown)!r}.")
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _require_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _optional_sequence(value: Any) -> Sequence[Any]:
    return () if value is None else _require_sequence(value, "optional sequence")


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _require_non_empty(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _parse_datetime(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 string or datetime.")
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 timestamp.") from exc


def _as_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware.")
    return value.astimezone(UTC)


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value
