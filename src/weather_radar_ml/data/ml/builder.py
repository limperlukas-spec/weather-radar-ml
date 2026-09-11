"""Build deterministic ML sample indexes from prepared radar datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any

import numpy as np
import xarray as xr

from weather_radar_ml.config.ml_dataset import (
    EventClass,
    FeatureConfig,
    MLDatasetConfig,
    PrecipEventProfileConfig,
    PreparedSourceConfig,
    SourceRole,
    SpatialMode,
)
from weather_radar_ml.data.ml.catalog import SQLiteSampleCatalog
from weather_radar_ml.data.ml.domain import SpatialWindow, SpatialWindowMode

_HASH_HEX_LENGTH = 16
_BUILD_HASH_HEX_LENGTH = 8


@dataclass(frozen=True, slots=True)
class PrecipitationStatistics:
    """Finite-value precipitation statistics for one sample segment."""

    mean: float
    maximum: float
    p95: float
    wet_fraction: float


@dataclass(frozen=True, slots=True)
class BuildReport:
    """Summary of one sample-index build."""

    dataset_build_id: str
    feature_schema_id: str
    valid_temporal_windows: int
    rejected_missing_frames: int
    rejected_outside_splits: int
    spatial_windows: int
    samples: int
    samples_by_split: Mapping[str, int]
    input_events: Mapping[str, int]
    target_events: Mapping[str, int]


class SampleIndexBuilder:
    """Create one immutable SQLite sample index from prepared xarray datasets."""

    def __init__(
        self,
        *,
        config: MLDatasetConfig,
        source_datasets: Mapping[str, xr.Dataset],
        source_checksums: Mapping[str, str],
        catalog: SQLiteSampleCatalog,
    ) -> None:
        self.config = config
        self.source_datasets = dict(source_datasets)
        self.source_checksums = dict(source_checksums)
        self.catalog = catalog
        self._validate_sources()

    def build(self) -> BuildReport:
        """Build and persist all valid samples for the configured dataset."""

        primary_source_id = self._primary_source_id()
        primary = self.source_datasets[primary_source_id]
        feature_schema = feature_schema_id(self.config)
        build_id = dataset_build_id(self.config, self.source_checksums)

        self.catalog.initialize()
        self.catalog.add_classification_profile(
            profile_id=self.config.events.profile_id,
            definition_json=_canonical_json(_event_profile_payload(self.config.events)),
        )
        self.catalog.add_dataset_build(
            dataset_build_id=build_id,
            dataset_name=self.config.name,
            sample_schema_version=self.config.sample_schema_version,
            feature_schema_id=feature_schema,
            classification_profile_id=self.config.events.profile_id,
        )
        for source in self.config.sources:
            self.catalog.add_source(
                dataset_build_id=build_id,
                source_id=source.source_id,
                role=source.role.value,
                artifact_id=source.artifact_id,
                checksum=self.source_checksums[source.source_id],
            )

        split_ids = {
            interval.name.value: self.catalog.add_split(
                dataset_build_id=build_id,
                name=interval.name.value,
            )
            for interval in self.config.splits.intervals
        }

        windows = _spatial_windows(primary, self.config)
        window_ids = {
            window: self.catalog.add_spatial_window(
                dataset_build_id=build_id,
                x=window.x,
                y=window.y,
                width=window.width,
                height=window.height,
                mode=window.mode,
            )
            for window in windows
        }

        times = _time_values(primary)
        time_to_index = {value: index for index, value in enumerate(times)}
        step = timedelta(minutes=self.config.temporal.step_minutes)
        history_steps = self.config.temporal.history_steps
        forecast_steps = self.config.temporal.forecast_steps
        precipitation_variable = _precipitation_variable(self.config, primary_source_id)

        valid_temporal_windows = 0
        rejected_missing = 0
        rejected_split = 0
        samples = 0
        split_counter: Counter[str] = Counter()
        input_event_counter: Counter[str] = Counter()
        target_event_counter: Counter[str] = Counter()

        for start in times:
            input_times = tuple(
                start + step * offset for offset in range(history_steps)
            )
            target_start = start + timedelta(
                minutes=self.config.temporal.history_minutes
            )
            target_times = tuple(
                target_start + step * offset for offset in range(forecast_steps)
            )
            required_times = (*input_times, *target_times)
            if any(value not in time_to_index for value in required_times):
                if target_times[-1] <= times[-1]:
                    rejected_missing += 1
                continue

            split = _assign_split(
                self.config,
                input_start=input_times[0],
                target_end=target_times[-1],
            )
            if split is None:
                rejected_split += 1
                continue

            valid_temporal_windows += 1
            input_indices = [time_to_index[value] for value in input_times]
            target_indices = [time_to_index[value] for value in target_times]

            for window in windows:
                input_stats = _precipitation_statistics(
                    primary,
                    precipitation_variable,
                    input_indices,
                    window,
                    self.config.events.wet_pixel_threshold_mm_per_h,
                )
                target_stats = _precipitation_statistics(
                    primary,
                    precipitation_variable,
                    target_indices,
                    window,
                    self.config.events.wet_pixel_threshold_mm_per_h,
                )
                input_event = classify_precipitation(input_stats, self.config.events)
                target_event = classify_precipitation(target_stats, self.config.events)
                sample_identifier = sample_id(
                    self.config,
                    self.source_checksums,
                    input_times=input_times,
                    target_times=target_times,
                    window=window,
                )

                (
                    input_mean,
                    input_max,
                    input_p95,
                    input_wet_fraction,
                ) = _stats_values(input_stats)
                (
                    target_mean,
                    target_max,
                    target_p95,
                    target_wet_fraction,
                ) = _stats_values(target_stats)
                self.catalog.add_sample(
                    sample_id=sample_identifier,
                    dataset_build_id=build_id,
                    feature_schema_id=feature_schema,
                    input_start=input_times[0],
                    input_end=input_times[-1],
                    target_start=target_times[0],
                    target_end=target_times[-1],
                    window_id=window_ids[window],
                    split_id=split_ids[split],
                    input_precip_mean=input_mean,
                    input_precip_max=input_max,
                    input_precip_p95=input_p95,
                    input_wet_fraction=input_wet_fraction,
                    target_precip_mean=target_mean,
                    target_precip_max=target_max,
                    target_precip_p95=target_p95,
                    target_wet_fraction=target_wet_fraction,
                    input_event_class=(
                        None if input_event is None else input_event.value
                    ),
                    target_event_class=(
                        None if target_event is None else target_event.value
                    ),
                )
                samples += 1
                split_counter[split] += 1
                if input_event is not None:
                    input_event_counter[input_event.value] += 1
                if target_event is not None:
                    target_event_counter[target_event.value] += 1

        return BuildReport(
            dataset_build_id=build_id,
            feature_schema_id=feature_schema,
            valid_temporal_windows=valid_temporal_windows,
            rejected_missing_frames=rejected_missing,
            rejected_outside_splits=rejected_split,
            spatial_windows=len(windows),
            samples=samples,
            samples_by_split=dict(split_counter),
            input_events=dict(input_event_counter),
            target_events=dict(target_event_counter),
        )

    def _primary_source_id(self) -> str:
        return next(
            source.source_id
            for source in self.config.sources
            if source.role is SourceRole.PRIMARY
        )

    def _validate_sources(self) -> None:
        configured = {source.source_id for source in self.config.sources}
        supplied = set(self.source_datasets)
        if configured != supplied:
            raise ValueError(
                "Prepared source datasets must exactly match configured source IDs."
            )
        if configured != set(self.source_checksums):
            raise ValueError(
                "Source checksums must exactly match configured source IDs."
            )
        if any(not value.strip() for value in self.source_checksums.values()):
            raise ValueError("Source checksums must not be empty.")

        primary = self.source_datasets[self._primary_source_id()]
        _validate_grid(primary, "primary source")
        primary_shape = (primary.sizes["y"], primary.sizes["x"])
        primary_times = _time_values(primary)

        for source in self.config.sources:
            dataset = self.source_datasets[source.source_id]
            _validate_grid(dataset, f"source {source.source_id!r}")
            if (dataset.sizes["y"], dataset.sizes["x"]) != primary_shape:
                raise ValueError(
                    "All prepared sources must use the primary spatial grid."
                )
            _validate_spatial_alignment(primary, dataset, source.source_id)

        dynamic_source_ids = {
            feature.source_id
            for feature in (
                *self.config.features.dynamic_inputs,
                *self.config.features.targets,
            )
        }
        for source_id in dynamic_source_ids:
            dataset = self.source_datasets[source_id]
            if _time_values(dataset) != primary_times:
                raise ValueError(
                    "All dynamic prepared sources must use the primary time grid "
                    "in 0.4."
                )

        for feature in (
            *self.config.features.dynamic_inputs,
            *self.config.features.static_inputs,
            *self.config.features.targets,
        ):
            dataset = self.source_datasets[feature.source_id]
            if feature.variable not in dataset:
                raise ValueError(
                    f"Feature {feature.name!r} references missing variable "
                    f"{feature.variable!r}."
                )


def dataset_build_id(
    config: MLDatasetConfig,
    source_checksums: Mapping[str, str],
) -> str:
    """Return a readable deterministic identity for one complete dataset build."""

    payload = {
        "config": config.to_resolved_dict(),
        "source_checksums": [
            {
                "source_id": source.source_id,
                "checksum": source_checksums[source.source_id],
            }
            for source in _identity_sources(config)
        ],
    }
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    slug = re.sub(r"[^a-z0-9]+", "-", config.name.lower()).strip("-")
    return f"{slug}-{digest[:_BUILD_HASH_HEX_LENGTH]}"


def feature_schema_id(config: MLDatasetConfig) -> str:
    """Return a deterministic identity for the ordered feature schema."""

    payload = _feature_schema_payload(config)
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return f"feature-schema-{digest[:12]}"


def sample_id(
    config: MLDatasetConfig,
    source_checksums: Mapping[str, str],
    *,
    input_times: Sequence[datetime],
    target_times: Sequence[datetime],
    window: SpatialWindow,
) -> str:
    """Return the readable prefix plus deterministic 64-bit SHA-256 prefix."""

    if not input_times or not target_times:
        raise ValueError("Sample identity requires non-empty input and target times.")
    payload = {
        "sample_schema_version": config.sample_schema_version,
        "sources": [
            {
                "source_id": source.source_id,
                "role": source.role.value,
                "artifact_id": source.artifact_id,
                "checksum": source_checksums[source.source_id],
            }
            for source in _identity_sources(config)
        ],
        "features": _feature_schema_payload(config),
        "temporal": {
            "step_minutes": config.temporal.step_minutes,
            "history_minutes": config.temporal.history_minutes,
            "forecast_minutes": config.temporal.forecast_minutes,
            "input_start": _utc_text(input_times[0]),
            "input_end": _utc_text(input_times[-1]),
            "target_start": _utc_text(target_times[0]),
            "target_end": _utc_text(target_times[-1]),
        },
        "spatial_window": {
            "x": window.x,
            "y": window.y,
            "width": window.width,
            "height": window.height,
            "mode": window.mode.value,
        },
    }
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    prefix = input_times[0].astimezone(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_x{window.x:04d}_y{window.y:04d}_{digest[:_HASH_HEX_LENGTH]}"


def classify_precipitation(
    statistics: PrecipitationStatistics | None,
    profile: PrecipEventProfileConfig,
) -> EventClass | None:
    """Classify precipitation using wet coverage plus p95 intensity."""

    if statistics is None:
        return None
    if statistics.wet_fraction < profile.min_wet_fraction:
        return EventClass.DRY
    for threshold in profile.thresholds:
        if statistics.p95 <= threshold.upper_p95_mm_per_h:
            return threshold.event_class
    return EventClass.EXTREME


def _assign_split(
    config: MLDatasetConfig,
    *,
    input_start: datetime,
    target_end: datetime,
) -> str | None:
    for interval in config.splits.intervals:
        if input_start >= interval.start and target_end < interval.end:
            return interval.name.value
    return None


def _spatial_windows(
    dataset: xr.Dataset,
    config: MLDatasetConfig,
) -> tuple[SpatialWindow, ...]:
    height = dataset.sizes["y"]
    width = dataset.sizes["x"]
    if config.spatial.mode is SpatialMode.FULL_FRAME:
        return (
            SpatialWindow(
                x=0,
                y=0,
                width=width,
                height=height,
                mode=SpatialWindowMode.FULL_FRAME,
            ),
        )

    patch_height = _required_int(config.spatial.patch_height, "patch_height")
    patch_width = _required_int(config.spatial.patch_width, "patch_width")
    stride_y = _required_int(config.spatial.stride_y, "stride_y")
    stride_x = _required_int(config.spatial.stride_x, "stride_x")
    if patch_height > height or patch_width > width:
        raise ValueError("Patch dimensions must fit inside the primary spatial grid.")

    return tuple(
        SpatialWindow(
            x=x,
            y=y,
            width=patch_width,
            height=patch_height,
            mode=SpatialWindowMode.PATCH,
        )
        for y in range(0, height - patch_height + 1, stride_y)
        for x in range(0, width - patch_width + 1, stride_x)
    )


def _precipitation_statistics(
    dataset: xr.Dataset,
    variable: str,
    time_indices: Sequence[int],
    window: SpatialWindow,
    wet_threshold: float,
) -> PrecipitationStatistics | None:
    values = np.asarray(
        dataset[variable]
        .isel(time=list(time_indices), y=window.y_slice, x=window.x_slice)
        .transpose("time", "y", "x")
        .values,
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return PrecipitationStatistics(
        mean=float(np.mean(finite)),
        maximum=float(np.max(finite)),
        p95=float(np.percentile(finite, 95)),
        wet_fraction=float(np.count_nonzero(finite >= wet_threshold) / finite.size),
    )


def _precipitation_variable(config: MLDatasetConfig, primary_source_id: str) -> str:
    candidates = [
        feature.variable
        for feature in config.features.dynamic_inputs
        if feature.source_id == primary_source_id
        and feature.variable == "precipitation_rate"
    ]
    if len(candidates) != 1:
        raise ValueError(
            "precip_event_v1 requires exactly one primary dynamic "
            "'precipitation_rate' feature."
        )
    return candidates[0]


def _validate_grid(dataset: xr.Dataset, name: str) -> None:
    missing = {dimension for dimension in ("y", "x") if dimension not in dataset.sizes}
    if missing:
        raise ValueError(f"{name} is missing spatial dimensions: {sorted(missing)!r}.")
    if dataset.sizes["y"] <= 0 or dataset.sizes["x"] <= 0:
        raise ValueError(f"{name} must not have empty spatial dimensions.")
    missing_coords = {
        coordinate for coordinate in ("y", "x") if coordinate not in dataset.coords
    }
    if missing_coords:
        raise ValueError(
            f"{name} is missing spatial coordinates: {sorted(missing_coords)!r}."
        )


def _validate_spatial_alignment(
    primary: xr.Dataset,
    candidate: xr.Dataset,
    source_id: str,
) -> None:
    for coordinate in ("y", "x"):
        if not np.array_equal(
            primary.coords[coordinate].values,
            candidate.coords[coordinate].values,
        ):
            raise ValueError(
                f"Prepared source {source_id!r} is not aligned to the primary "
                f"{coordinate!r} coordinate."
            )


def _time_values(dataset: xr.Dataset) -> tuple[datetime, ...]:
    if "time" not in dataset.coords:
        raise ValueError("Dynamic prepared source must expose a 'time' coordinate.")
    values = tuple(_to_utc(value) for value in dataset.coords["time"].values)
    if not values:
        raise ValueError("Prepared source time coordinate must not be empty.")
    if any(current >= following for current, following in pairwise(values)):
        raise ValueError("Prepared source timestamps must be strictly increasing.")
    return values


def _to_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("Prepared source contains NaT timestamp.")
        nanoseconds = int(value.astype("datetime64[ns]").astype(np.int64))
        return datetime.fromtimestamp(nanoseconds / 1_000_000_000, tz=UTC)
    raise TypeError(f"Unsupported prepared timestamp type: {type(value).__name__}.")


def _identity_sources(config: MLDatasetConfig) -> tuple[PreparedSourceConfig, ...]:
    primary = tuple(
        source for source in config.sources if source.role is SourceRole.PRIMARY
    )
    auxiliary = tuple(
        source for source in config.sources if source.role is SourceRole.AUXILIARY
    )
    return (*primary, *auxiliary)


def _feature_schema_payload(config: MLDatasetConfig) -> dict[str, list[dict[str, str]]]:
    def encode(features: Sequence[FeatureConfig]) -> list[dict[str, str]]:
        return [
            {
                "name": feature.name,
                "source_id": feature.source_id,
                "variable": feature.variable,
                "kind": feature.kind.value,
            }
            for feature in features
        ]

    return {
        "dynamic_inputs": encode(config.features.dynamic_inputs),
        "static_inputs": encode(config.features.static_inputs),
        "targets": encode(config.features.targets),
    }


def _event_profile_payload(profile: PrecipEventProfileConfig) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "wet_pixel_threshold_mm_per_h": profile.wet_pixel_threshold_mm_per_h,
        "min_wet_fraction": profile.min_wet_fraction,
        "thresholds": [
            {
                "event_class": threshold.event_class.value,
                "upper_p95_mm_per_h": threshold.upper_p95_mm_per_h,
            }
            for threshold in profile.thresholds
        ],
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Sample identity timestamps must be timezone-aware.")
    return value.astimezone(UTC).isoformat()


def _required_int(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required for patch sampling.")
    return value


def _stats_values(
    statistics: PrecipitationStatistics | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    if statistics is None:
        return None, None, None, None
    return (
        statistics.mean,
        statistics.maximum,
        statistics.p95,
        statistics.wet_fraction,
    )
