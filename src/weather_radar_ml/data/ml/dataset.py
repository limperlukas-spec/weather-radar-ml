"""Lazy framework-neutral dataset backed by SQLite sample references and Zarr."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import xarray as xr

from weather_radar_ml.config.ml_dataset import (
    FeatureConfig,
    FeatureKind,
    MLDatasetConfig,
)
from weather_radar_ml.data.ml.catalog import SampleRecord, SQLiteSampleCatalog
from weather_radar_ml.data.ml.domain import (
    RadarInputs,
    RadarSample,
    RadarTargets,
    SampleContext,
    SampleSelection,
    SpatialWindow,
)


class LazyRadarDataset:
    """Indexable lazy view over a deterministic catalog selection."""

    def __init__(
        self,
        *,
        catalog: SQLiteSampleCatalog,
        config: MLDatasetConfig,
        selection: SampleSelection | None = None,
        source_paths: Mapping[str, str | Path] | None = None,
    ) -> None:
        self._catalog = catalog
        self._config = config
        self._selection = selection or SampleSelection()
        self._records = catalog.query(self._selection)
        self._source_paths = {
            source_id: Path(path) for source_id, path in (source_paths or {}).items()
        }
        self._opened_pid: int | None = None
        self._sources: dict[str, xr.Dataset] = {}

        known_sources = {source.source_id for source in config.sources}
        unknown_overrides = set(self._source_paths) - known_sources
        if unknown_overrides:
            raise ValueError(
                f"Unknown source path overrides: {sorted(unknown_overrides)!r}."
            )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> RadarSample:
        return self._load(self._records[index])

    def get(self, sample_id: str) -> RadarSample:
        record = self._catalog.get(sample_id)
        if all(item.sample_id != sample_id for item in self._records):
            raise KeyError(sample_id)
        return self._load(record)

    def select(self, selection: SampleSelection) -> LazyRadarDataset:
        """Return a new view using the same catalog and source configuration."""
        return LazyRadarDataset(
            catalog=self._catalog,
            config=self._config,
            selection=selection,
            source_paths=self._source_paths,
        )

    def close(self) -> None:
        """Close all Zarr datasets opened by this process."""
        for dataset in self._sources.values():
            dataset.close()
        self._sources.clear()
        self._opened_pid = None

    def _load(self, record: SampleRecord) -> RadarSample:
        window_record = self._catalog.get_spatial_window(record.window_id)
        window = SpatialWindow(
            x=window_record.x,
            y=window_record.y,
            width=window_record.width,
            height=window_record.height,
            mode=window_record.mode,
        )
        input_times = _times(
            record.input_start,
            self._config.temporal.history_steps,
            self._config.temporal.step_minutes,
        )
        target_times = _times(
            record.target_start,
            self._config.temporal.forecast_steps,
            self._config.temporal.step_minutes,
        )

        dynamic = self._stack_dynamic(
            self._config.features.dynamic_inputs,
            input_times,
            window,
        )
        static = self._stack_static(self._config.features.static_inputs, window)
        targets = self._stack_dynamic(
            self._config.features.targets,
            target_times,
            window,
        )

        return RadarSample(
            inputs=RadarInputs(
                dynamic=dynamic,
                dynamic_features=tuple(
                    feature.name for feature in self._config.features.dynamic_inputs
                ),
                static=static,
                static_features=tuple(
                    feature.name for feature in self._config.features.static_inputs
                ),
            ),
            targets=RadarTargets(
                dynamic=targets,
                features=tuple(
                    feature.name for feature in self._config.features.targets
                ),
            ),
            context=SampleContext(
                sample_id=record.sample_id,
                dataset_id=record.dataset_build_id,
                input_times=input_times,
                target_times=target_times,
                spatial_window=window,
                metadata={
                    "split": record.split,
                    "input_event_class": record.input_event_class,
                    "target_event_class": record.target_event_class,
                    "feature_schema_id": record.feature_schema_id,
                },
            ),
        )

    def _stack_dynamic(
        self,
        features: tuple[FeatureConfig, ...],
        times: tuple[datetime, ...],
        window: SpatialWindow,
    ) -> np.ndarray:
        channels = [
            self._dynamic_feature(feature, times, window) for feature in features
        ]
        return np.stack(channels, axis=1)

    def _stack_static(
        self,
        features: tuple[FeatureConfig, ...],
        window: SpatialWindow,
    ) -> np.ndarray | None:
        if not features:
            return None
        channels = [self._static_feature(feature, window) for feature in features]
        return np.stack(channels, axis=0)

    def _dynamic_feature(
        self,
        feature: FeatureConfig,
        times: tuple[datetime, ...],
        window: SpatialWindow,
    ) -> np.ndarray:
        if feature.kind is not FeatureKind.DYNAMIC:
            raise ValueError(f"Feature {feature.name!r} is not dynamic.")
        data = self._source(feature.source_id)[feature.variable]
        _require_dims(data, ("time", "y", "x"), feature.name)
        selected = data.sel(time=_xarray_times(times)).isel(
            y=window.y_slice,
            x=window.x_slice,
        )
        values = np.asarray(selected.values)
        expected = (len(times), window.height, window.width)
        if values.shape != expected:
            raise ValueError(
                f"Feature {feature.name!r} produced shape {values.shape}, "
                f"expected {expected}."
            )
        return values

    def _static_feature(
        self,
        feature: FeatureConfig,
        window: SpatialWindow,
    ) -> np.ndarray:
        if feature.kind is not FeatureKind.STATIC:
            raise ValueError(f"Feature {feature.name!r} is not static.")
        data = self._source(feature.source_id)[feature.variable]
        _require_dims(data, ("y", "x"), feature.name)
        selected = data.isel(y=window.y_slice, x=window.x_slice)
        values = np.asarray(selected.values)
        expected = (window.height, window.width)
        if values.shape != expected:
            raise ValueError(
                f"Feature {feature.name!r} produced shape {values.shape}, "
                f"expected {expected}."
            )
        return values

    def _source(self, source_id: str) -> xr.Dataset:
        pid = os.getpid()
        if self._opened_pid != pid:
            self.close()
            self._opened_pid = pid

        cached = self._sources.get(source_id)
        if cached is not None:
            return cached

        source = next(
            (item for item in self._config.sources if item.source_id == source_id),
            None,
        )
        if source is None:
            raise KeyError(source_id)

        path = self._source_paths.get(source_id, source.path)
        if path is None:
            raise ValueError(f"No storage path configured for source {source_id!r}.")
        dataset = cast(xr.Dataset, xr.open_zarr(path, consolidated=False))
        self._sources[source_id] = dataset
        return dataset


def _times(
    start: datetime,
    count: int,
    step_minutes: int,
) -> tuple[datetime, ...]:
    step = timedelta(minutes=step_minutes)
    return tuple(start + index * step for index in range(count))


def _xarray_times(times: tuple[datetime, ...]) -> list[np.datetime64]:
    """Convert UTC-aware domain timestamps to xarray datetime64 labels."""
    return [np.datetime64(time.astimezone(UTC).replace(tzinfo=None)) for time in times]


def _require_dims(
    data: xr.DataArray,
    expected: tuple[str, ...],
    feature_name: str,
) -> None:
    if tuple(data.dims) != expected:
        raise ValueError(
            f"Feature {feature_name!r} must use dimensions {expected}, "
            f"got {tuple(data.dims)}."
        )
