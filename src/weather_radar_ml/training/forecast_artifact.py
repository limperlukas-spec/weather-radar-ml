"""Immutable Zarr artifacts for reference forecast predictions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from weather_radar_ml.models.domain import ModelSpec

FORECAST_ARTIFACT_FORMAT_VERSION = 1
Array = NDArray[Any]


@dataclass(frozen=True, slots=True)
class ForecastArtifactBatch:
    """One ordered prediction batch in physical precipitation units."""

    sample_ids: tuple[str, ...]
    learned_mm_h: Array
    persistence_mm_h: Array
    target_mm_h: Array
    valid_mask: Array

    def __post_init__(self) -> None:
        sample_ids = tuple(
            _require_name(value, "sample_id") for value in self.sample_ids
        )
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_ids must be unique within a forecast batch.")

        learned = np.asarray(self.learned_mm_h)
        persistence = np.asarray(self.persistence_mm_h)
        target = np.asarray(self.target_mm_h)
        valid = np.asarray(self.valid_mask)
        if learned.ndim != 5:
            raise ValueError("Forecast arrays must use shape [B, T, F, H, W].")
        if persistence.shape != learned.shape or target.shape != learned.shape:
            raise ValueError("Learned, persistence, and target shapes must match.")
        if valid.shape != learned.shape:
            raise ValueError("valid_mask must match the forecast tensor shape.")
        if valid.dtype != np.bool_:
            raise TypeError("valid_mask must use a boolean dtype.")
        if learned.shape[0] != len(sample_ids):
            raise ValueError("sample_ids must match the forecast batch dimension.")
        if not sample_ids:
            raise ValueError("ForecastArtifactBatch must contain at least one sample.")

        valid_values = valid.astype(bool, copy=False)
        if np.any(valid_values & ~np.isfinite(target)):
            raise ValueError("valid_mask marks a non-finite target value as valid.")
        if np.any(valid_values & (target < 0.0)):
            raise ValueError("Valid target precipitation must not be negative.")
        for name, values in (
            ("learned forecast", learned),
            ("persistence forecast", persistence),
        ):
            if np.any(valid_values & ~np.isfinite(values)):
                raise ValueError(f"{name} is non-finite on a valid target value.")
            if np.any(valid_values & (values < 0.0)):
                raise ValueError(f"{name} must not be negative on valid target values.")

        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "learned_mm_h", learned)
        object.__setattr__(self, "persistence_mm_h", persistence)
        object.__setattr__(self, "target_mm_h", target)
        object.__setattr__(self, "valid_mask", valid_values)


@dataclass(frozen=True, slots=True)
class ForecastArtifactResult:
    """Published immutable reference-forecast artifact paths and identity."""

    root: Path
    predictions: Path
    metadata: Path
    qualitative_samples: Path
    manifest: Path
    fingerprint: str
    sample_count: int
    split: str

    def __post_init__(self) -> None:
        for field_name in (
            "root",
            "predictions",
            "metadata",
            "qualitative_samples",
            "manifest",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        object.__setattr__(
            self,
            "fingerprint",
            _require_name(self.fingerprint, "fingerprint"),
        )
        object.__setattr__(self, "split", _require_name(self.split, "split"))
        if self.sample_count <= 0:
            raise ValueError("sample_count must be greater than zero.")


def write_forecast_artifact(
    runs_root: str | Path,
    *,
    artifact_id: str,
    spec: ModelSpec,
    lead_minutes: tuple[int, ...],
    batches: Iterable[ForecastArtifactBatch],
    split: str,
    provenance: Mapping[str, object],
    qualitative_count: int = 8,
) -> ForecastArtifactResult:
    """Atomically persist complete reference predictions as an xarray/Zarr artifact."""
    safe_id = _safe_segment(artifact_id, "artifact_id")
    normalized_split = _require_name(split, "split")
    leads = _validate_leads(lead_minutes, spec.forecast_steps)
    if qualitative_count <= 0:
        raise ValueError("qualitative_count must be greater than zero.")

    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / safe_id
    if destination.exists():
        raise FileExistsError(f"Forecast artifact already exists: {destination}")

    staging = Path(tempfile.mkdtemp(dir=root, prefix=f".{safe_id}.tmp-"))
    prediction_path = staging / "predictions.zarr"
    metadata_path = staging / "metadata.json"
    qualitative_path = staging / "qualitative-samples.json"
    manifest_path = staging / "manifest.json"

    sample_count = 0
    spatial_shape: tuple[int, int] | None = None
    seen_ids: set[str] = set()
    qualitative_candidates: list[tuple[float | None, str, int]] = []
    try:
        first = True
        for batch in batches:
            shape = tuple(int(value) for value in batch.learned_mm_h.shape)
            _, forecast_steps, target_features, height, width = shape
            if forecast_steps != spec.forecast_steps:
                raise ValueError(
                    "Forecast batch lead dimension does not match ModelSpec."
                )
            if target_features != len(spec.target_features):
                raise ValueError(
                    "Forecast batch feature dimension does not match ModelSpec."
                )
            if spatial_shape is None:
                spatial_shape = (height, width)
            elif spatial_shape != (height, width):
                raise ValueError("All forecast batches must share one spatial shape.")

            duplicates = seen_ids.intersection(batch.sample_ids)
            if duplicates:
                raise ValueError(
                    "Duplicate sample IDs across forecast batches: "
                    f"{sorted(duplicates)!r}."
                )
            seen_ids.update(batch.sample_ids)

            dataset = _batch_dataset(
                batch,
                spec=spec,
                lead_minutes=leads,
                sample_offset=sample_count,
            )
            encoding = _zarr_encoding(shape) if first else None
            if first:
                dataset.to_zarr(
                    prediction_path,
                    mode="w",
                    encoding=encoding,
                    consolidated=False,
                    zarr_format=2,
                )
                first = False
            else:
                dataset.to_zarr(
                    prediction_path,
                    mode="a",
                    append_dim="sample",
                    consolidated=False,
                    zarr_format=2,
                )

            for local_index, sample_id in enumerate(batch.sample_ids):
                mask = batch.valid_mask[local_index]
                target = batch.target_mm_h[local_index]
                target_max = float(np.max(target[mask])) if bool(np.any(mask)) else None
                qualitative_candidates.append(
                    (target_max, sample_id, sample_count + local_index)
                )
            sample_count += len(batch.sample_ids)

        if first or sample_count == 0 or spatial_shape is None:
            raise ValueError(
                "Cannot write a forecast artifact without prediction batches."
            )

        qualitative = _qualitative_payload(
            qualitative_candidates,
            limit=min(qualitative_count, sample_count),
        )
        _write_json(qualitative_path, qualitative)
        _write_json(
            metadata_path,
            _metadata_payload(
                artifact_id=safe_id,
                spec=spec,
                lead_minutes=leads,
                split=normalized_split,
                sample_count=sample_count,
                spatial_shape=spatial_shape,
                provenance=provenance,
            ),
        )

        records = {
            "predictions": _directory_record(prediction_path, staging),
            "metadata": _file_record(metadata_path, staging),
            "qualitative_samples": _file_record(qualitative_path, staging),
        }
        fingerprint = _artifact_fingerprint(records)
        _write_json(
            manifest_path,
            {
                "format_version": FORECAST_ARTIFACT_FORMAT_VERSION,
                "artifact_id": safe_id,
                "artifact_fingerprint": fingerprint,
                "split": normalized_split,
                "sample_count": sample_count,
                "files": records,
            },
        )

        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return ForecastArtifactResult(
        root=destination,
        predictions=destination / "predictions.zarr",
        metadata=destination / "metadata.json",
        qualitative_samples=destination / "qualitative-samples.json",
        manifest=destination / "manifest.json",
        fingerprint=fingerprint,
        sample_count=sample_count,
        split=normalized_split,
    )


def _batch_dataset(
    batch: ForecastArtifactBatch,
    *,
    spec: ModelSpec,
    lead_minutes: tuple[int, ...],
    sample_offset: int,
) -> xr.Dataset:
    batch_size, _, _, height, width = batch.learned_mm_h.shape
    sample_indices = np.arange(
        sample_offset,
        sample_offset + batch_size,
        dtype=np.int64,
    )
    coords: dict[str, object] = {
        "sample": sample_indices,
        "sample_id": ("sample", np.asarray(batch.sample_ids, dtype=str)),
        "lead": np.asarray(lead_minutes, dtype=np.int32),
        "feature": np.asarray(spec.target_features, dtype=str),
        "y": np.arange(height, dtype=np.int32),
        "x": np.arange(width, dtype=np.int32),
    }
    dimensions = ("sample", "lead", "feature", "y", "x")
    dataset = xr.Dataset(
        data_vars={
            "learned_mm_h": (
                dimensions,
                np.asarray(batch.learned_mm_h, dtype=np.float32),
            ),
            "persistence_mm_h": (
                dimensions,
                np.asarray(batch.persistence_mm_h, dtype=np.float32),
            ),
            "target_mm_h": (
                dimensions,
                np.asarray(batch.target_mm_h, dtype=np.float32),
            ),
            "valid_mask": (
                dimensions,
                np.asarray(batch.valid_mask, dtype=np.bool_),
            ),
        },
        coords=coords,
        attrs={
            "format_version": FORECAST_ARTIFACT_FORMAT_VERSION,
            "precipitation_units": "mm/h",
            "tensor_layout": "[sample, lead, feature, y, x]",
        },
    )
    for variable in ("learned_mm_h", "persistence_mm_h", "target_mm_h"):
        dataset[variable].attrs["units"] = "mm/h"
    dataset["valid_mask"].attrs["meaning"] = "valid target observation"
    return dataset


def _zarr_encoding(shape: tuple[int, ...]) -> dict[str, dict[str, object]]:
    _, _, features, height, width = shape
    chunks = (1, 1, features, min(64, height), min(64, width))
    return {
        "learned_mm_h": {"chunks": chunks},
        "persistence_mm_h": {"chunks": chunks},
        "target_mm_h": {"chunks": chunks},
        "valid_mask": {"chunks": chunks},
    }


def _metadata_payload(
    *,
    artifact_id: str,
    spec: ModelSpec,
    lead_minutes: tuple[int, ...],
    split: str,
    sample_count: int,
    spatial_shape: tuple[int, int],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    return {
        "format_version": FORECAST_ARTIFACT_FORMAT_VERSION,
        "artifact_id": artifact_id,
        "split": split,
        "sample_count": sample_count,
        "tensor_layout": ["sample", "lead", "feature", "y", "x"],
        "shape": [
            sample_count,
            spec.forecast_steps,
            len(spec.target_features),
            spatial_shape[0],
            spatial_shape[1],
        ],
        "lead_minutes": list(lead_minutes),
        "target_features": list(spec.target_features),
        "units": "mm/h",
        "variables": {
            "learned_mm_h": "learned forecast after expm1 and physical zero clamp",
            "persistence_mm_h": "last-observation persistence forecast",
            "target_mm_h": (
                "physical target precipitation; missing values remain non-finite"
            ),
            "valid_mask": "target-derived observation-validity mask",
        },
        "provenance": dict(provenance),
    }


def _qualitative_payload(
    candidates: list[tuple[float | None, str, int]],
    *,
    limit: int,
) -> dict[str, object]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            item[0] is None,
            -(item[0] if item[0] is not None else 0.0),
            item[1],
        ),
    )
    selected = ordered[:limit]
    return {
        "selection": "highest valid target precipitation, ties by sample_id",
        "samples": [
            {
                "sample_index": index,
                "sample_id": sample_id,
                "target_max_mm_h": target_max,
            }
            for target_max, sample_id, index in selected
        ],
    }


def _validate_leads(values: tuple[int, ...], expected: int) -> tuple[int, ...]:
    leads = tuple(int(value) for value in values)
    if len(leads) != expected:
        raise ValueError("lead_minutes must match ModelSpec.forecast_steps.")
    if any(value <= 0 for value in leads):
        raise ValueError("lead_minutes values must be greater than zero.")
    if any(current >= following for current, following in pairwise(leads)):
        raise ValueError("lead_minutes must be strictly increasing.")
    return leads


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _directory_record(path: Path, root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size_bytes = 0
    file_count = 0
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        size_bytes += item.stat().st_size
        file_count += 1
    if file_count == 0:
        raise ValueError(f"Directory artifact contains no files: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
        "file_count": file_count,
    }


def _artifact_fingerprint(records: Mapping[str, Mapping[str, object]]) -> str:
    payload = {
        key: {
            "sha256": value["sha256"],
            "size_bytes": value["size_bytes"],
        }
        for key, value in sorted(records.items())
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_segment(value: str, name: str) -> str:
    normalized = _require_name(value, name)
    if normalized in {".", ".."} or Path(normalized).name != normalized:
        raise ValueError(f"{name} must be a single safe path segment.")
    return normalized


def _require_name(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized
