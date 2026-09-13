"""Deterministic PyTorch batching for radar forecast training."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sized
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from weather_radar_ml.data.ml.torch_adapter import TorchSample
from weather_radar_ml.training.domain import ForecastBatch


def collate_forecast_batch(samples: list[TorchSample]) -> ForecastBatch:
    """Stack adapter samples into the common batched forecast tensor contract."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    dynamic = torch.stack(
        [_tensor_value(sample, "dynamic_inputs") for sample in samples], dim=0
    )
    target = torch.stack(
        [_tensor_value(sample, "targets") for sample in samples], dim=0
    )

    static_present = ["static_inputs" in sample for sample in samples]
    if any(static_present) and not all(static_present):
        raise ValueError("A batch cannot mix samples with and without static inputs.")
    static = (
        torch.stack(
            [_tensor_value(sample, "static_inputs") for sample in samples], dim=0
        )
        if all(static_present)
        else None
    )
    return ForecastBatch(dynamic=dynamic, target=target, static=static)


def make_forecast_dataloader(
    dataset: Dataset[TorchSample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> Iterable[ForecastBatch]:
    """Create a reproducible re-iterable DataLoader producing ForecastBatch values."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if num_workers < 0:
        raise ValueError("num_workers must not be negative.")
    if not isinstance(dataset, Sized):
        raise TypeError("dataset must provide a length.")
    if len(dataset) == 0:
        raise ValueError("Cannot build a DataLoader for an empty dataset.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_forecast_batch,
        generator=generator,
        worker_init_fn=_seed_worker if num_workers else None,
    )
    return cast(Iterable[ForecastBatch], loader)


def _tensor_value(sample: TorchSample, key: str) -> Tensor:
    value = sample.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"Torch sample field {key!r} must be a tensor.")
    return value


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
