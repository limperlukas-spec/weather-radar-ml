"""PyTorch adapter for framework-neutral radar datasets."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from weather_radar_ml.data.ml.contracts import RadarDatasetProtocol, SampleTransform
from weather_radar_ml.data.ml.domain import RadarSample

TorchSampleValue = Tensor | str
TorchSample = dict[str, TorchSampleValue]


class TorchRadarDataset(Dataset[TorchSample]):
    """Expose a framework-neutral radar dataset as PyTorch tensors."""

    def __init__(
        self,
        dataset: RadarDatasetProtocol,
        *,
        transform: SampleTransform | None = None,
    ) -> None:
        self._dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> TorchSample:
        return self._convert(self._dataset[index])

    def get(self, sample_id: str) -> TorchSample:
        """Return one sample by stable ID using the wrapped dataset."""
        return self._convert(self._dataset.get(sample_id))

    def _convert(self, sample: RadarSample) -> TorchSample:
        if self._transform is not None:
            sample = self._transform.transform(sample)

        item: TorchSample = {
            "dynamic_inputs": _to_tensor(sample.inputs.dynamic),
            "targets": _to_tensor(sample.targets.dynamic),
            "sample_id": sample.context.sample_id,
        }
        if sample.inputs.static is not None:
            item["static_inputs"] = _to_tensor(sample.inputs.static)
        return item


def _to_tensor(array: np.ndarray) -> Tensor:
    """Copy a read-only domain array into writable PyTorch-owned storage."""
    return torch.from_numpy(np.array(array, copy=True))
