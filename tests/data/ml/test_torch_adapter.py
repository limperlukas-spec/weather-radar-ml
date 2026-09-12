from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from weather_radar_ml.data.ml.domain import (
    RadarInputs,
    RadarSample,
    RadarTargets,
    SampleContext,
    SpatialWindow,
)
from weather_radar_ml.data.ml.torch_adapter import TorchRadarDataset
from weather_radar_ml.data.ml.transforms import CastDType, InputTargetTransform


class FakeDataset:
    def __init__(self, sample: RadarSample) -> None:
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> RadarSample:
        if index != 0:
            raise IndexError(index)
        return self.sample

    def get(self, sample_id: str) -> RadarSample:
        if sample_id != self.sample.context.sample_id:
            raise KeyError(sample_id)
        return self.sample


def _sample(*, with_static: bool = True) -> RadarSample:
    dynamic = np.arange(2 * 2 * 2 * 3, dtype=np.float64).reshape(2, 2, 2, 3)
    targets = np.arange(2 * 1 * 2 * 3, dtype=np.float64).reshape(2, 1, 2, 3)
    static = (
        np.arange(2 * 2 * 3, dtype=np.float64).reshape(2, 2, 3) if with_static else None
    )
    return RadarSample(
        inputs=RadarInputs(
            dynamic=dynamic,
            dynamic_features=("rain", "temperature"),
            static=static,
            static_features=("height", "land_use") if with_static else (),
        ),
        targets=RadarTargets(
            dynamic=targets,
            features=("rain_target",),
        ),
        context=SampleContext(
            sample_id="sample",
            dataset_id="dataset",
            input_times=(
                datetime(2023, 1, 1, 0, 0),
                datetime(2023, 1, 1, 0, 5),
            ),
            target_times=(
                datetime(2023, 1, 1, 0, 10),
                datetime(2023, 1, 1, 0, 15),
            ),
            spatial_window=SpatialWindow(x=0, y=0, width=3, height=2),
        ),
    )


def test_adapter_returns_expected_tensors() -> None:
    adapter = TorchRadarDataset(FakeDataset(_sample()))
    item = adapter[0]

    assert isinstance(item["dynamic_inputs"], torch.Tensor)
    assert isinstance(item["static_inputs"], torch.Tensor)
    assert isinstance(item["targets"], torch.Tensor)
    assert item["dynamic_inputs"].shape == (2, 2, 2, 3)
    assert item["static_inputs"].shape == (2, 2, 3)
    assert item["targets"].shape == (2, 1, 2, 3)
    assert item["sample_id"] == "sample"


def test_adapter_omits_static_key_when_no_static_inputs_exist() -> None:
    item = TorchRadarDataset(FakeDataset(_sample(with_static=False)))[0]
    assert "static_inputs" not in item


def test_adapter_applies_transform_before_conversion() -> None:
    transform = InputTargetTransform(
        dynamic_inputs=(CastDType(np.float32),),
        targets=(CastDType(np.float32),),
    )
    item = TorchRadarDataset(FakeDataset(_sample()), transform=transform)[0]

    assert item["dynamic_inputs"].dtype is torch.float32
    assert item["targets"].dtype is torch.float32


def test_tensor_storage_does_not_alias_domain_arrays() -> None:
    sample = _sample()
    item = TorchRadarDataset(FakeDataset(sample))[0]
    dynamic = item["dynamic_inputs"]
    assert isinstance(dynamic, torch.Tensor)
    dynamic[0, 0, 0, 0] = -999.0
    assert sample.inputs.dynamic[0, 0, 0, 0] != -999.0


def test_get_uses_stable_sample_id() -> None:
    item = TorchRadarDataset(FakeDataset(_sample())).get("sample")
    assert item["sample_id"] == "sample"


def test_default_dataloader_collates_adapter_output() -> None:
    adapter = TorchRadarDataset(FakeDataset(_sample()))
    batch = next(iter(DataLoader(adapter, batch_size=1)))

    assert batch["dynamic_inputs"].shape == (1, 2, 2, 2, 3)
    assert batch["static_inputs"].shape == (1, 2, 2, 3)
    assert batch["targets"].shape == (1, 2, 1, 2, 3)
    assert batch["sample_id"] == ["sample"]
