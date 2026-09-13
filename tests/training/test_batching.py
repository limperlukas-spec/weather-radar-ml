import pytest
import torch
from torch.utils.data import Dataset

from weather_radar_ml.data.ml.torch_adapter import TorchSample
from weather_radar_ml.training.batching import (
    collate_forecast_batch,
    make_forecast_dataloader,
)


class TinyDataset(Dataset[TorchSample]):
    def __init__(self, *, static: bool = False) -> None:
        self.static = static

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> TorchSample:
        item: TorchSample = {
            "dynamic_inputs": torch.full((2, 1, 2, 2), float(index)),
            "targets": torch.full((1, 1, 2, 2), float(index)),
            "sample_id": str(index),
        }
        if self.static:
            item["static_inputs"] = torch.full((1, 2, 2), float(index))
        return item


def test_collate_forecast_batch_stacks_expected_layout() -> None:
    dataset = TinyDataset(static=True)

    batch = collate_forecast_batch([dataset[0], dataset[1]])

    assert batch.dynamic.shape == (2, 2, 1, 2, 2)
    assert batch.target.shape == (2, 1, 1, 2, 2)
    assert batch.static is not None
    assert batch.static.shape == (2, 1, 2, 2)


def test_collate_rejects_mixed_static_presence() -> None:
    with pytest.raises(ValueError, match="cannot mix"):
        collate_forecast_batch([TinyDataset(static=True)[0], TinyDataset()[1]])


def test_dataloader_is_reiterable_and_batches_all_samples() -> None:
    loader = make_forecast_dataloader(
        TinyDataset(),
        batch_size=2,
        shuffle=False,
        seed=11,
    )

    first = list(loader)
    second = list(loader)

    assert [batch.batch_size for batch in first] == [2, 1]
    assert [batch.batch_size for batch in second] == [2, 1]
    torch.testing.assert_close(first[0].dynamic, second[0].dynamic)


def test_dataloader_rejects_empty_dataset() -> None:
    class EmptyDataset(TinyDataset):
        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="empty dataset"):
        make_forecast_dataloader(
            EmptyDataset(),
            batch_size=1,
            shuffle=False,
            seed=1,
        )
