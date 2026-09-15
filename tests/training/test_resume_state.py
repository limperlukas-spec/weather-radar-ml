from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from weather_radar_ml.data.ml.torch_adapter import TorchSample
from weather_radar_ml.evaluation.continuous import ContinuousForecastMetrics
from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.training.batching import (
    forecast_dataloader_generator_state,
    make_forecast_dataloader,
    restore_forecast_dataloader_generator_state,
)
from weather_radar_ml.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from weather_radar_ml.training.domain import ForecastBatch
from weather_radar_ml.training.engine import ForecastTrainer
from weather_radar_ml.training.losses import MeanSquaredForecastLoss


class _Dataset(torch.utils.data.Dataset[TorchSample]):
    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> TorchSample:
        value = torch.tensor([[[[float(index)]]], [[[float(index)]]]])
        target = torch.tensor([[[[float(index)]]]])
        return {
            "dynamic_inputs": value,
            "targets": target,
            "sample_id": str(index),
        }


class _Toy(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self._spec = spec
        self.scale = nn.Parameter(torch.tensor(1.0))

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def forward(self, dynamic: Tensor, static: Tensor | None = None) -> Tensor:
        del static
        return dynamic[:, -1:, :1] * self.scale


def _spec() -> ModelSpec:
    return ModelSpec(
        name="toy",
        dynamic_input_features=("rain",),
        target_features=("rain",),
        history_steps=2,
        forecast_steps=1,
    )


def test_dataloader_generator_state_round_trips() -> None:
    first = make_forecast_dataloader(_Dataset(), batch_size=2, shuffle=True, seed=17)
    second = make_forecast_dataloader(_Dataset(), batch_size=2, shuffle=True, seed=17)

    list(first)
    state = forecast_dataloader_generator_state(first)
    restore_forecast_dataloader_generator_state(second, state)

    assert torch.equal(
        forecast_dataloader_generator_state(first),
        forecast_dataloader_generator_state(second),
    )
    first_next = [batch.dynamic.clone() for batch in first]
    second_next = [batch.dynamic.clone() for batch in second]
    assert len(first_next) == len(second_next)
    for actual, expected in zip(first_next, second_next, strict=True):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_round_trips_history_and_loader_state(tmp_path: Path) -> None:
    spec = _spec()
    model = _Toy(spec)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = ForecastTrainer(
        model=model,
        loss=MeanSquaredForecastLoss(spec),
        optimizer=optimizer,
        metric_factory=lambda: ContinuousForecastMetrics(spec),
    )
    batch = ForecastBatch(
        dynamic=torch.ones((1, 2, 1, 1, 1)),
        target=torch.ones((1, 1, 1, 1, 1)),
    )
    history = trainer.fit([batch], [batch], epochs=1)
    loader = make_forecast_dataloader(_Dataset(), batch_size=2, shuffle=True, seed=17)
    list(loader)
    generator_state = forecast_dataloader_generator_state(loader)
    path = tmp_path / "resume.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        completed_epochs=1,
        history=history,
        dataloader_generator_state=generator_state,
    )
    state = load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        restore_rng_state=False,
    )

    assert state.history == history
    assert state.dataloader_generator_state is not None
    assert torch.equal(state.dataloader_generator_state, generator_state)


def _saved_resume_checkpoint(
    tmp_path: Path,
) -> tuple[Path, _Toy, torch.optim.Optimizer, object]:
    spec = _spec()
    model = _Toy(spec)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = ForecastTrainer(
        model=model,
        loss=MeanSquaredForecastLoss(spec),
        optimizer=optimizer,
        metric_factory=lambda: ContinuousForecastMetrics(spec),
    )
    batch = ForecastBatch(
        dynamic=torch.ones((1, 2, 1, 1, 1)),
        target=torch.ones((1, 1, 1, 1, 1)),
    )
    history = trainer.fit([batch], [batch], epochs=1)
    path = tmp_path / "resume.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        completed_epochs=1,
        history=history,
    )
    return path, model, optimizer, history


def _rewrite_checkpoint(path: Path, key: str, value: object) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    payload[key] = value
    torch.save(payload, path)


def test_dataloader_generator_state_rejects_invalid_inputs() -> None:
    loader = make_forecast_dataloader(_Dataset(), batch_size=2, shuffle=True, seed=17)
    invalid_state = torch.zeros((1, 1), dtype=torch.uint8)
    with pytest.raises(ValueError, match="one-dimensional uint8"):
        restore_forecast_dataloader_generator_state(loader, invalid_state)

    with pytest.raises(TypeError, match="torch DataLoader"):
        forecast_dataloader_generator_state([])

    raw_loader = torch.utils.data.DataLoader(_Dataset(), batch_size=2)
    with pytest.raises(ValueError, match="no deterministic generator"):
        forecast_dataloader_generator_state(raw_loader)


def test_checkpoint_rejects_history_epoch_mismatch_on_save(tmp_path: Path) -> None:
    path, model, optimizer, history = _saved_resume_checkpoint(tmp_path)
    path.unlink()

    with pytest.raises(ValueError, match="history must match completed_epochs"):
        save_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            completed_epochs=2,
            history=history,
        )


def test_checkpoint_rejects_non_forecast_model(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(TypeError, match="ForecastModel protocol"):
        save_training_checkpoint(
            tmp_path / "invalid.pt",
            model=model,
            optimizer=optimizer,
            completed_epochs=0,
        )


def test_checkpoint_rejects_invalid_loader_state_payload(tmp_path: Path) -> None:
    path, model, optimizer, _ = _saved_resume_checkpoint(tmp_path)
    _rewrite_checkpoint(path, "dataloader_generator_state", "invalid")

    with pytest.raises(ValueError, match="dataloader_generator_state must be a tensor"):
        load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            restore_rng_state=False,
        )


def test_checkpoint_rejects_loaded_history_epoch_mismatch(tmp_path: Path) -> None:
    path, model, optimizer, _ = _saved_resume_checkpoint(tmp_path)
    _rewrite_checkpoint(path, "completed_epochs", 2)

    with pytest.raises(ValueError, match="history does not match completed_epochs"):
        load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            restore_rng_state=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("best_validation_loss", True, "best_validation_loss must be numeric"),
        ("stopped_early", "no", "stopped_early must be boolean"),
    ],
)
def test_checkpoint_rejects_invalid_history_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path, _, _, _ = _saved_resume_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    history = payload["training_history"]
    assert isinstance(history, dict)
    history[field] = value
    torch.save(payload, path)
