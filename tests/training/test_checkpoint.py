import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)


def _spec(name: str = "toy") -> ModelSpec:
    return ModelSpec(
        name=name,
        dynamic_input_features=("rain",),
        target_features=("rain",),
        history_steps=2,
        forecast_steps=1,
    )


class ToyForecast(nn.Module):
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


def _trained_pair() -> tuple[ToyForecast, torch.optim.Adam]:
    model = ToyForecast(_spec())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = (model.scale - 4.0).square()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def test_checkpoint_restores_model_optimizer_and_epoch(tmp_path: Path) -> None:
    source_model, source_optimizer = _trained_pair()
    path = tmp_path / "resume.pt"
    save_training_checkpoint(
        path,
        model=source_model,
        optimizer=source_optimizer,
        completed_epochs=7,
    )

    restored_model = ToyForecast(_spec())
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.5)
    state = load_training_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
    )

    assert state.completed_epochs == 7
    torch.testing.assert_close(restored_model.scale, source_model.scale)
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    source_state = next(iter(source_optimizer.state.values()))
    restored_state = next(iter(restored_optimizer.state.values()))
    torch.testing.assert_close(restored_state["exp_avg"], source_state["exp_avg"])
    torch.testing.assert_close(restored_state["exp_avg_sq"], source_state["exp_avg_sq"])


def test_checkpoint_restores_python_numpy_and_torch_rng(tmp_path: Path) -> None:
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    model, optimizer = _trained_pair()
    path = tmp_path / "rng.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        completed_epochs=1,
    )

    expected = (random.random(), float(np.random.random()), torch.rand(3))
    random.random()
    np.random.random()
    torch.rand(3)

    load_training_checkpoint(path, model=model, optimizer=optimizer)
    actual = (random.random(), float(np.random.random()), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_checkpoint_can_skip_rng_restore(tmp_path: Path) -> None:
    random.seed(23)
    model, optimizer = _trained_pair()
    path = tmp_path / "rng-skip.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        completed_epochs=1,
    )

    random.seed(99)
    expected = random.random()
    random.seed(99)
    load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        restore_rng_state=False,
    )

    assert random.random() == expected


def test_checkpoint_rejects_mismatching_model_spec(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "spec.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        completed_epochs=2,
    )

    other_model = ToyForecast(_spec("different"))
    other_optimizer = torch.optim.Adam(other_model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="model_spec mismatch"):
        load_training_checkpoint(
            path,
            model=other_model,
            optimizer=other_optimizer,
        )


def test_checkpoint_rejects_mismatching_optimizer_type(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "optimizer.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        completed_epochs=2,
    )

    other_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="optimizer_type mismatch"):
        load_training_checkpoint(path, model=model, optimizer=other_optimizer)


def test_checkpoint_rejects_negative_completed_epochs(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()

    with pytest.raises(ValueError, match="must not be negative"):
        save_training_checkpoint(
            tmp_path / "invalid.pt",
            model=model,
            optimizer=optimizer,
            completed_epochs=-1,
        )


def test_checkpoint_save_replaces_existing_file_atomically(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"old")

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        completed_epochs=3,
    )

    assert path.stat().st_size > 3
    assert not list(tmp_path.glob(".checkpoint.pt.*.tmp"))
