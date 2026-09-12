import pytest
import torch
from torch import Tensor, nn

from weather_radar_ml.evaluation.continuous import ContinuousForecastMetrics
from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.training.domain import ForecastBatch
from weather_radar_ml.training.engine import ForecastTrainer
from weather_radar_ml.training.losses import MeanSquaredForecastLoss


def _spec() -> ModelSpec:
    return ModelSpec(
        name="toy",
        dynamic_input_features=("rain",),
        target_features=("rain",),
        history_steps=2,
        forecast_steps=1,
    )


class ToyForecast(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self._spec = spec
        self.scale = nn.Parameter(torch.tensor(0.0))

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def forward(self, dynamic: Tensor, static: Tensor | None = None) -> Tensor:
        del static
        return dynamic[:, -1:, :1] * self.scale


def _batch(value: float, target: float, *, size: int = 1) -> ForecastBatch:
    dynamic = torch.full((size, 2, 1, 1, 1), value)
    expected = torch.full((size, 1, 1, 1, 1), target)
    return ForecastBatch(dynamic=dynamic, target=expected)


def _trainer(*, lr: float = 0.1) -> tuple[ForecastTrainer, ToyForecast]:
    spec = _spec()
    model = ToyForecast(spec)
    trainer = ForecastTrainer(
        model=model,
        loss=MeanSquaredForecastLoss(spec),
        optimizer=torch.optim.SGD(model.parameters(), lr=lr),
        metric_factory=lambda: ContinuousForecastMetrics(spec),
    )
    return trainer, model


def test_train_epoch_updates_model_parameter() -> None:
    trainer, model = _trainer()
    before = float(model.scale.detach())

    result = trainer.run_train_epoch([_batch(2.0, 4.0)])

    assert float(model.scale.detach()) != before
    assert result.batches == 1
    assert result.samples == 1


def test_validation_epoch_does_not_update_model_parameter() -> None:
    trainer, model = _trainer()
    before = model.scale.detach().clone()

    result = trainer.run_validation_epoch([_batch(2.0, 4.0)])

    torch.testing.assert_close(model.scale.detach(), before)
    assert result.metrics.overall.count == 1


def test_epoch_loss_is_weighted_by_number_of_target_values() -> None:
    trainer, _ = _trainer(lr=0.0)

    result = trainer.run_validation_epoch(
        [_batch(1.0, 1.0, size=1), _batch(1.0, 3.0, size=3)]
    )

    assert result.loss == pytest.approx((1.0 + 3 * 9.0) / 4.0)
    assert result.samples == 4


def test_fit_returns_matching_train_and_validation_history() -> None:
    trainer, _ = _trainer()

    history = trainer.fit(
        [_batch(1.0, 1.0)],
        [_batch(1.0, 1.0)],
        epochs=2,
    )

    assert len(history.train) == 2
    assert len(history.validation) == 2


def test_epoch_rejects_empty_batch_iterable() -> None:
    trainer, _ = _trainer()

    with pytest.raises(ValueError, match="zero batches"):
        trainer.run_validation_epoch([])


def test_trainer_rejects_mismatching_loss_spec() -> None:
    spec = _spec()
    other = ModelSpec(
        name="other",
        dynamic_input_features=("rain",),
        target_features=("rain",),
        history_steps=2,
        forecast_steps=1,
    )
    model = ToyForecast(spec)

    with pytest.raises(ValueError, match="same ModelSpec"):
        ForecastTrainer(
            model=model,
            loss=MeanSquaredForecastLoss(other),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            metric_factory=lambda: ContinuousForecastMetrics(spec),
        )
