"""Tests for fair learned-versus-persistence research evaluation."""

import pytest
import torch

from weather_radar_ml.evaluation.research import ResearchForecastEvaluator
from weather_radar_ml.models.domain import ModelSpec


def _spec(*, forecast_steps: int = 2) -> ModelSpec:
    return ModelSpec(
        name="unet",
        dynamic_input_features=("rain",),
        target_features=("rain",),
        history_steps=2,
        forecast_steps=forecast_steps,
    )


def _log_prediction(values: torch.Tensor) -> torch.Tensor:
    return torch.log1p(values)


def test_research_evaluation_reports_physical_per_lead_metrics_and_skill() -> None:
    evaluator = ResearchForecastEvaluator(_spec())
    dynamic = torch.tensor([[[[[1.0]]], [[[2.0]]]]])
    target = torch.tensor([[[[[3.0]]], [[[5.0]]]]])
    learned = _log_prediction(torch.tensor([[[[[2.5]]], [[[4.0]]]]]))

    evaluator.update(
        dynamic_inputs=dynamic,
        learned_prediction_log1p=learned,
        target=target,
    )
    report = evaluator.compute()

    assert report.all.learned.overall.mae == pytest.approx(0.75)
    assert report.all.persistence.overall.mae == pytest.approx(2.0)
    assert report.all.learned.per_lead[0].rmse == pytest.approx(0.5)
    assert report.all.learned.per_lead[1].rmse == pytest.approx(1.0)
    assert report.all.skill.overall.mae == pytest.approx(0.625)
    assert report.all.learned.lead_minutes == (5, 10)


def test_categorical_scores_use_contingency_counts_and_none_for_zero_denominator() -> (
    None
):
    evaluator = ResearchForecastEvaluator(_spec(forecast_steps=1))
    dynamic = torch.tensor(
        [
            [[[[0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0]]]],
        ]
    )
    target = torch.tensor([[[[[2.0, 2.0, 0.0, 0.0]]]]])
    learned = _log_prediction(torch.tensor([[[[[2.0, 0.0, 2.0, 0.0]]]]]))

    evaluator.update(
        dynamic_inputs=dynamic,
        learned_prediction_log1p=learned,
        target=target,
    )
    score = evaluator.compute().all.learned.overall.categorical[1]

    assert score.threshold_mm_h == 1.0
    assert (score.hits, score.false_alarms, score.misses) == (1, 1, 1)
    assert score.csi == pytest.approx(1 / 3)
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(0.5)

    high = evaluator.compute().all.persistence.overall.categorical[2]
    assert high.csi is None
    assert high.precision is None
    assert high.recall is None


def test_wet_target_condition_selects_samples_not_only_wet_pixels() -> None:
    evaluator = ResearchForecastEvaluator(_spec(forecast_steps=1))
    dynamic = torch.zeros(2, 2, 1, 1, 2)
    target = torch.tensor(
        [
            [[[[1.0, 0.0]]]],
            [[[[0.0, 0.0]]]],
        ]
    )
    learned = _log_prediction(target.clone())

    evaluator.update(
        dynamic_inputs=dynamic,
        learned_prediction_log1p=learned,
        target=target,
    )
    report = evaluator.compute()

    assert report.all.learned.overall.count == 4
    assert report.wet_target is not None
    assert report.wet_target.learned.overall.count == 2


def test_wet_target_report_is_none_when_dataset_contains_no_wet_sample() -> None:
    evaluator = ResearchForecastEvaluator(_spec(forecast_steps=1))
    dynamic = torch.zeros(1, 2, 1, 1, 2)
    target = torch.zeros(1, 1, 1, 1, 2)

    evaluator.update(
        dynamic_inputs=dynamic,
        learned_prediction_log1p=torch.zeros_like(target),
        target=target,
    )

    assert evaluator.compute().wet_target is None


def test_target_nan_is_excluded_from_one_common_mask_for_both_forecasts() -> None:
    evaluator = ResearchForecastEvaluator(_spec(forecast_steps=1))
    dynamic = torch.ones(1, 2, 1, 1, 2)
    target = torch.tensor([[[[[2.0, float("nan")]]]]])
    learned = _log_prediction(torch.tensor([[[[[2.0, 99.0]]]]]))

    evaluator.update(
        dynamic_inputs=dynamic,
        learned_prediction_log1p=learned,
        target=target,
    )
    report = evaluator.compute()

    assert report.all.learned.overall.count == 1
    assert report.all.persistence.overall.count == 1
    assert report.all.learned.overall.mae == pytest.approx(0.0)
    assert report.all.persistence.overall.mae == pytest.approx(1.0)


def test_explicit_validity_mask_is_shared_by_learned_and_persistence() -> None:
    evaluator = ResearchForecastEvaluator(_spec(forecast_steps=1))
    dynamic = torch.ones(1, 2, 1, 1, 2)
    target = torch.tensor([[[[[2.0, 10.0]]]]])
    learned = _log_prediction(torch.tensor([[[[[2.0, 100.0]]]]]))
    mask = torch.tensor([[[[[True, False]]]]])

    evaluator.update(
        dynamic_inputs=dynamic,
        learned_prediction_log1p=learned,
        target=target,
        valid_mask=mask,
    )
    report = evaluator.compute()

    assert report.all.learned.overall.count == 1
    assert report.all.persistence.overall.count == 1
    assert report.all.learned.overall.mae == pytest.approx(0.0)


def test_negative_learned_physical_prediction_is_clamped_for_evaluation() -> None:
    evaluator = ResearchForecastEvaluator(_spec(forecast_steps=1))
    dynamic = torch.zeros(1, 2, 1, 1, 1)
    target = torch.zeros(1, 1, 1, 1, 1)

    evaluator.update(
        dynamic_inputs=dynamic,
        learned_prediction_log1p=torch.tensor([[[[[-0.5]]]]]),
        target=target,
    )

    assert evaluator.compute().all.learned.overall.mae == pytest.approx(0.0)


def test_skill_is_none_when_persistence_error_is_zero() -> None:
    evaluator = ResearchForecastEvaluator(_spec(forecast_steps=1))
    dynamic = torch.ones(1, 2, 1, 1, 1)
    target = torch.ones(1, 1, 1, 1, 1)

    evaluator.update(
        dynamic_inputs=dynamic,
        learned_prediction_log1p=_log_prediction(target),
        target=target,
    )
    skill = evaluator.compute().all.skill.overall

    assert skill.mae is None
    assert skill.rmse is None


def test_batch_partition_does_not_change_research_metrics() -> None:
    spec = _spec(forecast_steps=1)
    dynamic = torch.tensor(
        [
            [[[[1.0]]], [[[2.0]]]],
            [[[[2.0]]], [[[3.0]]]],
        ]
    )
    target = torch.tensor([[[[[3.0]]]], [[[[5.0]]]]])
    learned = _log_prediction(torch.tensor([[[[[2.5]]]], [[[[4.0]]]]]))

    together = ResearchForecastEvaluator(spec)
    together.update(
        dynamic_inputs=dynamic,
        learned_prediction_log1p=learned,
        target=target,
    )

    split = ResearchForecastEvaluator(spec)
    for index in range(2):
        split.update(
            dynamic_inputs=dynamic[index : index + 1],
            learned_prediction_log1p=learned[index : index + 1],
            target=target[index : index + 1],
        )

    assert split.compute() == together.compute()


def test_non_finite_learned_prediction_is_rejected_not_masked_away() -> None:
    evaluator = ResearchForecastEvaluator(_spec(forecast_steps=1))
    dynamic = torch.zeros(1, 2, 1, 1, 1)
    target = torch.zeros(1, 1, 1, 1, 1)

    with pytest.raises(ValueError, match="only finite"):
        evaluator.update(
            dynamic_inputs=dynamic,
            learned_prediction_log1p=torch.full_like(target, float("nan")),
            target=target,
        )


def test_explicit_mask_cannot_mark_non_finite_target_as_valid() -> None:
    evaluator = ResearchForecastEvaluator(_spec(forecast_steps=1))
    dynamic = torch.zeros(1, 2, 1, 1, 1)
    target = torch.full((1, 1, 1, 1, 1), float("nan"))

    with pytest.raises(ValueError, match="marks non-finite"):
        evaluator.update(
            dynamic_inputs=dynamic,
            learned_prediction_log1p=torch.zeros_like(target),
            target=target,
            valid_mask=torch.ones_like(target, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"thresholds_mm_h": ()}, "At least one"),
        ({"thresholds_mm_h": (1.0, 0.1)}, "strictly increasing"),
        ({"wet_threshold_mm_h": 0.0}, "wet_threshold"),
        ({"lead_step_minutes": 0}, "lead_step_minutes"),
    ],
)
def test_evaluator_rejects_invalid_research_policy(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ResearchForecastEvaluator(_spec(), **kwargs)  # type: ignore[arg-type]


def test_evaluator_supports_explicit_persistence_feature_mapping() -> None:
    spec = ModelSpec(
        name="unet",
        dynamic_input_features=("rain_input",),
        target_features=("rain_target",),
        history_steps=2,
        forecast_steps=1,
    )
    evaluator = ResearchForecastEvaluator(
        spec,
        persistence_input_features=("rain_input",),
    )
    dynamic = torch.tensor([[[[[1.0]]], [[[2.0]]]]])
    target = torch.tensor([[[[[3.0]]]]])

    evaluator.update(
        dynamic_inputs=dynamic,
        learned_prediction_log1p=_log_prediction(target),
        target=target,
    )

    assert evaluator.compute().all.persistence.overall.mae == pytest.approx(1.0)
