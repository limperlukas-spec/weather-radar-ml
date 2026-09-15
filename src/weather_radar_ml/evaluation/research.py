"""Research-grade learned-versus-persistence precipitation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import isfinite, sqrt

import torch
from torch import Tensor

from weather_radar_ml.models.baseline import PersistenceForecast
from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.models.validation import (
    validate_forecast_batch,
    validate_prediction,
)
from weather_radar_ml.transforms.precipitation import inverse_log1p_precipitation


@dataclass(frozen=True, slots=True)
class CategoricalScore:
    """Contingency-table score for one precipitation threshold."""

    threshold_mm_h: float
    hits: int
    false_alarms: int
    misses: int
    csi: float | None
    precision: float | None
    recall: float | None

    def __post_init__(self) -> None:
        if not isfinite(self.threshold_mm_h) or self.threshold_mm_h <= 0.0:
            raise ValueError("threshold_mm_h must be finite and greater than zero.")
        if min(self.hits, self.false_alarms, self.misses) < 0:
            raise ValueError("Contingency counts must not be negative.")
        for name, value in (
            ("csi", self.csi),
            ("precision", self.precision),
            ("recall", self.recall),
        ):
            if value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be None or a finite value in [0, 1].")


@dataclass(frozen=True, slots=True)
class ResearchMetricSummary:
    """Continuous and categorical scores over one common validity mask."""

    count: int
    mae: float
    rmse: float
    categorical: tuple[CategoricalScore, ...]

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("ResearchMetricSummary count must be greater than zero.")
        if not isfinite(self.mae) or self.mae < 0.0:
            raise ValueError("mae must be finite and non-negative.")
        if not isfinite(self.rmse) or self.rmse < 0.0:
            raise ValueError("rmse must be finite and non-negative.")
        if not self.categorical:
            raise ValueError("At least one categorical threshold is required.")


@dataclass(frozen=True, slots=True)
class ResearchForecastReport:
    """Overall and lead-resolved physical precipitation metrics."""

    overall: ResearchMetricSummary
    per_lead: tuple[ResearchMetricSummary, ...]
    lead_minutes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.per_lead:
            raise ValueError("ResearchForecastReport requires per-lead metrics.")
        if len(self.per_lead) != len(self.lead_minutes):
            raise ValueError("per_lead and lead_minutes must have equal length.")
        if any(value <= 0 for value in self.lead_minutes):
            raise ValueError("lead_minutes values must be greater than zero.")


@dataclass(frozen=True, slots=True)
class ErrorSkillSummary:
    """Persistence-relative error skill; None means an undefined denominator."""

    mae: float | None
    rmse: float | None

    def __post_init__(self) -> None:
        for name, value in (("mae", self.mae), ("rmse", self.rmse)):
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} skill must be None or finite.")


@dataclass(frozen=True, slots=True)
class ResearchSkillReport:
    """Overall and lead-resolved persistence-relative error skill."""

    overall: ErrorSkillSummary
    per_lead: tuple[ErrorSkillSummary, ...]
    lead_minutes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.per_lead:
            raise ValueError("ResearchSkillReport requires per-lead skill.")
        if len(self.per_lead) != len(self.lead_minutes):
            raise ValueError("per_lead and lead_minutes must have equal length.")


@dataclass(frozen=True, slots=True)
class ConditionComparison:
    """Learned and persistence results evaluated on exactly the same values."""

    learned: ResearchForecastReport
    persistence: ResearchForecastReport
    skill: ResearchSkillReport

    def __post_init__(self) -> None:
        if self.learned.overall.count != self.persistence.overall.count:
            raise ValueError("Learned and persistence counts must match.")
        if self.learned.lead_minutes != self.persistence.lead_minutes:
            raise ValueError("Learned and persistence lead times must match.")
        if self.learned.lead_minutes != self.skill.lead_minutes:
            raise ValueError("Forecast and skill lead times must match.")


@dataclass(frozen=True, slots=True)
class ResearchEvaluationReport:
    """Complete all-sample and wet-target learned-versus-persistence report."""

    all: ConditionComparison
    wet_target: ConditionComparison | None
    thresholds_mm_h: tuple[float, ...]
    wet_threshold_mm_h: float


class ResearchForecastEvaluator:
    """Accumulate fair learned-versus-persistence precipitation comparisons.

    Learned predictions are expected in log1p model space. They are transformed
    back to mm/h and clamped to zero only for research evaluation. Persistence is
    generated from the same physical input batch, and both forecasts are scored
    against the same target validity mask.

    A wet-target sample is a sample with at least one valid target value at or
    above ``wet_threshold_mm_h`` anywhere in its forecast horizon. Once selected,
    all valid target values from that sample participate in the wet-target report.

    CSI, precision, and recall are reported as ``None`` when their mathematical
    denominator is zero. Error skill is ``1 - error_model / error_persistence``;
    it is ``None`` when persistence error is zero.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        thresholds_mm_h: tuple[float, ...] = (0.1, 1.0, 5.0),
        wet_threshold_mm_h: float = 0.1,
        lead_step_minutes: int = 5,
        persistence_input_features: tuple[str, ...] | None = None,
    ) -> None:
        self._spec = spec
        self._thresholds = _validate_thresholds(thresholds_mm_h)
        if not isfinite(wet_threshold_mm_h) or wet_threshold_mm_h <= 0.0:
            raise ValueError("wet_threshold_mm_h must be finite and greater than zero.")
        if lead_step_minutes <= 0:
            raise ValueError("lead_step_minutes must be greater than zero.")
        self._wet_threshold = float(wet_threshold_mm_h)
        self._lead_minutes = tuple(
            lead_step_minutes * (index + 1) for index in range(spec.forecast_steps)
        )
        self._persistence = PersistenceForecast(
            spec,
            target_input_features=persistence_input_features,
        )
        self.reset()

    @property
    def spec(self) -> ModelSpec:
        """Return the common forecast tensor contract."""
        return self._spec

    def reset(self) -> None:
        """Discard all accumulated comparison statistics."""
        self._all_learned = _ScoreAccumulator(self.spec, self._thresholds)
        self._all_persistence = _ScoreAccumulator(self.spec, self._thresholds)
        self._wet_learned = _ScoreAccumulator(self.spec, self._thresholds)
        self._wet_persistence = _ScoreAccumulator(self.spec, self._thresholds)

    def update(
        self,
        *,
        dynamic_inputs: Tensor,
        learned_prediction_log1p: Tensor,
        target: Tensor,
        valid_mask: Tensor | None = None,
    ) -> None:
        """Accumulate one batch using a single shared observation-validity mask."""
        validate_forecast_batch(self.spec, dynamic_inputs, target)
        validate_prediction(self.spec, learned_prediction_log1p, target)
        _require_floating(dynamic_inputs, "dynamic_inputs")
        _require_floating(learned_prediction_log1p, "learned_prediction_log1p")
        _require_floating(target, "target")

        mask = _validity_mask(target, valid_mask)
        _require_non_negative(target, mask, "target")

        if not bool(torch.isfinite(learned_prediction_log1p).all()):
            raise ValueError(
                "learned_prediction_log1p must contain only finite values."
            )
        learned_physical = inverse_log1p_precipitation(
            learned_prediction_log1p.detach()
        ).clamp_min(0.0)
        _require_finite_on_mask(learned_physical, mask, "learned prediction")

        self._persistence.to(dynamic_inputs.device)
        with torch.no_grad():
            persistence = self._persistence(dynamic_inputs.detach())
        _require_finite_on_mask(persistence, mask, "persistence prediction")
        _require_non_negative(persistence, mask, "persistence prediction")

        # Research scoring intentionally uses float64 sufficient statistics.
        # MPS has no float64 tensors, so normalize detached evaluation values
        # to CPU before the double-precision scoring path.
        learned_physical = learned_physical.cpu().to(dtype=torch.float64)
        persistence = persistence.detach().cpu().to(dtype=torch.float64)
        physical_target = target.detach().cpu().to(dtype=torch.float64)
        mask = mask.cpu()

        self._all_learned.update(learned_physical, physical_target, mask)
        self._all_persistence.update(persistence, physical_target, mask)

        wet_samples = (
            (mask & (physical_target >= self._wet_threshold))
            .reshape(target.shape[0], -1)
            .any(dim=1)
        )
        if bool(wet_samples.any()):
            sample_mask = wet_samples.reshape(-1, 1, 1, 1, 1)
            wet_mask = mask & sample_mask
            self._wet_learned.update(learned_physical, physical_target, wet_mask)
            self._wet_persistence.update(persistence, physical_target, wet_mask)

    def compute(self) -> ResearchEvaluationReport:
        """Return the accumulated research comparison."""
        learned = self._all_learned.compute(self._lead_minutes)
        persistence = self._all_persistence.compute(self._lead_minutes)
        all_comparison = _comparison(learned, persistence)

        wet_comparison: ConditionComparison | None = None
        if self._wet_learned.count > 0:
            wet_learned = self._wet_learned.compute(self._lead_minutes)
            wet_persistence = self._wet_persistence.compute(self._lead_minutes)
            wet_comparison = _comparison(wet_learned, wet_persistence)

        return ResearchEvaluationReport(
            all=all_comparison,
            wet_target=wet_comparison,
            thresholds_mm_h=self._thresholds,
            wet_threshold_mm_h=self._wet_threshold,
        )


class _ScoreAccumulator:
    def __init__(self, spec: ModelSpec, thresholds: tuple[float, ...]) -> None:
        self._spec = spec
        self._thresholds = thresholds
        self._sum_abs = 0.0
        self._sum_squared = 0.0
        self._count = 0
        self._lead_sum_abs = [0.0] * spec.forecast_steps
        self._lead_sum_squared = [0.0] * spec.forecast_steps
        self._lead_count = [0] * spec.forecast_steps
        self._contingency = [[0, 0, 0] for _ in thresholds]
        self._lead_contingency = [
            [[0, 0, 0] for _ in thresholds] for _ in range(spec.forecast_steps)
        ]

    @property
    def count(self) -> int:
        return self._count

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        if prediction.shape != target.shape or target.shape != mask.shape:
            raise ValueError("prediction, target, and mask shapes must match.")

        error = prediction - target
        selected = error[mask]
        self._sum_abs += float(selected.abs().sum().item())
        self._sum_squared += float(selected.square().sum().item())
        self._count += int(mask.sum().item())

        for lead in range(self._spec.forecast_steps):
            lead_mask = mask[:, lead]
            lead_error = error[:, lead][lead_mask]
            self._lead_sum_abs[lead] += float(lead_error.abs().sum().item())
            self._lead_sum_squared[lead] += float(lead_error.square().sum().item())
            self._lead_count[lead] += int(lead_mask.sum().item())

        for threshold_index, threshold in enumerate(self._thresholds):
            _add_contingency(
                self._contingency[threshold_index],
                prediction,
                target,
                mask,
                threshold,
            )
            for lead in range(self._spec.forecast_steps):
                _add_contingency(
                    self._lead_contingency[lead][threshold_index],
                    prediction[:, lead],
                    target[:, lead],
                    mask[:, lead],
                    threshold,
                )

    def compute(self, lead_minutes: tuple[int, ...]) -> ResearchForecastReport:
        if self._count == 0:
            raise RuntimeError("No valid forecast values have been accumulated.")
        if any(count == 0 for count in self._lead_count):
            raise RuntimeError(
                "Every forecast lead must contain at least one valid value."
            )

        overall = _metric_summary(
            count=self._count,
            sum_abs=self._sum_abs,
            sum_squared=self._sum_squared,
            thresholds=self._thresholds,
            contingency=self._contingency,
        )
        per_lead = tuple(
            _metric_summary(
                count=self._lead_count[lead],
                sum_abs=self._lead_sum_abs[lead],
                sum_squared=self._lead_sum_squared[lead],
                thresholds=self._thresholds,
                contingency=self._lead_contingency[lead],
            )
            for lead in range(self._spec.forecast_steps)
        )
        return ResearchForecastReport(
            overall=overall,
            per_lead=per_lead,
            lead_minutes=lead_minutes,
        )


def _metric_summary(
    *,
    count: int,
    sum_abs: float,
    sum_squared: float,
    thresholds: tuple[float, ...],
    contingency: list[list[int]],
) -> ResearchMetricSummary:
    categorical = tuple(
        _categorical_score(threshold, counts)
        for threshold, counts in zip(thresholds, contingency, strict=True)
    )
    return ResearchMetricSummary(
        count=count,
        mae=sum_abs / count,
        rmse=sqrt(sum_squared / count),
        categorical=categorical,
    )


def _categorical_score(threshold: float, counts: list[int]) -> CategoricalScore:
    hits, false_alarms, misses = counts
    return CategoricalScore(
        threshold_mm_h=threshold,
        hits=hits,
        false_alarms=false_alarms,
        misses=misses,
        csi=_ratio(hits, hits + false_alarms + misses),
        precision=_ratio(hits, hits + false_alarms),
        recall=_ratio(hits, hits + misses),
    )


def _comparison(
    learned: ResearchForecastReport,
    persistence: ResearchForecastReport,
) -> ConditionComparison:
    return ConditionComparison(
        learned=learned,
        persistence=persistence,
        skill=ResearchSkillReport(
            overall=_skill_summary(learned.overall, persistence.overall),
            per_lead=tuple(
                _skill_summary(model, baseline)
                for model, baseline in zip(
                    learned.per_lead,
                    persistence.per_lead,
                    strict=True,
                )
            ),
            lead_minutes=learned.lead_minutes,
        ),
    )


def _skill_summary(
    learned: ResearchMetricSummary,
    persistence: ResearchMetricSummary,
) -> ErrorSkillSummary:
    return ErrorSkillSummary(
        mae=_error_skill(learned.mae, persistence.mae),
        rmse=_error_skill(learned.rmse, persistence.rmse),
    )


def _error_skill(model_error: float, persistence_error: float) -> float | None:
    if persistence_error == 0.0:
        return None
    return 1.0 - model_error / persistence_error


def _add_contingency(
    counts: list[int],
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    threshold: float,
) -> None:
    predicted_event = prediction >= threshold
    target_event = target >= threshold
    counts[0] += int((mask & predicted_event & target_event).sum().item())
    counts[1] += int((mask & predicted_event & ~target_event).sum().item())
    counts[2] += int((mask & ~predicted_event & target_event).sum().item())


def _validity_mask(target: Tensor, valid_mask: Tensor | None) -> Tensor:
    if valid_mask is None:
        return torch.isfinite(target)
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must use dtype torch.bool.")
    if valid_mask.shape != target.shape:
        raise ValueError("valid_mask must have the same shape as target.")
    mask = valid_mask.to(device=target.device)
    if bool((mask & ~torch.isfinite(target)).any()):
        raise ValueError("valid_mask marks non-finite target values as valid.")
    return mask


def _require_floating(value: Tensor, name: str) -> None:
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating-point dtype.")


def _require_finite_on_mask(value: Tensor, mask: Tensor, name: str) -> None:
    if bool((mask & ~torch.isfinite(value)).any()):
        raise ValueError(f"{name} is non-finite on a valid target value.")


def _require_non_negative(value: Tensor, mask: Tensor, name: str) -> None:
    if bool((mask & (value < 0.0)).any()):
        raise ValueError(f"{name} must not be negative on valid target values.")


def _validate_thresholds(values: tuple[float, ...]) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError("At least one precipitation threshold is required.")
    if any(not isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError(
            "Precipitation thresholds must be finite and greater than zero."
        )
    if any(current >= following for current, following in pairwise(thresholds)):
        raise ValueError("Precipitation thresholds must be strictly increasing.")
    return thresholds


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
