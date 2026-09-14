# ADR 0011: Research evaluation policy for learned precipitation forecasts

## Status

Accepted

## Context

The first learned precipitation forecast must be compared with persistence under
identical test conditions. Training loss is computed in log1p space, while the
scientific comparison must remain interpretable in physical precipitation units.
RADKLIM observations may also contain missing values, and categorical event
metrics can be mathematically undefined when no event occurs or is predicted.

## Decision

Version 0.6 uses a dedicated research-evaluation layer with these rules:

1. Learned U-Net output is interpreted as log1p precipitation and transformed
   back with `expm1` before evaluation.
2. Negative learned precipitation after the inverse transform is clamped to
   0 mm/h for evaluation only. The raw model output remains unchanged for loss
   computation and model development.
3. Learned and persistence forecasts are evaluated against exactly the same
   observation-validity mask. Missing target observations are excluded; a
   non-finite model prediction on a valid target is an error and is never
   silently removed.
4. Continuous metrics are MAE and RMSE in mm/h, overall and per forecast lead.
5. Categorical metrics are CSI, precision, and recall at 0.1, 1, and 5 mm/h.
   Their underlying hits, false alarms, and misses are retained.
6. A categorical metric with a zero denominator is represented as `None`,
   rather than assigning an artificial score.
7. The wet-target condition is sample-based: a sample is selected when at least
   one valid target value anywhere in its forecast horizon is at or above
   0.1 mm/h. All valid values of each selected sample are then evaluated.
8. Error skill is reported only for MAE and RMSE as
   `1 - error_model / error_persistence`. A zero persistence error yields
   undefined (`None`) skill.
9. With the 5-minute RADKLIM cadence, lead reports are labeled +5, +10, ...;
   the evaluator keeps the lead step configurable.

## Consequences

- Learned and persistence scores are directly comparable and use one common
  population of valid observations.
- Physical units are preserved in scientific result tables even though the
  learned model trains in transformed space.
- Undefined categorical cases remain visible instead of being hidden by a
  convention-dependent numeric value.
- The wet-target subset emphasizes precipitation situations without discarding
  dry pixels surrounding or occurring within those selected samples.
- Evaluation can be accumulated batch by batch without changing the result.
- Observation masks can be supplied explicitly when the real-data runtime
  exposes them; finite targets provide the fallback validity definition.

## Scope

This ADR defines deterministic research metrics and comparison semantics. It
does not define the final test period, spatial ROI, multi-seed aggregation,
prediction-artifact format, or browser visualization.
