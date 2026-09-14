# ADR 0012: Three-seed reference experiment aggregation

## Status

Accepted

## Context

The first learned precipitation forecast must not be judged from one random
initialization. Version 0.6 therefore needs a small, fixed multi-seed protocol
that is reproducible, cheap enough for the reference experiment, and compatible
with the research-evaluation policy from ADR 0011.

The reference run used for qualitative artifacts must also be selected without
cherry-picking the visually best forecast.

## Decision

Version 0.6 uses exactly three fixed seeds: `17`, `42`, and `73`.

1. Every child run uses the same dataset, model, optimizer, loss, and runtime
   configuration except for the training seed.
2. The best checkpoint of each child run is evaluated on the validation split
   with the common learned-versus-persistence research evaluator. This is
   explicitly validation research, not the final test-set benchmark.
3. Numeric research metrics are aggregated with their arithmetic mean and
   sample standard deviation across defined seed values.
4. Undefined metrics remain undefined. Aggregates record how many child runs
   contributed a defined value; no artificial numeric replacement is used.
5. Best validation loss is aggregated in the same way and is the sole basis for
   selecting the reference seed.
6. The reference seed is the run with median best-validation performance.
   Ties are resolved deterministically by seed value.
7. A canonical multi-seed artifact stores all three child identities, their
   best-validation metadata, complete research reports, aggregate statistics,
   and the selected reference seed.
8. When MLflow tracking is enabled, one parent run groups the three child runs.
   Child runs carry the standard `mlflow.parentRunId` tag, while the parent
   receives the aggregate summary artifact and aggregate mean/std metrics.
9. Child run artifacts remain canonical experiment records. MLflow remains a
   secondary projection and does not replace local artifacts.

## Consequences

- Reported performance is less sensitive to one favorable or unfavorable
  initialization.
- The qualitative reference run is selected by a deterministic rule rather
  than by visual inspection.
- Undefined categorical cases remain visible during aggregation.
- The fixed three-run protocol is intentionally small and is not a
  hyperparameter-search or ensemble strategy.
- The multi-seed summary can later drive prediction-artifact selection in 0.6.5
  and the real RADKLIM reference execution in 0.6.6.

## Scope

This decision does not define a larger repeated-cross-validation study, model
ensembling, hyperparameter optimization, the final RADKLIM test period, or the
prediction Zarr artifact format.
