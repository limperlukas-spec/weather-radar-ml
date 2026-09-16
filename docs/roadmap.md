# Project roadmap

This roadmap records the current milestone sequence for Weather Radar ML.
Milestones are evidence-driven: a later model or product decision should follow
from the measured limitations of the preceding benchmark rather than from
feature count alone.

## Completed foundations

### 0.1-0.4 — Data foundation

Establish the project toolchain, canonical radar representation, native
RADKLIM-YW ingestion, prepared storage, immutable ML datasets, stable sample
identity, temporal splits, and DVC-backed data reproducibility.

### 0.5 — Training and experiment foundation

Establish executable training, checkpoints, run artifacts, experiment tracking,
common forecast contracts, and the infrastructure required for learned models.

### 0.6 — First learned reference benchmark

Status: completed.

Goals:

- implement the first deliberately simple learned precipitation forecast,
- evaluate learned forecasts and persistence under identical semantics,
- aggregate a fixed three-seed protocol,
- select the qualitative reference run using validation only,
- execute a real RADKLIM reference run,
- evaluate once on the untouched held-out test split,
- persist reproducible prediction and benchmark artifacts.

Outcome:

- the U-Net reference has positive MAE and RMSE skill over persistence at every
  forecast lead from +5 to +30 minutes,
- aggregate test skill is approximately 39.7% for MAE and 33.6% for RMSE,
- intense precipitation at longer lead times is the clearest identified
  weakness,
- the benchmark is strong enough to justify comparison against a more
  competitive non-ML nowcasting baseline.

See `docs/benchmarks/0.6-real-radklim-reference.md`.

## Active next milestone

### 0.7 — Classical nowcasting benchmark

Research question:

> Does the learned U-Net reference provide skill beyond an established
> motion-based classical nowcasting method, not only beyond persistence?

Planned scope:

- add one established optical-flow / extrapolation nowcasting strategy, with
  PySTEPS-compatible methodology preferred where practical,
- adapt it through the existing forecast-strategy boundary rather than creating
  a parallel evaluation path,
- use the same real RADKLIM dataset, ROI, temporal splits, forecast leads, masks,
  physical units, and categorical thresholds as the 0.6 reference,
- compare persistence, classical nowcasting, and U-Net under the same evaluator,
- keep the held-out test split out of implementation and method-selection
  decisions,
- persist a reproducible benchmark artifact and qualitative forecast fields.

Completion criteria:

- the classical strategy can run end to end through the reference dataset,
- all three strategies are evaluated on a common population,
- per-lead and aggregate continuous and categorical metrics are available,
- the result clearly identifies where the learned model adds value and where a
  classical motion model remains competitive,
- no neural-model redesign is introduced merely to improve the 0.7 score.

## Candidate later milestones

The exact numbering and ordering after 0.7 remain intentionally provisional.
The 0.7 evidence should determine which research branch has the highest value.

Likely branches include:

- benchmark-informed model improvement, including architecture or loss changes,
- focused treatment of intense precipitation and longer lead times,
- broader temporal coverage and seasonal/generalization studies,
- larger spatial context and regional-transfer experiments,
- additional meteorological variables such as temperature and wind,
- browser-based research and layperson forecast visualization,
- deployment/runtime optimization.

Hyperparameter optimization, probabilistic ensembles, ConvLSTM/Transformer
families, Germany-wide training, transfer learning, and browser deployment are
not silently pulled into 0.7. They remain separate decisions after the
classical benchmark establishes a stronger reference point.
