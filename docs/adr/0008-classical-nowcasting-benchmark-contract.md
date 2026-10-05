# ADR 0008: Classical nowcasting benchmark contract

- Status: Accepted
- Date: 2026-10-03

## Context

Version 0.6 established a real RADKLIM-YW reference benchmark using a fixed
128 x 128 km Ruhrgebiet ROI, twelve five-minute input frames, direct forecast
leads from +5 to +30 minutes, a persistence baseline, and a deterministic U-Net.
The U-Net was selected without test leakage and retained positive MAE and RMSE
skill over persistence at every lead. The next benchmark must determine whether
that learned skill remains meaningful against an established motion-based
classical method.

The classical method must not introduce a second data or evaluation path. Its
configuration and selection policy must also be fixed before the held-out test
split is opened for classical evaluation.

## Decision

### Common benchmark population

The 0.7 benchmark reuses the 0.6 dataset identity, 128 x 128 km ROI, temporal
splits, 5-minute cadence, twelve available input frames, forecast leads, missing
observation policy, physical mm/h output units, validity masks, and categorical
thresholds. The frozen 0.6 U-Net reference is not retrained or redesigned for
0.7.

The official comparison is:

1. persistence,
2. deterministic classical S-PROG nowcasting,
3. the frozen 0.6 U-Net reference.

Pure optical-flow extrapolation is retained as a diagnostic baseline rather
than promoted to the official classical comparator.

### Classical method

PySTEPS is the reference implementation. S-PROG uses the accepted deterministic
configuration committed in `configs/classical/sprog.yaml`:

- six cascade levels,
- AR order two,
- Gaussian band-pass filtering,
- FFT decomposition,
- semi-Lagrangian extrapolation,
- unconditional statistics,
- CDF probability matching.

Radar precipitation is converted internally using a 0.1 mm/h rain threshold.
The corresponding threshold is -10 dBR and the dry value is -15 dBR. Forecasts
are converted back to non-negative mm/h before entering the common evaluator.

Unknown inflow from outside the fixed ROI is not masked out of the official
benchmark. Semi-Lagrangian out-of-domain values use the minimum/dry value. An
interior-ROI analysis may be reported separately as diagnostics but cannot
replace the full-ROI result.

### Motion selection

The motion estimator is selected only on the validation split. The complete
candidate set is committed before execution in
`configs/classical/selection.yaml`:

- VET with two motion frames,
- Lucas-Kanade with 2, 3, 6, and 12 motion frames.

VET is restricted to two frames by method semantics. Multi-frame Lucas-Kanade
is allowed to exploit its native ability to pool sparse vectors from more than
one image pair. `LK_2` versus `VET_2` remains a controlled diagnostic comparison
under identical motion-history length.

The selection criterion is MSE in `log1p` precipitation space over the complete
validation forecast. Runtime, visual appearance, test metrics, and categorical
metrics do not participate in motion-method selection.

The selected motion configuration must be persisted and committed before the
classical pipeline is evaluated on the held-out test split.

### Evaluation and diagnostics

MAE and RMSE, plus skill relative to persistence, remain the primary test
metrics. Results are retained per lead and in aggregate. Existing wet-target and
categorical evaluation at 0.1, 1, and 5 mm/h is reused, with intense
precipitation at >= 5 mm/h explicitly analyzed because 0.6 identified it as a
major weakness.

Planned secondary diagnostics include motion-only extrapolation, `LK_2` versus
`VET_2`, interior-ROI behavior, paired time/event-aware bootstrap confidence
intervals, and inference runtime. Runtime is hardware-specific and is reported
only on the declared reference system. Energy consumption is not reported
because the current Windows/WSL2/ROCm stack does not provide sufficiently clean
process-level energy attribution.

### Dependency boundary

PySTEPS is an optional `classical` project extra rather than a core dependency.
The benchmark pins PySTEPS 1.21.5; OpenCV is supplied explicitly because the
Lucas-Kanade implementation depends on it while PySTEPS's package metadata does
not install it. The lockfile records the resolved dependency graph used by
benchmark runs.

Core application code continues to receive typed Python configuration objects
and must not depend on Hydra APIs. PySTEPS-specific representations remain
behind the classical forecast adapter introduced in 0.7.2.

## Consequences

The test split remains protected from classical-method selection. The benchmark
can distinguish gains from motion estimation, deterministic classical
nowcasting, and learned forecasting without changing the 0.6 evaluation policy.
Lucas-Kanade and VET may use method-appropriate histories while still drawing
from the same twelve available observations.

The optional dependency keeps existing persistence and U-Net installations
independent of PySTEPS and OpenCV. PySTEPS contains compiled extensions, so the
classical extra may require a platform toolchain even though normal project
development does not.
