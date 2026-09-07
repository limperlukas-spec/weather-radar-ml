# ADR 0006: Missing observation policy

- Status: Accepted
- Date: 2026-09-05

## Context

Radar products may contain invalid pixels or entirely missing frames. Zero precipitation is a valid observation and must never be used as an implicit missing-data marker. Offline preprocessing can also see future observations that are unavailable during live inference, creating a risk of temporal leakage.

## Decision

Canonical data explicitly records provenance states: **missing, observed, interpolated, extrapolated**. Imputed values remain distinguishable from measurements.

Inference-time repair may use only information available at the forecast issue time. Bidirectional interpolation is therefore not a valid simulation of live inference when it consumes a future frame. Initial inference policy will prefer last-observation/persistence for a small frame gap and reject or flag forecasts when data availability falls below a configured threshold. Motion-based extrapolation and missingness-aware ML are future strategies to benchmark.

## Consequences

Data loaders and evaluation can stratify results by data quality and quantify robustness to missing observations. The pipeline must carry provenance metadata alongside precipitation values, adding modest storage and implementation cost.
