# ADR 0004: Data lifecycle and canonical radar model

- Status: Accepted
- Date: 2026-09-05

## Context

Experiments must vary observation domains, prediction domains, temporal context, forecast horizons, and eventually data sources without spreading DWD-specific details through the project. Raw observations also need to remain scientifically auditable.

## Decision

Use four explicit lifecycle stages: **Raw -> Canonical -> Prepared -> ML Dataset**.

Raw source artifacts are immutable. Every derived artifact must be reproducible through code. Canonical radar data uses framework-independent domain types backed by xarray with named `time`, `y`, and `x` dimensions. Observation provenance distinguishes observed, interpolated, extrapolated, and missing values.

Observation and prediction domains remain independent experiment concepts. The canonical layer does not crop data merely because a later experiment predicts a smaller target region.

Source acquisition and decoding are separate protocol boundaries. Source-specific terms such as RADKLIM-YW belong in adapters, not the canonical domain model.

## Alternatives

A direct Raw -> ML Dataset pipeline was rejected because it couples source decoding to experiments. A three-stage Raw -> Canonical -> ML Dataset lifecycle was rejected because reusable expensive preprocessing would have no explicit home.

## Consequences

The project has more explicit stages and metadata, but source adapters, preprocessing, and ML sampling can evolve independently. Missingness cannot silently become zero precipitation. Deterministic transformations and immutable raw data support reproducibility.
