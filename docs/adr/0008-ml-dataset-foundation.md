# ADR 0008: Framework-neutral ML dataset foundation

## Status

Accepted

## Context

The project needs a reusable dataset layer between prepared meteorological data
and individual model implementations. The same prepared data must support
different model families while preserving identical sample semantics, splits,
feature ordering, spatial windows, and forecast targets.

## Decision

Data Foundation 0.4 introduces a framework-neutral ML dataset core based on
NumPy-backed domain objects: `RadarInputs`, `RadarTargets`, `RadarSample`,
`SampleContext`, and `SpatialWindow`.

Sample construction is separated into:

1. typed dataset configuration
2. deterministic sample-index builder
3. immutable SQLite sample catalog
4. lazy framework-neutral dataset
5. explicit sample transforms
6. optional framework adapters such as PyTorch

Dynamic inputs and targets use `[T, C, H, W]`; static inputs use `[C, H, W]`.
Feature order is part of the resolved dataset semantics.

The PyTorch adapter converts complete samples into tensors only at the framework
boundary. The domain and lazy dataset layers do not depend on PyTorch.

## Consequences

- Model implementations share identical sample semantics.
- Sample IDs and splits remain stable across training frameworks.
- Lazy loading keeps large prepared arrays out of memory until required.
- New frameworks require a small adapter layer.
- Framework-specific optimizations must not leak into the domain layer.

## Scope

Data Foundation 0.4 does not silently introduce temporal interpolation,
