# Architecture

## Current boundary

Foundation 0.1 establishes infrastructure without prematurely implementing domain components.

```text
Hydra YAML/config groups
          |
          v
Hydra composition boundary
(entrypoints/)
          |
          v
typed RunConfig dataclasses
(config/)
          |
          v
future domain/data/model/evaluation components
```

The important rule is dependency direction: future core components may depend on typed configuration schemas, but they must not depend on Hydra APIs. This keeps experiment configuration replaceable and prevents framework-specific objects from spreading through the codebase.

## Planned architectural areas

The next milestones are expected to introduce explicit boundaries for data ingestion, forecast representation, evaluation, baseline strategies, and later ML training. Interfaces will be added only when a known variation in the project requires them.

## Architecture documentation policy

Significant decisions are stored as Architecture Decision Records in `docs/adr/`. ADRs describe the context known at decision time, the selected decision, relevant alternatives, and consequences. Accepted ADRs are historical records; if a decision changes, a new ADR supersedes the previous one rather than rewriting history.

## Data Foundation 0.2

The data architecture separates source acquisition from interpretation and experiment sampling:

```text
remote source
    |
    v
DataSource -> immutable Raw artifacts
                  |
                  v
               Decoder
                  |
                  v
        Canonical radar domain (xarray)
                  |
                  v
             Preprocessor
                  |
                  v
          Prepared Zarr store
          + dataset manifest
                  |
                  v
            Sample Builder
                  |
                  v
        model-specific adapter
```

Canonical radar dimensions are named `time`, `y`, and `x`. Source-specific concepts remain at adapter boundaries. Observation and prediction domains are intentionally independent. Prepared-store chunking is not fixed by the architecture and must be benchmarked against real access patterns and storage.

DVC versions large data artifacts separately from Git. Machine-specific DVC remotes, including NAS paths or credentials, must not be committed.
