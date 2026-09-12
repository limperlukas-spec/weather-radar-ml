# ADR 0009: Reproducible immutable ML dataset artifacts

## Status

Accepted

## Context

Training and evaluation require a stable definition of which samples belong to
a dataset build. Re-running discovery against mutable paths must not silently
change the training population.

## Decision

Each ML dataset build is represented by an immutable artifact directory named by
a deterministic `dataset_build_id`.

```text
<output>/<dataset_build_id>/
├── samples.sqlite
├── manifest.json
└── config.resolved.yaml
```

The build identity is derived from resolved dataset semantics and explicit
source checksums.

The SQLite catalog materializes sample identities and metadata. Radar arrays
remain in prepared Zarr sources and are loaded lazily.

Builds are first written to a temporary directory and moved into their final
location only after successful completion.

If a complete artifact already exists, it is validated and reused without
requiring the original prepared source paths to be currently available.

DVC remains outside the build service. It may version prepared inputs and
completed artifacts, but the builder itself consumes only configuration, paths,
artifact IDs, and explicit checksums.

## Consequences

- Dataset builds are reproducible and auditable.
- Changing a source checksum changes dataset identity.
- Existing artifacts remain usable if source storage is unavailable.
- Failed builds do not leave apparently complete artifacts behind.
- Source checksums must be supplied explicitly.
