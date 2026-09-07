# ADR 0005: Prepared storage and data versioning

- Status: Accepted
- Date: 2026-09-05

## Context

Prepared radar data is multidimensional and training repeatedly reads partial temporal and spatial windows. Scientific exchange, efficient ML access, and exact dataset reproducibility have different requirements.

## Decision

Use **Zarr** as the primary prepared-data store and xarray as the in-memory labeled-array representation. Chunk sizes remain a benchmarked configuration concern rather than an architecture constant. NetCDF remains an intended scientific interchange/export format.

Use **DVC** for data-artifact versioning. Git stores DVC metadata while the large artifacts live outside Git. A dataset manifest records semantic identity, source product/version, source checksums, pipeline version, and canonical schema version. DVC and manifests are complementary: DVC identifies artifact versions; manifests explain what they contain and how they were derived.

No NAS remote path or credentials are committed. Each development machine configures its own DVC remote. A TrueNAS-backed local mount or SSH remote can be selected during environment setup.

## Alternatives

NetCDF/HDF5 was considered for prepared storage because of its strong meteorological ecosystem. It remains useful for interchange, but Zarr is preferred for the training store because chunked partial access and parallel/object-style storage are first-class concerns. NumPy files were rejected as the primary store because they provide weaker multidimensional metadata and access patterns.

## Consequences

Zarr performance depends on chunking and the underlying filesystem, so performance on the actual NAS/local storage must be measured. DVC adds tooling and storage metadata but allows code, configuration, and exact data revisions to be associated reproducibly.
