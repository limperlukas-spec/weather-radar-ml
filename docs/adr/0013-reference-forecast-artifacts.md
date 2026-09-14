# ADR 0013: Reference forecast artifacts

## Status

Accepted

## Context

The multi-seed experiment selects one reference seed by median validation
performance. The project now needs complete prediction fields from that selected
run for reproducible qualitative analysis, later browser visualization, and the
held-out RADKLIM benchmark. Training and metric summaries alone are insufficient
because they do not retain spatial forecast structure or stable sample identity.

The prediction artifact must preserve the same learned-versus-persistence
semantics defined by ADR 0011 without introducing a parallel data format or a
second sample identity system.

## Decision

Version 0.6.5 introduces an immutable reference-forecast artifact with these
rules:

1. Forecast fields are stored as Zarr through xarray, matching the project's
   prepared-data storage stack.
2. Arrays use `[sample, lead, feature, y, x]` and contain learned forecast,
   persistence forecast, target precipitation, and the target-derived validity
   mask.
3. Learned and persistence values are stored in physical precipitation units
   (mm/h). Learned output is inverse-transformed with `expm1` and clamped to
   zero exactly as in the research-evaluation policy.
4. Stable dataset `sample_id` values are retained in canonical inference order,
   together with explicit lead-minute and target-feature coordinates.
5. Version 0.6.5 materializes the complete validation predictions for the
   selected reference seed. This validates the artifact path without consuming
   or selecting on the held-out test split. Version 0.6.6 will reuse the format
   for the final test benchmark.
6. A small deterministic qualitative subset is recorded by selecting samples
   with the highest valid target precipitation, breaking ties by `sample_id`.
   The subset references the complete store instead of duplicating forecast
   arrays.
7. The artifact records experiment/dataset configuration, seed, best-epoch
   selection, validation research results, environment metadata, device/CUDA
   information, and the SHA-256 hash of the selected best checkpoint.
8. Publication is atomic. A manifest records content hashes for metadata,
   qualitative selection, and the complete Zarr directory, and derives a stable
   artifact fingerprint from those records.
9. The multi-seed summary references the forecast artifact of the median
   validation-performance seed. Large prediction data remain local canonical
   artifacts rather than being uploaded wholesale to MLflow.

## Consequences

- Spatial predictions can be inspected later without rerunning training.
- Learned, persistence, and ground-truth fields remain aligned by sample and
  lead time.
- Qualitative visualization can select interesting cases reproducibly while
  scientific metrics remain independent of that subset.
- Integrity can be checked without treating a Zarr directory as one opaque
  file.
- The held-out test set remains untouched during model/seed selection in 0.6.5.
- The final real-data run can reuse the same artifact schema instead of defining
  another prediction format.

## Scope

This ADR defines the reference prediction artifact format and its validation-
split integration. It does not run the final RADKLIM test benchmark, select the
production ROI, or implement browser visualization.
