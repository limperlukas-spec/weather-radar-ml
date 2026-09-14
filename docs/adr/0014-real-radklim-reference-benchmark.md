# ADR 0014: Real RADKLIM-YW reference benchmark

## Status

Accepted

## Context

Version 0.6 has established a learned U-Net forecast, validation-based checkpoint
selection, research evaluation against Persistence, three-seed aggregation, and
immutable forecast artifacts. The milestone still needs one real-data reference
experiment whose held-out test data cannot influence model or reference-seed
selection.

The available immutable source is the complete September 2023 RADKLIM-YW monthly
archive on the native 1100 x 900, 1 km grid. Training on the complete grid would
add large I/O and compute cost without serving the first local Ruhrgebiet
reference objective. The learned model also does not yet implement a
missingness-aware input strategy.

## Decision

The 0.6 real reference benchmark uses the following fixed policy.

1. The source month is RADKLIM-YW September 2023, product version `2017.002`.
2. Prepared data are cropped before storage to one 128 x 128 native-grid ROI.
   The ROI origin is `(x=196, y=549)` and Dortmund is approximately centred at
   native pixel `(260, 613)`. Native projected x/y coordinates are preserved.
3. Prepared frames containing any non-finite or negative precipitation value
   inside the ROI are omitted instead of being silently imputed. Their absence
   remains visible as a temporal gap; the existing sample builder rejects any
   history/target window that crosses such a gap.
4. The ML sample geometry is 60 minutes history and 30 minutes direct forecast
   on the 5-minute RADKLIM grid: 12 input frames and 6 target frames.
5. The prepared ROI is already spatially fixed, so the ML dataset uses
   `full_frame` rather than introducing a project-specific sampling mode.
6. The chronological split is:
   - train: `[2023-09-01, 2023-09-21)`
   - validation: `[2023-09-22, 2023-09-26)`
   - test: `[2023-09-27, 2023-10-01)`

   September 21 and September 26 are intentional full-day guard gaps.
7. Seeds remain `(17, 42, 73)`. Each seed trains independently and selects its
   best checkpoint only by validation MSE in log1p space.
8. The reference seed is selected as the median best-validation-loss run before
   any test evaluation begins.
9. Only after that selection is fixed are all three best checkpoints evaluated
   on the held-out test split. Test research metrics are aggregated with the
   same mean/sample-standard-deviation policy as validation metrics.
10. Complete test predictions are persisted only for the already selected
    reference seed, together with Persistence, targets, validity, provenance,
    sample IDs, and the qualitative subset defined by ADR 0013.

The official reference training settings are the 0.6 learned defaults: compact
2D U-Net with `base_channels=32`, log1p-space MSE, Adam at `1e-3`, batch size 4,
maximum 50 epochs, early-stopping patience 7, deterministic algorithms enabled,
and seeds 17/42/73.

## Consequences

- Test targets cannot influence checkpoint or reference-seed selection.
- The final benchmark reports test metrics for all three seeds rather than only
  the visually selected reference run.
- Cropping at the prepared-data boundary substantially reduces storage and I/O
  while keeping the exact native coordinates auditable.
- Rejecting incomplete ROI frames is conservative and reproducible, but the
  reference benchmark does not measure robustness to missing observations.
  Missingness-aware models or causal repair remain later research topics.
- One month and one ROI are deliberately a first learned reference, not a claim
  of seasonal or Germany-wide generalization.
