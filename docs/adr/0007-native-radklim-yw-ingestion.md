# ADR 0007: Native RADKLIM-YW ingestion

- Status: Accepted
- Date: 2026-09-07

## Context

The first real data adapter must validate the Data Foundation 0.2 boundaries against an authoritative meteorological source without coupling the canonical domain to DWD file conventions.

RADKLIM-YW 2017.002 is the primary historical source. DWD distributes the native reprocessed binary data in monthly tar archives. Each composite uses the RADOLAN/RADKLIM binary structure and the extended 1100 x 900 grid at 1 km spacing.

## Decision

- Acquire the original monthly DWD archive as an immutable raw artifact.
- Keep archive extraction and native binary decoding in `weather_radar_ml.data.dwd`.
- Decode native YW 5-minute precipitation amounts to the canonical `precipitation_rate` unit mm/h.
- Map DWD missing-value flags to canonical `MISSING` provenance and NaN values.
- Preserve source-specific secondary/clutter bits separately as `source_flags`; they are not canonical provenance states.
- Represent coordinates at 1 km pixel centres in metres using the legacy RADOLAN polar-stereographic CRS.
- Validate product identifier, timestamp, grid metadata, precision, interval, and payload size before accepting a frame.
- Use a bounded sample for integration validation before scaling ingestion to months or years.

## Consequences

The canonical domain remains independent of DWD naming and binary layout. Source quality information is not silently discarded. Full historical acquisition remains intentionally separate from this milestone because a monthly YW archive is hundreds of megabytes and a full day contains 288 Germany-wide frames.

The hand-written decoder intentionally supports only the RADKLIM-YW subset required here. Supporting additional RADOLAN products must be a deliberate extension or use a dedicated radar I/O dependency rather than accumulating undocumented format assumptions.


## Cross-format validation

The native decoder was independently validated against the official DWD ASCII
representation of RADKLIM-YW 2017.002 for 2023-09-01 00:00 UTC.

Both representations contain 1100 x 900 cells. The ESRI-ASCII rows are
converted from north-to-south file order into the canonical south-to-north
orientation. After this transformation, the missing-data masks were identical
for all 990,000 cells: 422,272 missing cells and 567,728 valid cells.

The binary representation retains finer numeric precision than the ASCII
representation. Removing the canonical 5-minute-to-hour normalization from the
prepared values reproduces the official ASCII values within its 0.1
quantization. Validation therefore uses a 0.05 quantization tolerance and does
not assume a specific DWD rounding implementation.

This cross-format comparison independently validates:

- `PR` value scaling;
- missing-value decoding;
- raster dimensions;
- raster row orientation;
- spatial grid placement.

Native decoding and canonical unit normalization are separate semantic steps.
The canonical domain stores precipitation rate in mm/h. For `INT 5`:

```text
precipitation_rate_mm_h = native_interval_amount * 60 / 5
```

The full external ASCII archive is a reference-validation artifact and is not
required by the normal unit-test suite or CI.
