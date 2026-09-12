# ML Dataset Foundation 0.4

Data Foundation 0.4 defines the reproducible bridge from prepared
meteorological data to model training.

## Data flow

```text
Prepared Zarr sources
        │
        ▼
MLDatasetConfig
        │
        ▼
SampleIndexBuilder
        │
        ▼
samples.sqlite
        │
        ▼
LazyRadarDataset
        │
        ▼
SampleTransform(s)
        │
        └── TorchRadarDataset -> PyTorch DataLoader
```

Prepared arrays are not copied into SQLite. The catalog stores identities and
metadata; `LazyRadarDataset` opens prepared Zarr sources only when needed.

## Sample semantics

A `RadarSample` contains dynamic inputs `[T, C, H, W]`, optional static inputs
`[C, H, W]`, dynamic targets `[T, C, H, W]`, timestamps, spatial window, stable
IDs, and immutable context.

The target contains the full configured forecast sequence. Experiments may
select horizons later without redefining the dataset build.

A missing required observation rejects the complete sample. Version 0.4 does
not interpolate or pad missing required frames.

## Spatial sampling and splits

Full-frame and fixed patch sampling share the same abstraction. Patch mode uses
integer pixel windows and no padding.

Train, validation, and test use explicit non-overlapping temporal blocks.
Intentional gaps are allowed. Split assignment is materialized in the catalog.

## Event metadata

Samples may include precipitation statistics and event labels:

- dry
- light
- moderate
- heavy
- extreme

Thresholds are technical defaults and must not be treated as meteorologically
calibrated without validation against the target data distribution.

## Transforms and PyTorch

Transforms operate on complete `RadarSample` objects and must not mutate their
input.

`TorchRadarDataset` is an adapter at the framework boundary. It exposes
`dynamic_inputs`, optional `static_inputs`, `targets`, and `sample_id`.

## Reproducible artifact build

The build creates:

```text
<output>/<dataset_build_id>/
├── samples.sqlite
├── manifest.json
└── config.resolved.yaml
```

A checksum JSON maps each configured `source_id` to its immutable checksum.
DVC remains outside the builder.

## CLI

Plan a build without opening the prepared sources:

```bash
uv run python -m weather_radar_ml.entrypoints.build_ml_dataset \
  --config <dataset-config.yaml> \
  --checksums <checksums.json> \
  --output <artifact-root> \
  --dry-run
```

Build the artifact by omitting `--dry-run`.

## Quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```
