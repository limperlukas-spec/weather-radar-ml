# Weather Radar ML

A reproducible research framework for benchmarking radar-based weather forecasting strategies under consistent data and evaluation conditions.

> **Status:** Training & Experiment Foundation 0.5.

## Why this project exists

The project has two goals: investigate radar-based weather forecasting methods and demonstrate disciplined ML/software engineering through reproducible experiments, explicit architectural decisions, meaningful tests, and understandable documentation.

The framework is deliberately designed so that data sources and forecasting strategies can evolve while evaluation remains comparable. The first implemented forecast will be a simple persistence baseline; more sophisticated approaches will be added only after the data and evaluation foundations exist.

## Engineering principles

- Correctness and reproducibility before feature count.
- Tests are selected for behavior and defect-detection value, not just coverage.
- Architecture decisions are recorded as ADRs.
- Hydra is restricted to configuration composition at the application boundary.
- Core code uses typed Python configuration objects and must not depend on Hydra APIs.
- New ideas outside the active milestone go to the backlog instead of blocking delivery.

## Quick start

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run weather-radar-ml
```

The default command executes a bounded synthetic persistence-baseline run and
publishes an immutable local run artifact. Its output is similar to:

```text
run_id=<run-id> artifact=<runs-root>/<run-id>
```

The artifact contains the resolved experiment configuration, epoch history, and a
manifest with checksums, runtime metadata, and final metrics. Persistence has no
trainable parameters, so the default run intentionally contains no checkpoint.

Hydra remains restricted to the application boundary. Runtime settings can be
overridden explicitly, for example:

```bash
uv run weather-radar-ml training.seed=7 output.runs_root=./runs
```

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv build
```

Mutation testing is intentionally not part of every CI run. It is used selectively on critical modules; surviving mutants must be reviewed and documented. See [`docs/testing-strategy.md`](docs/testing-strategy.md).

## Repository structure

```text
weather-radar-ml/
├── configs/               # Hydra composition layer
├── docs/                  # Architecture, testing and research documentation
│   └── adr/               # Architecture Decision Records
├── experiments/           # Human-readable experiment notes / manifests
├── scripts/               # Operational helper scripts
├── src/weather_radar_ml/  # Installable application package
├── tests/                 # Automated tests
└── .github/workflows/     # CI quality gates
```

## Current scope

Data Foundation 0.4 provides reproducible ML dataset artifacts, a SQLite-backed
sample catalog, lazy Zarr loading, framework-neutral sample semantics, transforms,
and a PyTorch adapter.

Training & Experiment Foundation 0.5 adds the model contract, persistence baseline,
continuous forecast metrics, trainer core, deterministic execution, checkpoint and
resume support, immutable run artifacts, optional MLflow projection, and an
executable Hydra/CLI pipeline. The default persistence run evaluates the complete
train and validation splits without pretending that a parameter-free baseline is
trainable.

More sophisticated learned forecast models, model-specific architecture research,
hyperparameter search, distributed training, and browser visualization remain
outside this milestone.

## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/testing-strategy.md`](docs/testing-strategy.md)
- [`docs/development.md`](docs/development.md)
- [`docs/adr/`](docs/adr/)

## Experiment lifecycle

A configured run follows one explicit path:

```text
Hydra
  -> typed RunConfig
  -> dataset + reproducible DataLoader
  -> ModelSpec + model
  -> loss + optional optimizer
  -> ForecastTrainer
  -> TrainingHistory
  -> optional checkpoint
  -> immutable local run artifact
  -> optional MLflow projection
```

The local run artifact is the canonical experiment record. MLflow is a secondary
comparison/index/UI projection and must not become the only copy of experiment
state. If tracking fails after local publication, the canonical artifact remains
intact.

Run reproducibility includes the resolved experiment fingerprint, dataset identity,
seed and deterministic-algorithm settings, software/platform metadata, and, where
applicable, model/optimizer state plus Python, NumPy, and PyTorch RNG state.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

## Data lifecycle

The project uses an immutable `Raw -> Canonical -> Prepared -> ML Dataset` lifecycle. Canonical radar data is represented with typed domain objects and xarray; prepared training data uses Zarr. Dataset manifests plus DVC provide semantic and artifact-level reproducibility. See `docs/architecture.md` and ADRs 0004-0006.


## Real RADKLIM-YW sample

After the normal quality gate, a bounded real-data smoke test can be run explicitly:

```bash
uv run python scripts/ingest_radklim_yw_sample.py --date 2023-09-01 --frames 12
```

This downloads the authoritative monthly DWD archive (hundreds of MB), extracts the selected day, decodes only the requested first frames, writes prepared Zarr data, and creates a dataset manifest. Raw and prepared artifacts stay outside Git and can be tracked with DVC after validation.
