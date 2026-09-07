# Weather Radar ML

A reproducible research framework for benchmarking radar-based weather forecasting strategies under consistent data and evaluation conditions.

> **Status:** Data Foundation 0.3 — native RADKLIM-YW ingestion.

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

The foundation smoke run should print a composed configuration similar to:

```text
data=synthetic, model=persistence, seed=42, tracking=weather-radar-ml
```

Hydra overrides already work at the boundary:

```bash
uv run weather-radar-ml training.seed=7
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

Data Foundation 0.3 adds the first real DWD RADKLIM-YW ingestion path on top of the canonical data model. Evaluation metrics and learned forecast models remain intentionally out of scope.

## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/testing-strategy.md`](docs/testing-strategy.md)
- [`docs/development.md`](docs/development.md)
- [`docs/adr/`](docs/adr/)

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

## Data lifecycle

Data Foundation 0.2 uses an immutable `Raw -> Canonical -> Prepared -> ML Dataset` lifecycle. Canonical radar data is represented with typed domain objects and xarray; prepared training data uses Zarr. Dataset manifests plus DVC provide semantic and artifact-level reproducibility. See `docs/architecture.md` and ADRs 0004-0006.


## Real RADKLIM-YW sample

After the normal quality gate, a bounded real-data smoke test can be run explicitly:

```bash
uv run python scripts/ingest_radklim_yw_sample.py --date 2023-09-01 --frames 12
```

This downloads the authoritative monthly DWD archive (hundreds of MB), extracts the selected day, decodes only the requested first frames, writes prepared Zarr data, and creates a dataset manifest. Raw and prepared artifacts stay outside Git and can be tracked with DVC after validation.
