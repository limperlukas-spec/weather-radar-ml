# ADR 0010: Training and experiment foundation

## Status

Accepted

## Context

Data Foundation 0.4 established reproducible, framework-neutral ML dataset
artifacts and a PyTorch adapter. The next layer must execute forecast experiments
without coupling dataset semantics, model implementations, experiment tracking,
and the Hydra configuration system.

Experiments must remain comparable and auditable. A failed external tracking
service must not invalidate a completed local experiment, and checkpoint/resume
must preserve enough state to continue deterministic training where supported.

## Decision

Training & Experiment Foundation 0.5 introduces one explicit experiment runtime
with the following boundaries:

1. Hydra is confined to configuration composition at the application boundary.
2. The core receives an immutable typed `RunConfig` and derives an
   `ExperimentConfig` with a stable fingerprint.
3. Forecast models implement a common tensor contract described by `ModelSpec`.
4. Dataset samples are converted to `ForecastBatch` objects through the PyTorch
   adapter and a reproducibly seeded `DataLoader`.
5. `ForecastTrainer` owns train/validation execution and reports exact dataset-level
   continuous metrics. Optimizers are optional so parameter-free baselines can be
   evaluated without artificial training steps.
6. Checkpoints store model state, optimizer state when present, completed epochs,
   and Python, NumPy, and PyTorch RNG state for reproducible resume.
7. A completed run is atomically published as an immutable local artifact
   containing `manifest.json`, `config.json`, `history.json`, and optional
   checkpoints.
8. The local run artifact is the canonical experiment record. MLflow is an
   optional secondary projection for comparison, indexing, and UI use.
9. MLflow receives metadata, parameters, metrics, and lightweight run-artifact
   files, but checkpoints remain in canonical local storage.
10. The installed `weather-radar-ml` CLI executes the same runtime path that is
    covered by automated end-to-end tests.

The initial executable baseline is persistence forecasting. Because persistence has
no trainable parameters, its default run evaluates complete train and validation
splits and intentionally produces no checkpoint.

## Consequences

- Dataset, model, trainer, artifact, and tracking concerns have explicit boundaries.
- Reproducibility information is persisted independently of MLflow availability.
- Parameter-free baselines remain scientifically honest rather than being forced
  through fake optimization steps.
- Future trainable models can reuse the same trainer, checkpoint, artifact, metric,
  and tracking infrastructure.
- External tracking failures can surface to the caller without deleting an already
  published canonical run artifact.
- Runtime configuration that materially affects execution is part of the
  experiment identity or persisted run metadata.
- The CLI provides a real end-to-end integration path instead of a configuration-
  only smoke test.

## Scope

This ADR does not select a learned forecast architecture, hyperparameter-search
system, distributed-training strategy, deployment runtime, or browser
visualization. Those remain separate future decisions.
