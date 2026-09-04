# ADR 0002: Restrict Hydra to the application boundary

- Status: Accepted
- Date: 2026-09-04

## Context

The research scope already requires repeated comparisons across known configuration dimensions such as data source, preprocessing, forecast strategy/model, training parameters, and evaluation settings. Plain YAML is familiar and initially simple, but composition, overrides, validation, and multi-run support would otherwise accumulate as custom infrastructure.

At the same time, allowing Hydra `DictConfig` objects to propagate through the application would couple core data/model/evaluation code to a configuration framework.

## Decision

Use Hydra for YAML configuration groups, defaults/composition, command-line overrides, and later controlled multi-run experiments.

Hydra is an outer-layer dependency only. The entrypoint converts composed Hydra configuration into typed Python dataclasses before configuration reaches core application components. Core data, model, evaluation, and training modules must not depend on Hydra APIs.

## Alternatives considered

- YAML + custom loader: lower initial learning cost, but the known experiment-composition requirements would force the project to implement an increasing portion of Hydra-like behavior itself.
- Python-only dataclasses: strong typing but less convenient for human-readable experiment composition and command-line overrides.

## Consequences

The project gains mature experiment composition without making Hydra part of its domain model. The boundary adapter adds a small amount of code and must be tested. Replacing Hydra later remains feasible because core components receive ordinary typed Python objects.
