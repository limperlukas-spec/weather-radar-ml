# ADR 0001: Project foundation toolchain

- Status: Accepted
- Date: 2026-09-04

## Context

The project is both a radar-weather forecasting research environment and a public engineering portfolio. Development speed matters, but reproducibility, correctness, documentation, and external comprehensibility take priority over minimizing the number of development tools.

## Decision

Use Python 3.11, `uv` with `pyproject.toml`, a `src` layout, pytest, Ruff, mypy strict mode, branch-aware coverage, selective mutation testing, GitHub Actions, Markdown documentation designed to remain compatible with a later MkDocs migration, short-lived feature branches, and Apache License 2.0.

MLflow is selected for local experiment tracking. Hydra is selected for experiment configuration composition subject to the architectural restriction recorded in ADR 0002.

## Alternatives considered

- `pip`/requirements files: simpler but weaker as a unified reproducible project workflow.
- Poetry: capable, but `uv` provides a faster workflow already familiar to the project owner.
- Black + Flake8 + isort: established, but Ruff consolidates the relevant checks with less configuration overhead.
- Pyright: strong alternative; mypy was selected because it is already familiar and sufficient for the intended strict typing policy.
- GPLv3: stronger copyleft, but Apache 2.0 better supports the portfolio/open-engineering purpose of this repository.

## Consequences

The repository has several automated quality gates from its first milestone. This adds some configuration up front, but should reduce later migration work and make quality expectations explicit to contributors and reviewers.
