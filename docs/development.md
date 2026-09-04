# Development Guide

## Environment

The project targets Python 3.11 and uses `uv` for environment and dependency management.

```bash
uv lock
uv sync --all-groups
```

## Local quality gate

Run before completing a milestone or opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv build
```

To apply formatting:

```bash
uv run ruff format .
```

## Git workflow

Development follows trunk-based principles with short-lived feature branches. `main` should remain releasable. Larger milestones may use a pull request even for solo development because the PR provides a reviewable record of scope, checks, and design changes.

## Definition of Done

A scoped implementation is complete when:

- its agreed requirement is met;
- appropriate unit/integration/reference tests exist;
- all configured automated quality checks pass;
- relevant documentation and ADRs are updated;
- public APIs are documented where appropriate;
- no known critical TODO remains in scope;
- a manual or end-to-end smoke test is completed when relevant.

Ideas outside the current scope are recorded for later work rather than silently expanding the active milestone.

## Lockfile policy

`uv.lock` is committed to the repository and is the authoritative lockfile for
development and CI environments. CI installs dependencies with
`uv sync --all-groups --locked` to prevent implicit dependency resolution.
