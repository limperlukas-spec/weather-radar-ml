# Testing Strategy

## Purpose

The test suite is intended to provide evidence that important behavior is correct and regressions are detected. Code coverage is useful evidence of test reach, but it is not treated as proof of test quality.

## Quality model

Test quality is evaluated on three complementary levels:

1. **Reach:** line and branch coverage indicate which implementation paths execute during tests.
2. **Defect-detection strength:** mutation testing measures whether selected tests detect small, plausible implementation defects.
3. **Behavioral relevance:** a maintained test matrix records which critical behaviors, invariants, integration boundaries, and scientific reference cases are covered.

## CI policy

Every change must pass:

- Ruff linting
- Ruff formatting check
- mypy strict type checking for project code
- pytest
- branch-aware coverage with the configured project threshold
- package build

The initial project-wide coverage gate is 85%. The threshold is a guardrail, not a target: tests must not be added merely to increase the percentage.

## Mutation testing policy

Mutation testing is run selectively for critical deterministic modules, especially evaluation logic, data transformations, and configuration/domain validation. It is not executed on every commit because ML and integration paths can make exhaustive mutation runs disproportionately expensive.

For every targeted mutation run:

- the mutation score is recorded;
- surviving mutants are reviewed;
- each survivor is either killed by an improved test or documented as equivalent/non-actionable;
- the result is referenced from the relevant milestone or experiment documentation.

A global mutation-score threshold is intentionally deferred until enough production code exists to establish a meaningful baseline. Once established, the project should ratchet rather than relax that baseline without a documented reason.

## Test levels

| Level | Primary purpose | Typical targets |
| --- | --- | --- |
| Unit | Isolated deterministic behavior | transforms, metrics, schemas |
| Integration | Boundary compatibility | configuration composition, data adapters, pipelines |
| Scientific/reference | Numerical or domain correctness | known metric values, benchmark cases |
| Invariant/property | General rules across many inputs | metric bounds, shape/time invariants |
| End-to-end | User-visible workflow confidence | selected experiment smoke runs |

## Test design rule

A test should normally fail for a meaningful behavioral regression. Tests that merely mirror implementation details are avoided unless those details are themselves part of a public contract.
