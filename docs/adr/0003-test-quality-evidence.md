# ADR 0003: Measure test quality beyond code coverage

- Status: Accepted
- Date: 2026-09-04

## Context

High line coverage can coexist with weak assertions and tests that fail to detect meaningful defects. Because correctness is a primary goal of the project, test quality itself should be inspectable rather than inferred from the existence of tests.

## Decision

Use three forms of evidence: branch-aware coverage for reach, selective mutation testing for defect-detection strength, and a documented behavioral test matrix for critical project behavior and scientific reference cases.

Mutation testing is not a per-commit CI gate initially. Results for critical modules are recorded and surviving mutants are reviewed. A global mutation-score threshold will only be introduced after enough production code exists to establish a meaningful baseline.

## Alternatives considered

- Coverage percentage only: fast and useful, but insufficient as evidence of assertion quality.
- Mutation testing on every commit: stronger continuous evidence but disproportionate cost for an ML-oriented repository.

## Consequences

Testing work becomes more deliberate and measurable. The project accepts occasional additional analysis cost in exchange for stronger evidence that tests detect regressions rather than merely execute code.
