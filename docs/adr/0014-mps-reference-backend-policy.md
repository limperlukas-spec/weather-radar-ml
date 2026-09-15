# ADR 0014: Apple MPS reference-backend policy

## Status

Accepted

## Context

The 0.6.6 real RADKLIM-YW reference benchmark is computationally impractical on
the MacBook Air CPU for iterative execution. Apple MPS reduces one real
train-plus-validation epoch to a practical runtime, but the current PyTorch MPS
backend cannot execute the benchmark backward pass while
`torch.use_deterministic_algorithms(True)` is enabled because
`index_put_with_accumulate_mps` has no deterministic implementation. MPS also
does not support float64 tensors, so metric accumulation is performed on CPU in
float64 while model training remains on MPS.

Silently disabling deterministic algorithms would weaken the benchmark policy.
Rejecting MPS entirely would make the fixed three-seed reference benchmark much
less practical to execute on the available hardware.

## Decision

1. Strict deterministic PyTorch algorithms remain the default reference policy.
2. The MPS smoke test may disable strict deterministic algorithms because it is
   operational only and never evaluates the held-out test split.
3. A full official MPS run is permitted only when the operator explicitly passes
   `--allow-nondeterministic-mps`. Without that acknowledgement, the runner
   rejects a non-smoke MPS invocation before training starts.
4. All three official seeds (17, 42, 73) must use the same MPS backend policy.
5. Dataset, split boundaries, architecture, optimizer, loss, seed set,
   validation-only checkpoint/reference-seed selection, and held-out test
   evaluation remain unchanged.
6. The persisted run configuration and reference-forecast provenance record the
   selected device and `deterministic_algorithms=false`; these artifacts are the
   machine-readable evidence of the backend exception.
7. Multi-seed mean and sample standard deviation remain the primary aggregate
   reporting mechanism. The benchmark is not described as bitwise reproducible
   across repeated MPS executions.
8. CPU runs continue to use strict deterministic algorithms and require no
   exception flag.

## Consequences

- The official benchmark can run in a practical amount of time on the available
  Apple Silicon hardware.
- The methodological exception is opt-in, visible, and recorded in provenance
  rather than being an implicit backend side effect.
- Fixed seeds still control initialization and data-order randomness, but they do
  not guarantee identical floating-point results when nondeterministic MPS
  kernels are involved.
- Comparisons within the official 0.6.6 result use one consistent backend policy
  for all three seeds.
- A later CUDA or MPS implementation that supports all required deterministic
  operations can restore strict determinism without changing the benchmark
  dataset or selection protocol.

## Scope

This ADR only defines execution/repeatability policy for the 0.6.6 Apple MPS
reference run. It does not change model selection, test-set isolation, metrics,
or the reference forecast artifact schema.
