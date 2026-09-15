# Scripts

Operational helper scripts belong here when they do not form part of the
installable application package.

## Real RADKLIM-YW reference data

The bounded ingestion smoke test remains available via:

```bash
uv run python scripts/ingest_radklim_yw_sample.py --date 2023-09-01 --frames 12
```

Version 0.6.6 adds a reproducible real reference workflow for the complete
September 2023 source month. It crops the fixed Dortmund-centred 128 x 128 ROI,
keeps only complete physical frames, writes the prepared Zarr plus provenance,
and builds the immutable ML sample catalog:

```bash
uv run python scripts/prepare_radklim_yw_reference.py
```

After reviewing/versioning the generated data artifacts, execute the official
three-seed benchmark with the held-out test split:

```bash
uv run python scripts/run_radklim_yw_reference.py
```

The benchmark uses seeds 17, 42 and 73. Checkpoint and reference-seed selection
are validation-only; test evaluation starts only after those choices are fixed.
Use `--device` to choose the explicitly recorded PyTorch device. Local MLflow
tracking is enabled for the official run by default; `--no-tracking` is an explicit
operational opt-out.

Before committing to a long Apple Silicon run, benchmark one real epoch through
MPS without MLflow or held-out test evaluation:

```bash
uv run python scripts/run_radklim_yw_reference.py --device mps --smoke --no-tracking
```

The full runner prints one line per completed epoch and keeps resumable seed
checkpoints below `runs/.resume-radklim-reference/` by default. A repeated full
invocation resumes compatible interrupted seed runs. Use `--no-resume` only with
an empty or different `--resume-root`; an existing workspace is rejected rather
than overwritten or silently mixed with a different configuration or Git commit.

## Apple MPS reference-backend policy

Strict PyTorch determinism remains the default for the official reference
benchmark. The current Apple MPS backend cannot execute the benchmark backward
pass with `torch.use_deterministic_algorithms(True)` because the required
`index_put_with_accumulate_mps` path has no deterministic implementation.

The one-epoch MPS smoke test therefore runs with strict deterministic algorithms
disabled. A full MPS reference benchmark requires an additional explicit
acknowledgement instead of silently relaxing the policy:

```bash
uv run python scripts/run_radklim_yw_reference.py \
  --device mps \
  --allow-nondeterministic-mps
```

The three official seeds, validation-only checkpoint/reference-seed selection,
held-out test protocol, and all model/data hyperparameters remain unchanged.
The resulting run configuration and forecast provenance record the MPS device
and `deterministic_algorithms=false`. Consequently, the run is reproducible as
a fixed protocol but is not claimed to be bitwise reproducible across repeated
MPS executions. All three official seeds must use the same backend policy.
