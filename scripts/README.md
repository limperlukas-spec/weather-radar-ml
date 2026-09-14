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
