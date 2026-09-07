"""Cross-format validation for a real RADKLIM-YW reference frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def _read_ascii(path: Path) -> np.ndarray:
    """Read DWD ESRI ASCII into canonical south-to-north orientation."""
    data = np.loadtxt(path, skiprows=6)
    data[data == -9.0] = np.nan
    return np.flipud(data)


def validate(ascii_path: Path, prepared_path: Path, time_index: int) -> None:
    """Validate one prepared frame against the official DWD ASCII product."""
    ascii_data = _read_ascii(ascii_path)
    dataset = xr.open_zarr(prepared_path)
    canonical_rate = dataset["precipitation_rate"].isel(time=time_index).values

    if ascii_data.shape != canonical_rate.shape:
        raise ValueError(
            f"Shape mismatch: ASCII {ascii_data.shape}, "
            f"prepared {canonical_rate.shape}."
        )

    ascii_missing = ~np.isfinite(ascii_data)
    prepared_missing = ~np.isfinite(canonical_rate)
    if not np.array_equal(ascii_missing, prepared_missing):
        differing = int(np.count_nonzero(ascii_missing != prepared_missing))
        raise ValueError(f"Missing-data masks differ at {differing} pixels.")

    valid = ~ascii_missing
    native_amount = canonical_rate[valid] / 12.0
    reference = ascii_data[valid]
    difference = np.abs(reference - native_amount)

    # The official ASCII representation has 0.1 resolution. Do not assume a
    # particular DWD rounding implementation at exact quantization boundaries.
    within_quantization = difference <= 0.050001
    match_fraction = float(np.mean(within_quantization))
    max_difference = float(np.max(difference))

    if match_fraction < 0.999:
        raise ValueError(
            f"Binary/ASCII agreement below threshold: {match_fraction:.6%} within 0.05."
        )
    if max_difference > 0.050001:
        raise ValueError(
            f"Maximum Binary/ASCII deviation {max_difference:.6f} exceeds 0.05."
        )

    print(f"shape={ascii_data.shape}")
    print(f"valid_pixels={int(np.count_nonzero(valid))}")
    print(f"missing_pixels={int(np.count_nonzero(ascii_missing))}")
    print("missing_masks_identical=True")
    print(f"within_0.05={match_fraction:.6%}")
    print(f"max_native_ascii_difference={max_difference:.6f}")
    print("validation=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate prepared RADKLIM-YW against official DWD ASCII."
    )
    parser.add_argument("--ascii", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--time-index", type=int, default=0)
    args = parser.parse_args()
    validate(args.ascii, args.prepared, args.time_index)


if __name__ == "__main__":
    main()
