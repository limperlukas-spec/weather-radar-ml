"""Numerically explicit precipitation transforms used by learned forecasts."""

from __future__ import annotations

import torch
from torch import Tensor


def log1p_precipitation(values: Tensor) -> Tensor:
    """Map non-negative precipitation rates from mm/h to log1p space.

    The transform intentionally contains no fitted state so that identical physical
    values always map to identical model-space values across runs and machines.
    """
    _require_floating_tensor(values, "precipitation")
    _require_finite(values, "precipitation")
    if torch.any(values < 0).item():
        raise ValueError("precipitation must not contain negative values.")
    return torch.log1p(values)


def inverse_log1p_precipitation(values: Tensor) -> Tensor:
    """Map finite model-space values back to precipitation rates in mm/h.

    This is the mathematical inverse only. It deliberately does not clip negative
    model predictions; physical post-processing belongs to the evaluation policy.
    """
    _require_floating_tensor(values, "log1p precipitation")
    _require_finite(values, "log1p precipitation")
    return torch.expm1(values)


def _require_floating_tensor(values: Tensor, name: str) -> None:
    if not torch.is_floating_point(values):
        raise TypeError(f"{name} must use a floating-point dtype.")


def _require_finite(values: Tensor, name: str) -> None:
    if not torch.all(torch.isfinite(values)).item():
        raise ValueError(f"{name} must contain only finite values.")
