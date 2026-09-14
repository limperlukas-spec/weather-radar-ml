"""Tests for the stateless precipitation transformation."""

import pytest
import torch

from weather_radar_ml.transforms.precipitation import (
    inverse_log1p_precipitation,
    log1p_precipitation,
)


def test_log1p_round_trip_preserves_precipitation() -> None:
    values = torch.tensor([0.0, 0.1, 1.0, 5.0, 25.0], dtype=torch.float32)

    restored = inverse_log1p_precipitation(log1p_precipitation(values))

    torch.testing.assert_close(restored, values)


def test_zero_remains_exactly_zero() -> None:
    zero = torch.zeros((2, 3), dtype=torch.float32)

    transformed = log1p_precipitation(zero)

    assert torch.equal(transformed, zero)
    assert torch.equal(inverse_log1p_precipitation(transformed), zero)


def test_log1p_rejects_negative_precipitation() -> None:
    with pytest.raises(ValueError, match="negative"):
        log1p_precipitation(torch.tensor([0.0, -0.1], dtype=torch.float32))


@pytest.mark.parametrize(
    "function",
    [log1p_precipitation, inverse_log1p_precipitation],
)
def test_transform_rejects_non_finite_values(function) -> None:
    with pytest.raises(ValueError, match="finite"):
        function(torch.tensor([0.0, float("nan")], dtype=torch.float32))


def test_transform_rejects_integer_tensors() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        log1p_precipitation(torch.tensor([0, 1], dtype=torch.int64))
