"""Compact fully convolutional U-Net for the first learned radar forecast."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from weather_radar_ml.models.domain import ModelSpec


class UNetForecast(nn.Module):
    """Forecast a fixed number of future radar frames from a radar history.

    Public tensors use the project forecast contract ``[B, T, F, H, W]``. Time
    and feature axes are flattened to channels only inside the U-Net, keeping the
    external contract explicit while allowing a conventional 2-D architecture.
    """

    def __init__(
        self,
        spec: ModelSpec | None = None,
        *,
        input_steps: int = 12,
        input_features: int = 1,
        output_steps: int = 6,
        output_features: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if spec is None:
            spec = ModelSpec(
                name="unet",
                dynamic_input_features=tuple(
                    f"dynamic_{index}" for index in range(input_features)
                ),
                target_features=tuple(
                    f"target_{index}" for index in range(output_features)
                ),
                history_steps=input_steps,
                forecast_steps=output_steps,
            )
        elif spec.static_input_features:
            raise ValueError("UNetForecast does not support static inputs in 0.6.")

        self._spec = spec
        self.input_steps = _positive(spec.history_steps, "input_steps")
        self.input_features = _positive(
            len(spec.dynamic_input_features), "input_features"
        )
        self.output_steps = _positive(spec.forecast_steps, "output_steps")
        self.output_features = _positive(len(spec.target_features), "output_features")
        base_channels = _positive(base_channels, "base_channels")

        input_channels = self.input_steps * self.input_features
        output_channels = self.output_steps * self.output_features

        self.encoder_1 = _ConvBlock(input_channels, base_channels)
        self.encoder_2 = _ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = _ConvBlock(base_channels * 2, base_channels * 4)
        self.decoder_2 = _ConvBlock(base_channels * 6, base_channels * 2)
        self.decoder_1 = _ConvBlock(base_channels * 3, base_channels)
        self.output_head = nn.Conv2d(base_channels, output_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    @property
    def spec(self) -> ModelSpec:
        """Return the forecast contract implemented by this U-Net."""
        return self._spec

    def forward(
        self,
        dynamic_inputs: Tensor,
        static_inputs: Tensor | None = None,
    ) -> Tensor:
        """Return forecasts using the public ``[B, T, F, H, W]`` contract."""
        if static_inputs is not None:
            raise ValueError("UNetForecast does not support static inputs in 0.6.")
        batch_size, height, width = self._validate_input(dynamic_inputs)
        flattened = dynamic_inputs.reshape(
            batch_size,
            self.input_steps * self.input_features,
            height,
            width,
        )

        encoder_1 = self.encoder_1(flattened)
        encoder_2 = self.encoder_2(self.pool(encoder_1))
        bottleneck = self.bottleneck(self.pool(encoder_2))

        decoder_2 = F.interpolate(
            bottleneck,
            size=encoder_2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder_2 = self.decoder_2(torch.cat((decoder_2, encoder_2), dim=1))

        decoder_1 = F.interpolate(
            decoder_2,
            size=encoder_1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder_1 = self.decoder_1(torch.cat((decoder_1, encoder_1), dim=1))
        output = cast(Tensor, self.output_head(decoder_1))

        return output.reshape(
            batch_size,
            self.output_steps,
            self.output_features,
            height,
            width,
        )

    def _validate_input(self, dynamic_inputs: Tensor) -> tuple[int, int, int]:
        if dynamic_inputs.ndim != 5:
            raise ValueError(
                "dynamic_inputs must have shape [B, T, F, H, W]; "
                f"got {tuple(dynamic_inputs.shape)}."
            )
        batch_size, steps, features, height, width = dynamic_inputs.shape
        if steps != self.input_steps:
            raise ValueError(f"expected {self.input_steps} input steps, got {steps}.")
        if features != self.input_features:
            raise ValueError(
                f"expected {self.input_features} input features, got {features}."
            )
        if height < 4 or width < 4:
            raise ValueError("spatial dimensions must each be at least 4 pixels.")
        if not torch.is_floating_point(dynamic_inputs):
            raise TypeError("dynamic_inputs must use a floating-point dtype.")
        return batch_size, height, width


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.layers(inputs))


def _positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value
