"""PyTorch-facing forecasting model contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from weather_radar_ml.models.domain import ModelSpec


@runtime_checkable
class ForecastModel(Protocol):
    """Minimal interface required by the 0.5 training foundation.

    Dynamic inputs use ``[B, T_in, C_in, H, W]``. Optional static inputs use
    ``[B, C_static, H, W]``. The returned prediction uses
    ``[B, T_out, C_out, H, W]``.

    Implementations may be ordinary ``torch.nn.Module`` instances. Requiring
    only this protocol keeps the training layer independent of a particular
    base class or higher-level training framework.
    """

    @property
    def spec(self) -> ModelSpec: ...  # pragma: no cover

    def __call__(
        self,
        dynamic: Tensor,
        static: Tensor | None = None,
    ) -> Tensor: ...  # pragma: no cover
