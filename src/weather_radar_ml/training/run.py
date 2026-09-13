"""Runtime metadata captured for one forecast experiment run."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import torch

from weather_radar_ml.config.experiment import ExperimentConfig


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Environment and identity metadata attached to a concrete run."""

    run_id: str
    config_fingerprint: str
    created_at: datetime
    git_commit: str | None
    python_version: str
    numpy_version: str
    torch_version: str
    platform: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_name(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "config_fingerprint",
            _require_name(self.config_fingerprint, "config_fingerprint"),
        )
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware.")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        if self.git_commit is not None:
            object.__setattr__(
                self,
                "git_commit",
                _require_name(self.git_commit, "git_commit"),
            )
        for field_name in (
            "python_version",
            "numpy_version",
            "torch_version",
            "platform",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )


def create_run_metadata(
    config: ExperimentConfig,
    *,
    run_id: str | None = None,
    created_at: datetime | None = None,
    git_commit: str | None = None,
) -> RunMetadata:
    """Capture immutable identity and environment metadata for one run."""
    return RunMetadata(
        run_id=uuid4().hex if run_id is None else run_id,
        config_fingerprint=config.fingerprint,
        created_at=datetime.now(UTC) if created_at is None else created_at,
        git_commit=git_commit,
        python_version=platform.python_version(),
        numpy_version=str(np.__version__),
        torch_version=str(torch.__version__),
        platform=platform.platform(),
    )


def _require_name(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty.")
    return normalized
