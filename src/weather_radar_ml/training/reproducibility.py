"""Global random-state configuration for reproducible forecast experiments."""

from __future__ import annotations

import random

import numpy as np
import torch

from weather_radar_ml.config.experiment import TrainingSettings


def configure_reproducibility(settings: TrainingSettings) -> None:
    """Seed supported RNGs and apply the requested PyTorch determinism policy."""
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    torch.use_deterministic_algorithms(settings.deterministic_algorithms)
