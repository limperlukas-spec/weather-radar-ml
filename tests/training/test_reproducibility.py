import random

import numpy as np
import torch

from weather_radar_ml.config.experiment import TrainingSettings
from weather_radar_ml.training.reproducibility import configure_reproducibility


def test_configure_reproducibility_repeats_supported_random_sequences() -> None:
    settings = TrainingSettings(epochs=1, batch_size=2, seed=1234)

    try:
        configure_reproducibility(settings)
        first = (random.random(), float(np.random.random()), torch.rand(3))

        configure_reproducibility(settings)
        second = (random.random(), float(np.random.random()), torch.rand(3))

        assert first[0] == second[0]
        assert first[1] == second[1]
        torch.testing.assert_close(first[2], second[2])
    finally:
        torch.use_deterministic_algorithms(False)


def test_configure_reproducibility_applies_deterministic_policy() -> None:
    try:
        configure_reproducibility(
            TrainingSettings(
                epochs=1,
                batch_size=1,
                seed=1,
                deterministic_algorithms=True,
            )
        )
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(False)
