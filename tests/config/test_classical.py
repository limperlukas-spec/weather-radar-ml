from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from weather_radar_ml.config.classical import (
    MotionCandidate,
    classical_nowcast_config_from_mapping,
    classical_selection_config_from_mapping,
)


def test_committed_sprog_config_matches_accepted_contract() -> None:
    config = classical_nowcast_config_from_mapping(
        _load_mapping(Path("configs/classical/sprog.yaml"))
    )

    assert config.name == "sprog"
    assert config.precip_threshold_mm_h == 0.1
    assert config.dry_value_db == -15.0
    assert config.outval == "min"
    assert config.sprog.n_cascade_levels == 6
    assert config.sprog.ar_order == 2
    assert config.sprog.bandpass_filter == "gaussian"
    assert config.sprog.decomposition_method == "fft"
    assert config.sprog.extrapolation_method == "semilagrangian"
    assert config.sprog.conditional is False
    assert config.sprog.probmatching_method == "cdf"


def test_committed_motion_candidates_are_frozen_before_validation() -> None:
    config = classical_selection_config_from_mapping(
        _load_mapping(Path("configs/classical/selection.yaml"))
    )

    assert config.selection_split == "validation"
    assert config.criterion == "log1p_mse"
    assert tuple(
        (candidate.method, candidate.history_frames) for candidate in config.candidates
    ) == (
        ("vet", 2),
        ("lucaskanade", 2),
        ("lucaskanade", 3),
        ("lucaskanade", 6),
        ("lucaskanade", 12),
    )


def test_vet_rejects_more_than_two_motion_frames() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        MotionCandidate(method="vet", history_frames=3)


def test_selection_rejects_test_split() -> None:
    with pytest.raises(ValueError, match="validation only"):
        classical_selection_config_from_mapping(
            {
                "selection_split": "test",
                "criterion": "log1p_mse",
                "candidates": [{"method": "lucaskanade", "history_frames": 2}],
            }
        )


def test_selection_rejects_duplicate_candidates() -> None:
    with pytest.raises(ValueError, match="unique"):
        classical_selection_config_from_mapping(
            {
                "selection_split": "validation",
                "criterion": "log1p_mse",
                "candidates": [
                    {"method": "lucaskanade", "history_frames": 2},
                    {"method": "lucaskanade", "history_frames": 2},
                ],
            }
        )


def test_motion_candidate_rejects_invalid_method_and_short_history() -> None:
    with pytest.raises(ValueError, match="motion method"):
        MotionCandidate(method="farneback", history_frames=2)

    with pytest.raises(ValueError, match="at least two"):
        MotionCandidate(method="lucaskanade", history_frames=1)


def test_sprog_rejects_non_positive_order_and_cascade_count() -> None:
    with pytest.raises(ValueError, match="n_cascade_levels"):
        classical_nowcast_config_from_mapping({"sprog": {"n_cascade_levels": 0}})

    with pytest.raises(ValueError, match="ar_order"):
        classical_nowcast_config_from_mapping({"sprog": {"ar_order": 0}})


@pytest.mark.parametrize(
    "field_name",
    (
        "bandpass_filter",
        "decomposition_method",
        "extrapolation_method",
        "probmatching_method",
    ),
)
def test_sprog_rejects_empty_method_names(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        classical_nowcast_config_from_mapping({"sprog": {field_name: "   "}})


def test_nowcast_rejects_changes_to_frozen_reference_contract() -> None:
    with pytest.raises(ValueError, match="must be 'sprog'"):
        classical_nowcast_config_from_mapping({"name": "steps"})

    with pytest.raises(ValueError, match="greater than zero"):
        classical_nowcast_config_from_mapping({"precip_threshold_mm_h": 0.0})

    with pytest.raises(ValueError, match="below the dB value"):
        classical_nowcast_config_from_mapping({"dry_value_db": -10.0})

    with pytest.raises(ValueError, match="outval='min'"):
        classical_nowcast_config_from_mapping({"outval": "nan"})


def test_selection_rejects_invalid_criterion_and_empty_candidates() -> None:
    with pytest.raises(ValueError, match="log1p_mse"):
        classical_selection_config_from_mapping({"criterion": "rmse", "candidates": []})

    with pytest.raises(ValueError, match="At least one"):
        classical_selection_config_from_mapping({"candidates": []})


def test_selection_rejects_malformed_candidate_structures() -> None:
    with pytest.raises(TypeError, match="must be a sequence"):
        classical_selection_config_from_mapping({"candidates": "lucaskanade"})

    with pytest.raises(TypeError, match="candidate 0 must be a mapping"):
        classical_selection_config_from_mapping({"candidates": ["lucaskanade"]})

    with pytest.raises(KeyError, match="requires method and history_frames"):
        classical_selection_config_from_mapping(
            {"candidates": [{"method": "lucaskanade"}]}
        )


def test_nowcast_rejects_non_mapping_sprog_config() -> None:
    with pytest.raises(TypeError, match=r"classical\.sprog must be a mapping"):
        classical_nowcast_config_from_mapping({"sprog": "invalid"})


def _load_mapping(path: Path) -> Mapping[str, Any]:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Configuration root must be a mapping: {path}")
    return cast(Mapping[str, Any], raw)
