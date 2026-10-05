"""Typed configuration for the 0.7 classical-nowcasting benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import log10
from typing import Any, cast

_MOTION_METHODS = frozenset({"lucaskanade", "vet"})


@dataclass(frozen=True, slots=True)
class MotionCandidate:
    """One predeclared motion-estimation candidate for validation selection."""

    method: str
    history_frames: int

    def __post_init__(self) -> None:
        method = self.method.strip().lower()
        if method not in _MOTION_METHODS:
            raise ValueError(
                "motion method must be one of "
                f"{sorted(_MOTION_METHODS)!r}, got {self.method!r}."
            )
        if self.history_frames < 2:
            raise ValueError("motion history_frames must be at least two.")
        if method == "vet" and self.history_frames != 2:
            raise ValueError("VET is restricted to exactly two input frames.")
        object.__setattr__(self, "method", method)


@dataclass(frozen=True, slots=True)
class SProgConfig:
    """Accepted deterministic S-PROG configuration for the reference benchmark."""

    n_cascade_levels: int = 6
    ar_order: int = 2
    bandpass_filter: str = "gaussian"
    decomposition_method: str = "fft"
    extrapolation_method: str = "semilagrangian"
    conditional: bool = False
    probmatching_method: str = "cdf"

    def __post_init__(self) -> None:
        if self.n_cascade_levels <= 0:
            raise ValueError("sprog.n_cascade_levels must be greater than zero.")
        if self.ar_order <= 0:
            raise ValueError("sprog.ar_order must be greater than zero.")
        for field_name in (
            "bandpass_filter",
            "decomposition_method",
            "extrapolation_method",
            "probmatching_method",
        ):
            value = getattr(self, field_name).strip().lower()
            if not value:
                raise ValueError(f"sprog.{field_name} must not be empty.")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class ClassicalNowcastConfig:
    """Framework-independent configuration of the deterministic S-PROG method."""

    name: str = "sprog"
    precip_threshold_mm_h: float = 0.1
    dry_value_db: float = -15.0
    outval: str = "min"
    sprog: SProgConfig = field(default_factory=SProgConfig)

    def __post_init__(self) -> None:
        name = self.name.strip().lower()
        if name != "sprog":
            raise ValueError("The 0.7 classical reference method must be 'sprog'.")
        if self.precip_threshold_mm_h <= 0.0:
            raise ValueError("precip_threshold_mm_h must be greater than zero.")
        threshold_db = 10.0 * log10(self.precip_threshold_mm_h)
        if self.dry_value_db >= threshold_db:
            raise ValueError(
                "dry_value_db must be below the dB value of the rain threshold."
            )
        outval = self.outval.strip().lower()
        if outval != "min":
            raise ValueError("The 0.7 boundary policy requires outval='min'.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "outval", outval)


@dataclass(frozen=True, slots=True)
class ClassicalSelectionConfig:
    """Validation-only motion-selection contract for milestone 0.7."""

    selection_split: str
    criterion: str
    candidates: tuple[MotionCandidate, ...]

    def __post_init__(self) -> None:
        split = self.selection_split.strip().lower()
        criterion = self.criterion.strip().lower()
        if split != "validation":
            raise ValueError("Classical method selection must use validation only.")
        if criterion != "log1p_mse":
            raise ValueError("Classical method selection must use log1p_mse.")
        if not self.candidates:
            raise ValueError("At least one classical motion candidate is required.")
        identities = tuple(
            (candidate.method, candidate.history_frames)
            for candidate in self.candidates
        )
        if len(identities) != len(set(identities)):
            raise ValueError("Classical motion candidates must be unique.")
        object.__setattr__(self, "selection_split", split)
        object.__setattr__(self, "criterion", criterion)


def classical_nowcast_config_from_mapping(
    mapping: Mapping[str, Any],
) -> ClassicalNowcastConfig:
    """Convert a plain mapping into the typed S-PROG benchmark configuration."""

    sprog_raw = _require_mapping(mapping.get("sprog", {}), "classical.sprog")
    return ClassicalNowcastConfig(
        name=str(mapping.get("name", "sprog")),
        precip_threshold_mm_h=float(mapping.get("precip_threshold_mm_h", 0.1)),
        dry_value_db=float(mapping.get("dry_value_db", -15.0)),
        outval=str(mapping.get("outval", "min")),
        sprog=SProgConfig(
            n_cascade_levels=int(sprog_raw.get("n_cascade_levels", 6)),
            ar_order=int(sprog_raw.get("ar_order", 2)),
            bandpass_filter=str(sprog_raw.get("bandpass_filter", "gaussian")),
            decomposition_method=str(sprog_raw.get("decomposition_method", "fft")),
            extrapolation_method=str(
                sprog_raw.get("extrapolation_method", "semilagrangian")
            ),
            conditional=bool(sprog_raw.get("conditional", False)),
            probmatching_method=str(sprog_raw.get("probmatching_method", "cdf")),
        ),
    )


def classical_selection_config_from_mapping(
    mapping: Mapping[str, Any],
) -> ClassicalSelectionConfig:
    """Convert a plain mapping into the typed validation-selection contract."""

    raw_candidates = mapping.get("candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(
        raw_candidates, (str, bytes)
    ):
        raise TypeError("classical selection candidates must be a sequence.")
    candidates = tuple(
        _motion_candidate_from_mapping(item, index=index)
        for index, item in enumerate(raw_candidates)
    )
    return ClassicalSelectionConfig(
        selection_split=str(mapping.get("selection_split", "validation")),
        criterion=str(mapping.get("criterion", "log1p_mse")),
        candidates=candidates,
    )


def _motion_candidate_from_mapping(value: object, *, index: int) -> MotionCandidate:
    mapping = _require_mapping(value, f"classical selection candidate {index}")
    if "method" not in mapping or "history_frames" not in mapping:
        raise KeyError(
            f"classical selection candidate {index} requires method and history_frames."
        )
    return MotionCandidate(
        method=str(mapping["method"]),
        history_frames=int(mapping["history_frames"]),
    )


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return cast(Mapping[str, Any], value)
