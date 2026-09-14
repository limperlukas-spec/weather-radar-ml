import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from weather_radar_ml.config.experiment import ComponentConfig
from weather_radar_ml.config.schema import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    RunConfig,
    TrackingConfig,
    TrainingConfig,
)
from weather_radar_ml.models.domain import ModelSpec
from weather_radar_ml.training.forecast_artifact import (
    ForecastArtifactBatch,
    write_forecast_artifact,
)
from weather_radar_ml.training.pipeline import (
    run_experiment,
    write_reference_forecast_artifact,
)


def _spec() -> ModelSpec:
    return ModelSpec(
        name="unet",
        dynamic_input_features=("rain",),
        target_features=("rain",),
        history_steps=2,
        forecast_steps=2,
    )


def _batch(
    sample_ids: tuple[str, ...],
    *,
    offset: float,
) -> ForecastArtifactBatch:
    batch_size = len(sample_ids)
    target = np.arange(batch_size * 2 * 1 * 2 * 2, dtype=np.float32).reshape(
        batch_size, 2, 1, 2, 2
    )
    target += offset
    learned = target + 0.5
    persistence = target - 0.25
    persistence = np.maximum(persistence, 0.0)
    valid = np.ones_like(target, dtype=np.bool_)
    return ForecastArtifactBatch(
        sample_ids=sample_ids,
        learned_mm_h=learned,
        persistence_mm_h=persistence,
        target_mm_h=target,
        valid_mask=valid,
    )


def test_forecast_artifact_preserves_order_shapes_units_and_integrity(
    tmp_path: Path,
) -> None:
    result = write_forecast_artifact(
        tmp_path,
        artifact_id="forecast-reference",
        spec=_spec(),
        lead_minutes=(5, 10),
        batches=(
            _batch(("sample-a", "sample-b"), offset=0.0),
            _batch(("sample-c",), offset=20.0),
        ),
        split="validation",
        provenance={"seed": 42, "experiment_fingerprint": "abc123"},
        qualitative_count=2,
    )

    dataset = xr.open_zarr(result.predictions, consolidated=False)
    try:
        assert dataset["learned_mm_h"].shape == (3, 2, 1, 2, 2)
        assert dataset["persistence_mm_h"].shape == (3, 2, 1, 2, 2)
        assert dataset["target_mm_h"].shape == (3, 2, 1, 2, 2)
        assert dataset["valid_mask"].shape == (3, 2, 1, 2, 2)
        assert dataset["learned_mm_h"].attrs["units"] == "mm/h"
        assert dataset["sample_id"].values.tolist() == [
            "sample-a",
            "sample-b",
            "sample-c",
        ]
        assert dataset["lead"].values.tolist() == [5, 10]
    finally:
        dataset.close()

    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["shape"] == [3, 2, 1, 2, 2]
    assert metadata["provenance"]["seed"] == 42
    assert metadata["units"] == "mm/h"

    qualitative = json.loads(result.qualitative_samples.read_text(encoding="utf-8"))
    assert [item["sample_id"] for item in qualitative["samples"]] == [
        "sample-c",
        "sample-b",
    ]

    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["artifact_fingerprint"] == result.fingerprint
    assert manifest["sample_count"] == 3
    assert manifest["files"]["predictions"]["file_count"] > 0


def test_forecast_artifact_rejects_duplicate_sample_ids_atomically(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Duplicate sample IDs"):
        write_forecast_artifact(
            tmp_path,
            artifact_id="forecast-duplicate",
            spec=_spec(),
            lead_minutes=(5, 10),
            batches=(
                _batch(("sample-a",), offset=0.0),
                _batch(("sample-a",), offset=10.0),
            ),
            split="validation",
            provenance={},
        )

    assert not (tmp_path / "forecast-duplicate").exists()
    assert not tuple(tmp_path.glob(".forecast-duplicate.tmp-*"))


def test_reference_forecast_artifact_uses_validation_order_and_best_checkpoint(
    tmp_path: Path,
) -> None:
    config = RunConfig(
        name="forecast-artifact-smoke",
        data=DataConfig(
            name="synthetic",
            train_samples=2,
            validation_samples=2,
            history_steps=2,
            forecast_steps=2,
            dynamic_input_features=("rain",),
            target_features=("rain",),
            height=8,
            width=8,
        ),
        model=ModelConfig(name="unet", parameters={"base_channels": 2}),
        training=TrainingConfig(
            seed=17,
            epochs=2,
            batch_size=2,
            optimizer=ComponentConfig("adam", {"lr": 0.01}),
        ),
        tracking=TrackingConfig(
            uri="./mlruns",
            experiment_name="tests",
            enabled=False,
        ),
        output=OutputConfig(runs_root=tmp_path / "runs"),
    )

    run = run_experiment(config)
    artifact = write_reference_forecast_artifact(
        config,
        run,
        qualitative_count=1,
    )

    dataset = xr.open_zarr(artifact.predictions, consolidated=False)
    try:
        assert dataset["sample_id"].values.tolist() == [
            "synthetic-00000002",
            "synthetic-00000003",
        ]
        assert dataset["lead"].values.tolist() == [5, 10]
        assert bool(dataset["valid_mask"].values.all())
        assert float(dataset["learned_mm_h"].min()) >= 0.0
        assert dataset["persistence_mm_h"].values[0, 0, 0, 0, 0] == pytest.approx(3.0)
        assert dataset["target_mm_h"].values[0, 0, 0, 0, 0] == pytest.approx(3.25)
    finally:
        dataset.close()

    metadata = json.loads(artifact.metadata.read_text(encoding="utf-8"))
    provenance = metadata["provenance"]
    assert provenance["seed"] == 17
    assert provenance["best_epoch"] == run.history.best_epoch
    assert provenance["checkpoint"]["role"] == "best"
    assert len(provenance["checkpoint"]["sha256"]) == 64
    assert provenance["validation_research"] is not None


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("duplicate_ids", ValueError, "unique within"),
        ("wrong_rank", ValueError, r"shape \[B, T, F, H, W\]"),
        ("shape_mismatch", ValueError, "shapes must match"),
        ("mask_shape", ValueError, "valid_mask must match"),
        ("mask_dtype", TypeError, "boolean dtype"),
        ("sample_count", ValueError, "batch dimension"),
        ("nonfinite_target", ValueError, "non-finite target"),
        ("negative_target", ValueError, "target precipitation"),
        ("nonfinite_learned", ValueError, "learned forecast is non-finite"),
        (
            "negative_persistence",
            ValueError,
            "persistence forecast must not be negative",
        ),
    ],
)
def test_forecast_artifact_batch_rejects_invalid_physical_data(
    case: str,
    error: type[Exception],
    message: str,
) -> None:
    shape = (1, 2, 1, 2, 2)
    learned = np.ones(shape, dtype=np.float32)
    persistence = np.ones(shape, dtype=np.float32)
    target = np.ones(shape, dtype=np.float32)
    valid = np.ones(shape, dtype=np.bool_)
    sample_ids = ("sample",)

    if case == "duplicate_ids":
        learned = np.ones((2, 2, 1, 2, 2), dtype=np.float32)
        persistence = learned.copy()
        target = learned.copy()
        valid = np.ones_like(learned, dtype=np.bool_)
        sample_ids = ("sample", "sample")
    elif case == "wrong_rank":
        learned = learned[0]
        persistence = persistence[0]
        target = target[0]
        valid = valid[0]
    elif case == "shape_mismatch":
        persistence = np.ones((1, 1, 1, 2, 2), dtype=np.float32)
    elif case == "mask_shape":
        valid = np.ones((1, 1, 1, 2, 2), dtype=np.bool_)
    elif case == "mask_dtype":
        valid = np.ones(shape, dtype=np.uint8)
    elif case == "sample_count":
        sample_ids = ("sample-a", "sample-b")
    elif case == "nonfinite_target":
        target[0, 0, 0, 0, 0] = np.nan
    elif case == "negative_target":
        target[0, 0, 0, 0, 0] = -1.0
    elif case == "nonfinite_learned":
        learned[0, 0, 0, 0, 0] = np.nan
    elif case == "negative_persistence":
        persistence[0, 0, 0, 0, 0] = -1.0

    with pytest.raises(error, match=message):
        ForecastArtifactBatch(
            sample_ids=sample_ids,
            learned_mm_h=learned,
            persistence_mm_h=persistence,
            target_mm_h=target,
            valid_mask=valid,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lead_minutes": (5,)}, "forecast_steps"),
        ({"lead_minutes": (0, 5)}, "greater than zero"),
        ({"lead_minutes": (10, 5)}, "strictly increasing"),
        ({"qualitative_count": 0}, "qualitative_count"),
        ({"artifact_id": "../escape"}, "safe path segment"),
    ],
)
def test_forecast_artifact_rejects_invalid_policy(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "artifact_id": "forecast-policy",
        "spec": _spec(),
        "lead_minutes": (5, 10),
        "batches": (_batch(("sample",), offset=0.0),),
        "split": "validation",
        "provenance": {},
        "qualitative_count": 1,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        write_forecast_artifact(tmp_path, **arguments)  # type: ignore[arg-type]


def test_forecast_artifact_rejects_empty_stream_and_existing_destination(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="without prediction batches"):
        write_forecast_artifact(
            tmp_path,
            artifact_id="forecast-empty",
            spec=_spec(),
            lead_minutes=(5, 10),
            batches=(),
            split="validation",
            provenance={},
        )

    existing = tmp_path / "forecast-existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        write_forecast_artifact(
            tmp_path,
            artifact_id="forecast-existing",
            spec=_spec(),
            lead_minutes=(5, 10),
            batches=(_batch(("sample",), offset=0.0),),
            split="validation",
            provenance={},
        )
