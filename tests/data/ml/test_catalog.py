import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from weather_radar_ml.data.ml.catalog import SCHEMA_VERSION, SQLiteSampleCatalog
from weather_radar_ml.data.ml.domain import SampleSelection, SpatialWindowMode


def _dt(hour: int) -> datetime:
    return datetime(2023, 9, 1, hour, tzinfo=UTC)


def _catalog(tmp_path) -> tuple[SQLiteSampleCatalog, int, int]:
    catalog = SQLiteSampleCatalog(tmp_path / "samples.sqlite")
    catalog.initialize()
    catalog.add_classification_profile(
        profile_id="precip_event_v1",
        definition_json='{"version": 1}',
    )
    catalog.add_dataset_build(
        dataset_build_id="dataset-a1b2",
        dataset_name="radklim-yw",
        sample_schema_version=1,
        feature_schema_id="features-v1",
        classification_profile_id="precip_event_v1",
    )
    catalog.add_source(
        dataset_build_id="dataset-a1b2",
        source_id="radklim",
        role="primary",
        artifact_id="prepared-v1",
        checksum="abc123",
    )
    train = catalog.add_split(dataset_build_id="dataset-a1b2", name="train")
    test = catalog.add_split(dataset_build_id="dataset-a1b2", name="test")
    window = catalog.add_spatial_window(
        dataset_build_id="dataset-a1b2",
        x=0,
        y=0,
        width=64,
        height=64,
        mode=SpatialWindowMode.PATCH,
    )
    return catalog, train, test, window


def _add_sample(
    catalog: SQLiteSampleCatalog,
    *,
    sample_id: str,
    input_start: datetime,
    split_id: int,
    window_id: int,
    input_event: str | None = "light",
    target_event: str | None = "heavy",
) -> None:
    catalog.add_sample(
        sample_id=sample_id,
        dataset_build_id="dataset-a1b2",
        feature_schema_id="features-v1",
        input_start=input_start,
        input_end=input_start + timedelta(minutes=55),
        target_start=input_start + timedelta(minutes=60),
        target_end=input_start + timedelta(minutes=175),
        window_id=window_id,
        split_id=split_id,
        input_precip_mean=1.0,
        input_precip_max=3.0,
        input_precip_p95=2.0,
        input_wet_fraction=0.4,
        target_precip_mean=4.0,
        target_precip_max=30.0,
        target_precip_p95=20.0,
        target_wet_fraction=0.7,
        input_event_class=input_event,
        target_event_class=target_event,
    )


def test_initialize_sets_schema_versions_and_seeds_event_classes(tmp_path) -> None:
    catalog = SQLiteSampleCatalog(tmp_path / "samples.sqlite")
    catalog.initialize()

    assert catalog.schema_version() == SCHEMA_VERSION
    with sqlite3.connect(catalog.path) as connection:
        metadata = connection.execute(
            "SELECT value FROM schema_metadata WHERE key='schema_version'"
        ).fetchone()
        event_classes = connection.execute(
            "SELECT name FROM event_classes ORDER BY name"
        ).fetchall()

    assert metadata == (str(SCHEMA_VERSION),)
    assert [row[0] for row in event_classes] == [
        "dry",
        "extreme",
        "heavy",
        "light",
        "moderate",
    ]


def test_foreign_keys_are_enforced(tmp_path) -> None:
    catalog = SQLiteSampleCatalog(tmp_path / "samples.sqlite")
    catalog.initialize()

    with pytest.raises(sqlite3.IntegrityError):
        catalog.add_source(
            dataset_build_id="missing",
            source_id="radklim",
            role="primary",
            artifact_id="prepared-v1",
            checksum="abc",
        )


def test_only_one_primary_source_is_allowed_per_build(tmp_path) -> None:
    catalog, _, _, _ = _catalog(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        catalog.add_source(
            dataset_build_id="dataset-a1b2",
            source_id="other",
            role="primary",
            artifact_id="prepared-v2",
            checksum="def456",
        )


def test_spatial_windows_are_reused(tmp_path) -> None:
    catalog, _, _, window = _catalog(tmp_path)

    again = catalog.add_spatial_window(
        dataset_build_id="dataset-a1b2",
        x=0,
        y=0,
        width=64,
        height=64,
        mode=SpatialWindowMode.PATCH,
    )

    assert again == window


def test_sample_round_trip_preserves_metadata_and_utc(tmp_path) -> None:
    catalog, train, _, window = _catalog(tmp_path)
    _add_sample(
        catalog,
        sample_id="20230901T0000_x0_y0_deadbeef",
        input_start=_dt(0),
        split_id=train,
        window_id=window,
    )

    sample = catalog.get("20230901T0000_x0_y0_deadbeef")

    assert sample.input_start == _dt(0)
    assert sample.split == "train"
    assert sample.window_id == window
    assert sample.input_event_class == "light"
    assert sample.target_event_class == "heavy"
    assert sample.target_precip_max == 30.0


def test_missing_sample_raises_key_error(tmp_path) -> None:
    catalog, _, _, _ = _catalog(tmp_path)

    with pytest.raises(KeyError):
        catalog.get("missing")


def test_query_filters_and_uses_deterministic_order(tmp_path) -> None:
    catalog, train, test, window = _catalog(tmp_path)
    _add_sample(
        catalog,
        sample_id="later",
        input_start=_dt(2),
        split_id=train,
        window_id=window,
    )
    _add_sample(
        catalog,
        sample_id="earlier",
        input_start=_dt(1),
        split_id=train,
        window_id=window,
        target_event="extreme",
    )
    _add_sample(
        catalog,
        sample_id="test",
        input_start=_dt(3),
        split_id=test,
        window_id=window,
    )

    train_rows = catalog.query(SampleSelection(split="train"))
    extreme_rows = catalog.query(SampleSelection(target_event_classes=("extreme",)))
    bounded_rows = catalog.query(SampleSelection(start_time=_dt(2), end_time=_dt(3)))

    assert [row.sample_id for row in train_rows] == ["earlier", "later"]
    assert [row.sample_id for row in extreme_rows] == ["earlier"]
    assert [row.sample_id for row in bounded_rows] == ["later"]
    assert len(catalog) == 3


def test_nullable_event_classification_is_allowed(tmp_path) -> None:
    catalog, train, _, window = _catalog(tmp_path)
    _add_sample(
        catalog,
        sample_id="unknown-events",
        input_start=_dt(0),
        split_id=train,
        window_id=window,
        input_event=None,
        target_event=None,
    )

    sample = catalog.get("unknown-events")
    assert sample.input_event_class is None
    assert sample.target_event_class is None


def test_wet_fraction_check_constraint_is_enforced(tmp_path) -> None:
    catalog, train, _, window = _catalog(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        catalog.add_sample(
            sample_id="invalid-wet-fraction",
            dataset_build_id="dataset-a1b2",
            feature_schema_id="features-v1",
            input_start=_dt(0),
            input_end=_dt(0) + timedelta(minutes=55),
            target_start=_dt(1),
            target_end=_dt(2),
            window_id=window,
            split_id=train,
            input_wet_fraction=1.5,
        )


def test_composite_identity_constraint_catches_duplicate_logical_sample(
    tmp_path,
) -> None:
    catalog, train, _, window = _catalog(tmp_path)
    _add_sample(
        catalog,
        sample_id="first-hash",
        input_start=_dt(0),
        split_id=train,
        window_id=window,
    )

    with pytest.raises(sqlite3.IntegrityError):
        _add_sample(
            catalog,
            sample_id="different-hash",
            input_start=_dt(0),
            split_id=train,
            window_id=window,
        )


def test_sample_timestamps_must_be_timezone_aware(tmp_path) -> None:
    catalog, train, _, window = _catalog(tmp_path)

    with pytest.raises(ValueError, match="timezone-aware"):
        catalog.add_sample(
            sample_id="naive",
            dataset_build_id="dataset-a1b2",
            feature_schema_id="features-v1",
            input_start=datetime(2023, 9, 1),
            input_end=_dt(1),
            target_start=_dt(2),
            target_end=_dt(3),
            window_id=window,
            split_id=train,
        )
