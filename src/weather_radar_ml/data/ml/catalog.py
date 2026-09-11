"""SQLite-backed immutable sample catalog for ML dataset builds."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from weather_radar_ml.data.ml.domain import SampleSelection, SpatialWindowMode

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SpatialWindowRecord:
    """Persisted integer pixel window on the primary prepared grid."""

    id: int
    x: int
    y: int
    width: int
    height: int
    mode: SpatialWindowMode


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """Persisted sample reference and analysis metadata."""

    id: int
    sample_id: str
    dataset_build_id: str
    feature_schema_id: str
    input_start: datetime
    input_end: datetime
    target_start: datetime
    target_end: datetime
    window_id: int
    split: str
    input_precip_mean: float | None = None
    input_precip_max: float | None = None
    input_precip_p95: float | None = None
    input_wet_fraction: float | None = None
    target_precip_mean: float | None = None
    target_precip_max: float | None = None
    target_precip_p95: float | None = None
    target_wet_fraction: float | None = None
    input_event_class: str | None = None
    target_event_class: str | None = None


class SQLiteSampleCatalog:
    """Read/write access to one versioned SQLite sample catalog.

    The database is treated as an immutable build artifact after construction.
    This class deliberately exposes metadata and sample references only; radar
    arrays are loaded by the later dataset layer.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        """Create a new catalog schema or verify an existing compatible one."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA_SQL)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.execute(
                """
                INSERT INTO schema_metadata(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )
            _seed_event_classes(connection)
            connection.commit()

    def schema_version(self) -> int:
        """Return the SQLite technical schema version."""
        with self._connect() as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
        if row is None:
            raise RuntimeError("Unable to read SQLite schema version.")
        return int(row[0])

    def add_dataset_build(
        self,
        *,
        dataset_build_id: str,
        dataset_name: str,
        sample_schema_version: int,
        feature_schema_id: str,
        classification_profile_id: str,
    ) -> None:
        """Insert the identity metadata for one immutable dataset build."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dataset_builds(
                    id, name, sample_schema_version, feature_schema_id,
                    classification_profile_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    dataset_build_id,
                    dataset_name,
                    sample_schema_version,
                    feature_schema_id,
                    classification_profile_id,
                ),
            )
            connection.commit()

    def add_source(
        self,
        *,
        dataset_build_id: str,
        source_id: str,
        role: str,
        artifact_id: str,
        checksum: str,
    ) -> None:
        """Register a prepared source used by a dataset build."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dataset_sources(
                    dataset_build_id, source_id, role, artifact_id, checksum
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (dataset_build_id, source_id, role, artifact_id, checksum),
            )
            connection.commit()

    def add_classification_profile(
        self,
        *,
        profile_id: str,
        definition_json: str,
    ) -> None:
        """Register an immutable versioned event-classification profile."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO classification_profiles(id, definition_json)
                VALUES (?, ?)
                """,
                (profile_id, definition_json),
            )
            connection.commit()

    def add_split(self, *, dataset_build_id: str, name: str) -> int:
        """Create one split entry and return its internal integer key."""
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO splits(dataset_build_id, name) VALUES (?, ?)",
                (dataset_build_id, name),
            )
            connection.commit()
            return _lastrowid(cursor)

    def add_spatial_window(
        self,
        *,
        dataset_build_id: str,
        x: int,
        y: int,
        width: int,
        height: int,
        mode: SpatialWindowMode,
    ) -> int:
        """Insert or reuse one spatial window and return its internal key."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO spatial_windows(
                    dataset_build_id, x, y, width, height, mode
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_build_id, x, y, width, height, mode)
                DO NOTHING
                """,
                (dataset_build_id, x, y, width, height, mode.value),
            )
            row = connection.execute(
                """
                SELECT id FROM spatial_windows
                WHERE dataset_build_id = ? AND x = ? AND y = ?
                  AND width = ? AND height = ? AND mode = ?
                """,
                (dataset_build_id, x, y, width, height, mode.value),
            ).fetchone()
            connection.commit()
        if row is None:
            raise RuntimeError("Unable to resolve inserted spatial window.")
        return int(row[0])

    def add_sample(
        self,
        *,
        sample_id: str,
        dataset_build_id: str,
        feature_schema_id: str,
        input_start: datetime,
        input_end: datetime,
        target_start: datetime,
        target_end: datetime,
        window_id: int,
        split_id: int,
        input_precip_mean: float | None = None,
        input_precip_max: float | None = None,
        input_precip_p95: float | None = None,
        input_wet_fraction: float | None = None,
        target_precip_mean: float | None = None,
        target_precip_max: float | None = None,
        target_precip_p95: float | None = None,
        target_wet_fraction: float | None = None,
        input_event_class: str | None = None,
        target_event_class: str | None = None,
    ) -> int:
        """Insert one sample reference and return its internal integer key."""
        timestamps = tuple(
            _utc_text(value)
            for value in (input_start, input_end, target_start, target_end)
        )
        with self._connect() as connection:
            cursor = connection.execute(
                _INSERT_SAMPLE_SQL,
                (
                    sample_id,
                    dataset_build_id,
                    feature_schema_id,
                    *timestamps,
                    window_id,
                    split_id,
                    input_precip_mean,
                    input_precip_max,
                    input_precip_p95,
                    input_wet_fraction,
                    target_precip_mean,
                    target_precip_max,
                    target_precip_p95,
                    target_wet_fraction,
                    input_event_class,
                    target_event_class,
                ),
            )
            connection.commit()
            return _lastrowid(cursor)

    def get(self, sample_id: str) -> SampleRecord:
        """Return one sample by its stable fachliche sample ID."""
        with self._connect() as connection:
            row = connection.execute(
                _SELECT_SAMPLE_SQL + " WHERE s.sample_id = ?",
                (sample_id,),
            ).fetchone()
        if row is None:
            raise KeyError(sample_id)
        return _sample_from_row(row)

    def get_spatial_window(self, window_id: int) -> SpatialWindowRecord:
        """Return one persisted spatial window by its internal key."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, x, y, width, height, mode
                FROM spatial_windows
                WHERE id = ?
                """,
                (window_id,),
            ).fetchone()
        if row is None:
            raise KeyError(window_id)
        return SpatialWindowRecord(
            id=int(row["id"]),
            x=int(row["x"]),
            y=int(row["y"]),
            width=int(row["width"]),
            height=int(row["height"]),
            mode=SpatialWindowMode(str(row["mode"])),
        )

    def query(
        self,
        selection: SampleSelection | None = None,
    ) -> tuple[SampleRecord, ...]:
        """Return a deterministic storage-independent sample selection."""
        if selection is None:
            selection = SampleSelection()
        clauses: list[str] = []
        parameters: list[object] = []

        if selection.split is not None:
            clauses.append("sp.name = ?")
            parameters.append(selection.split)
        _append_in_filter(
            clauses,
            parameters,
            "iec.name",
            selection.input_event_classes,
        )
        _append_in_filter(
            clauses,
            parameters,
            "tec.name",
            selection.target_event_classes,
        )
        if selection.start_time is not None:
            clauses.append("s.input_start >= ?")
            parameters.append(_utc_text(selection.start_time))
        if selection.end_time is not None:
            clauses.append("s.input_start < ?")
            parameters.append(_utc_text(selection.end_time))

        sql = _SELECT_SAMPLE_SQL
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY s.input_start, s.window_id, s.sample_id"

        with self._connect() as connection:
            rows = connection.execute(sql, tuple(parameters)).fetchall()
        return tuple(_sample_from_row(row) for row in rows)

    def __len__(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM samples").fetchone()
        return 0 if row is None else int(row[0])

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()


def _append_in_filter(
    clauses: list[str],
    parameters: list[object],
    column: str,
    values: Sequence[str],
) -> None:
    if not values:
        return
    placeholders = ", ".join("?" for _ in values)
    clauses.append(f"{column} IN ({placeholders})")
    parameters.extend(values)


def _sample_from_row(row: sqlite3.Row) -> SampleRecord:
    return SampleRecord(
        id=int(row["id"]),
        sample_id=str(row["sample_id"]),
        dataset_build_id=str(row["dataset_build_id"]),
        feature_schema_id=str(row["feature_schema_id"]),
        input_start=_parse_utc(str(row["input_start"])),
        input_end=_parse_utc(str(row["input_end"])),
        target_start=_parse_utc(str(row["target_start"])),
        target_end=_parse_utc(str(row["target_end"])),
        window_id=int(row["window_id"]),
        split=str(row["split"]),
        input_precip_mean=_optional_float(row["input_precip_mean"]),
        input_precip_max=_optional_float(row["input_precip_max"]),
        input_precip_p95=_optional_float(row["input_precip_p95"]),
        input_wet_fraction=_optional_float(row["input_wet_fraction"]),
        target_precip_mean=_optional_float(row["target_precip_mean"]),
        target_precip_max=_optional_float(row["target_precip_max"]),
        target_precip_p95=_optional_float(row["target_precip_p95"]),
        target_wet_fraction=_optional_float(row["target_wet_fraction"]),
        input_event_class=_optional_str(row["input_event_class"]),
        target_event_class=_optional_str(row["target_event_class"]),
    )


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Catalog timestamps must be timezone-aware.")
    return value.astimezone(UTC).isoformat()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Persisted catalog timestamp is not timezone-aware.")
    return parsed.astimezone(UTC)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str, bytes)):
        raise TypeError(f"Unsupported SQLite numeric value: {type(value).__name__}")
    return float(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _lastrowid(cursor: sqlite3.Cursor) -> int:
    if cursor.lastrowid is None:
        raise RuntimeError("SQLite did not return a row id.")
    return int(cursor.lastrowid)


def _seed_event_classes(connection: sqlite3.Connection) -> None:
    connection.executemany(
        "INSERT OR IGNORE INTO event_classes(name) VALUES (?)",
        ((name,) for name in ("dry", "light", "moderate", "heavy", "extreme")),
    )


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_metadata(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classification_profiles(
    id TEXT PRIMARY KEY,
    definition_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_builds(
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sample_schema_version INTEGER NOT NULL CHECK(sample_schema_version > 0),
    feature_schema_id TEXT NOT NULL,
    classification_profile_id TEXT NOT NULL,
    FOREIGN KEY(classification_profile_id) REFERENCES classification_profiles(id)
);

CREATE TABLE IF NOT EXISTS dataset_sources(
    id INTEGER PRIMARY KEY,
    dataset_build_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('primary', 'auxiliary')),
    artifact_id TEXT NOT NULL,
    checksum TEXT NOT NULL,
    UNIQUE(dataset_build_id, source_id),
    FOREIGN KEY(dataset_build_id) REFERENCES dataset_builds(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_primary_source_per_build
ON dataset_sources(dataset_build_id)
WHERE role = 'primary';

CREATE TABLE IF NOT EXISTS spatial_windows(
    id INTEGER PRIMARY KEY,
    dataset_build_id TEXT NOT NULL,
    x INTEGER NOT NULL CHECK(x >= 0),
    y INTEGER NOT NULL CHECK(y >= 0),
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    mode TEXT NOT NULL CHECK(mode IN ('patch', 'full_frame')),
    UNIQUE(dataset_build_id, x, y, width, height, mode),
    FOREIGN KEY(dataset_build_id) REFERENCES dataset_builds(id)
);

CREATE TABLE IF NOT EXISTS splits(
    id INTEGER PRIMARY KEY,
    dataset_build_id TEXT NOT NULL,
    name TEXT NOT NULL CHECK(name IN ('train', 'validation', 'test')),
    UNIQUE(dataset_build_id, name),
    FOREIGN KEY(dataset_build_id) REFERENCES dataset_builds(id)
);

CREATE TABLE IF NOT EXISTS event_classes(
    name TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS samples(
    id INTEGER PRIMARY KEY,
    sample_id TEXT NOT NULL UNIQUE,
    dataset_build_id TEXT NOT NULL,
    feature_schema_id TEXT NOT NULL,
    input_start TEXT NOT NULL,
    input_end TEXT NOT NULL,
    target_start TEXT NOT NULL,
    target_end TEXT NOT NULL,
    window_id INTEGER NOT NULL,
    split_id INTEGER NOT NULL,
    input_precip_mean REAL,
    input_precip_max REAL,
    input_precip_p95 REAL,
    input_wet_fraction REAL CHECK(
        input_wet_fraction IS NULL OR
        (input_wet_fraction >= 0 AND input_wet_fraction <= 1)
    ),
    target_precip_mean REAL,
    target_precip_max REAL,
    target_precip_p95 REAL,
    target_wet_fraction REAL CHECK(
        target_wet_fraction IS NULL OR
        (target_wet_fraction >= 0 AND target_wet_fraction <= 1)
    ),
    input_event_class TEXT,
    target_event_class TEXT,
    CHECK(input_start < input_end),
    CHECK(input_end < target_start),
    CHECK(target_start <= target_end),
    UNIQUE(
        dataset_build_id, feature_schema_id,
        input_start, input_end, target_start, target_end, window_id
    ),
    FOREIGN KEY(dataset_build_id) REFERENCES dataset_builds(id),
    FOREIGN KEY(window_id) REFERENCES spatial_windows(id),
    FOREIGN KEY(split_id) REFERENCES splits(id),
    FOREIGN KEY(input_event_class) REFERENCES event_classes(name),
    FOREIGN KEY(target_event_class) REFERENCES event_classes(name)
);

CREATE INDEX IF NOT EXISTS ix_samples_input_start ON samples(input_start);
CREATE INDEX IF NOT EXISTS ix_samples_target_start ON samples(target_start);
CREATE INDEX IF NOT EXISTS ix_samples_window_id ON samples(window_id);
CREATE INDEX IF NOT EXISTS ix_samples_split_input ON samples(split_id, input_start);
CREATE INDEX IF NOT EXISTS ix_samples_input_event ON samples(input_event_class);
CREATE INDEX IF NOT EXISTS ix_samples_target_event ON samples(target_event_class);
CREATE INDEX IF NOT EXISTS ix_samples_target_event_max
ON samples(target_event_class, target_precip_max);
"""

_INSERT_SAMPLE_SQL = """
INSERT INTO samples(
    sample_id, dataset_build_id, feature_schema_id,
    input_start, input_end, target_start, target_end,
    window_id, split_id,
    input_precip_mean, input_precip_max, input_precip_p95, input_wet_fraction,
    target_precip_mean, target_precip_max, target_precip_p95, target_wet_fraction,
    input_event_class, target_event_class
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_SAMPLE_SQL = """
SELECT
    s.id,
    s.sample_id,
    s.dataset_build_id,
    s.feature_schema_id,
    s.input_start,
    s.input_end,
    s.target_start,
    s.target_end,
    s.window_id,
    sp.name AS split,
    s.input_precip_mean,
    s.input_precip_max,
    s.input_precip_p95,
    s.input_wet_fraction,
    s.target_precip_mean,
    s.target_precip_max,
    s.target_precip_p95,
    s.target_wet_fraction,
    iec.name AS input_event_class,
    tec.name AS target_event_class
FROM samples AS s
JOIN splits AS sp ON sp.id = s.split_id
LEFT JOIN event_classes AS iec ON iec.name = s.input_event_class
LEFT JOIN event_classes AS tec ON tec.name = s.target_event_class
"""
