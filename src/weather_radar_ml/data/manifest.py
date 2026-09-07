"""Versioned metadata describing a reproducible dataset artifact."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    """Identity and provenance metadata for a generated dataset."""

    dataset_id: str
    source_product: str
    source_version: str
    pipeline_version: str
    schema_version: str
    source_checksums: dict[str, str]

    def __post_init__(self) -> None:
        required = (
            self.dataset_id,
            self.source_product,
            self.source_version,
            self.pipeline_version,
            self.schema_version,
        )
        if any(not value.strip() for value in required):
            raise ValueError("Manifest identity fields must not be empty.")
        if not self.source_checksums:
            raise ValueError("Manifest must contain at least one source checksum.")

    def write(self, path: Path) -> None:
        """Write deterministic JSON suitable for review and hashing."""
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> "DatasetManifest":
        """Read and validate a manifest from JSON."""
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file without loading it fully into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
