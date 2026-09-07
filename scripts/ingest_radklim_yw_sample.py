"""Ingest a small real RADKLIM-YW sample through the 0.3 data pipeline."""

import argparse
from datetime import date
from pathlib import Path

from weather_radar_ml.data.dwd.archive import extract_day
from weather_radar_ml.data.dwd.radklim_yw import RadklimYwDecoder
from weather_radar_ml.data.dwd.source import DwdRadklimYwMonthlySource
from weather_radar_ml.data.manifest import DatasetManifest, sha256_file
from weather_radar_ml.data.store import write_zarr


def main() -> None:
    """Download a monthly archive and persist a bounded one-day sample."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=date.fromisoformat, default=date(2023, 9, 1))
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    source = DwdRadklimYwMonthlySource(args.date.year, args.date.month)
    raw_dir = (
        args.root / "raw" / "radklim-yw" / f"{args.date.year}-{args.date.month:02d}"
    )
    archive = source.acquire(raw_dir)[0]
    extracted = extract_day(archive, args.date, raw_dir / args.date.isoformat())
    selected = extracted[: args.frames]

    sequence = RadklimYwDecoder().decode_many(selected)
    dataset_id = f"radklim-yw-{args.date.isoformat()}-{len(selected):03d}f-v1"
    prepared = args.root / "prepared" / f"{dataset_id}.zarr"
    write_zarr(sequence, prepared)

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        source_product="RADKLIM-YW",
        source_version="2017.002",
        pipeline_version="0.3.0",
        schema_version="1",
        source_checksums={archive.name: sha256_file(archive)},
    )
    manifest_path = args.root / "prepared" / f"{dataset_id}.manifest.json"
    manifest.write(manifest_path)

    print(f"source={source.url}")
    print(f"frames={len(selected)}")
    print(f"prepared={prepared}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
