"""Safe extraction helpers for RADKLIM-YW tar archives."""

import tarfile
from datetime import date
from pathlib import Path


def _extract_regular_files(
    archive: Path,
    destination: Path,
) -> tuple[Path, ...]:
    """Extract all regular files from a tar archive without preserving paths."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with tarfile.open(archive, mode="r:*") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue

            source = handle.extractfile(member)
            if source is None:
                continue

            target = destination / Path(member.name).name

            if not target.exists():
                with target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)

            extracted.append(target)

    return tuple(sorted(extracted))


def extract_day(
    archive: Path,
    day: date,
    destination: Path,
) -> tuple[Path, ...]:
    """Extract all RADKLIM-YW frames belonging to one UTC calendar day."""
    destination.mkdir(parents=True, exist_ok=True)
    stamp = day.strftime("%y%m%d")

    daily_archives: list[Path] = []

    with tarfile.open(archive, mode="r:*") as handle:
        for member in handle.getmembers():
            name = Path(member.name).name

            if not member.isfile() or stamp not in name:
                continue

            source = handle.extractfile(member)
            if source is None:
                continue

            target = destination / name

            if not target.exists():
                with target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)

            daily_archives.append(target)

    if not daily_archives:
        raise ValueError(
            f"Archive contains no RADKLIM-YW archive for {day.isoformat()}."
        )

    frames: list[Path] = []

    for daily_archive in daily_archives:
        frames.extend(_extract_regular_files(daily_archive, destination))

    if not frames:
        raise ValueError(
            f"Daily archive contains no RADKLIM-YW frames for {day.isoformat()}."
        )

    return tuple(sorted(frames))
