"""Acquisition of immutable RADKLIM-YW source archives from DWD Open Data."""

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

RADKLIM_YW_BASE_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/grids_germany/5_minutes/"
    "radolan/reproc/2017_002/bin"
)


@dataclass(frozen=True)
class DwdRadklimYwMonthlySource:
    """Download one monthly RADKLIM-YW binary archive without modifying it."""

    year: int
    month: int
    base_url: str = RADKLIM_YW_BASE_URL

    def __post_init__(self) -> None:
        if self.year < 2001:
            raise ValueError("RADKLIM-YW is not available before 2001.")
        if not 1 <= self.month <= 12:
            raise ValueError("Month must be between 1 and 12.")

    @property
    def filename(self) -> str:
        """Return the DWD archive name for the selected month."""
        return f"YW2017.002_{self.year}{self.month:02d}.tar"

    @property
    def url(self) -> str:
        """Return the authoritative DWD Open Data URL."""
        return f"{self.base_url}/{self.year}/{self.filename}"

    def acquire(self, destination: Path) -> tuple[Path, ...]:
        """Download the archive once and keep an existing raw artifact immutable."""
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / self.filename
        if target.exists():
            return (target,)

        partial = target.with_suffix(target.suffix + ".part")
        try:
            with urlopen(self.url) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            partial.replace(target)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return (target,)
