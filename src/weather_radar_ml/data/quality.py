"""Quality and provenance states for canonical radar observations."""

from enum import IntEnum


class ObservationQuality(IntEnum):
    """Origin of a canonical radar value.

    Numeric values are stable storage codes and therefore part of the data schema.
    """

    MISSING = 0
    OBSERVED = 1
    INTERPOLATED = 2
    EXTRAPOLATED = 3
