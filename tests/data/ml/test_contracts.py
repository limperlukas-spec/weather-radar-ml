from collections.abc import Iterable

from weather_radar_ml.data.ml import (
    FittableSampleTransform,
    RadarDatasetProtocol,
    RadarSample,
    SampleCatalogProtocol,
    SampleSelection,
    SampleTransform,
    SplitStrategy,
    TemporalSplitRule,
)


class _Dataset:
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> RadarSample:
        raise IndexError(index)

    def get(self, sample_id: str) -> RadarSample:
        raise KeyError(sample_id)


class _Catalog:
    def query(self, selection: SampleSelection) -> RadarDatasetProtocol:
        del selection
        return _Dataset()


class _SplitStrategy:
    def rules(self) -> tuple[TemporalSplitRule, ...]:
        return ()


class _Transform:
    def transform(self, sample: RadarSample) -> RadarSample:
        return sample


class _FittableTransform(_Transform):
    def fit(self, samples: Iterable[RadarSample]) -> None:
        tuple(samples)


def test_runtime_checkable_contracts_accept_structural_implementations() -> None:
    assert isinstance(_Dataset(), RadarDatasetProtocol)
    assert isinstance(_Catalog(), SampleCatalogProtocol)
    assert isinstance(_SplitStrategy(), SplitStrategy)
    assert isinstance(_Transform(), SampleTransform)
    assert isinstance(_FittableTransform(), FittableSampleTransform)
