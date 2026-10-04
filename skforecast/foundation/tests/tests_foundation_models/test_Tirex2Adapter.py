# Unit tests for the TiRex-2 adapter.
# ==============================================================================
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from skforecast.foundation._adapters import TiRex2Adapter


class FakeTimeseriesType:
    def __init__(self, target, past_covariates, future_covariates):
        assert hasattr(target, "to")
        if past_covariates is not None:
            assert hasattr(past_covariates, "to")
        if future_covariates is not None:
            assert hasattr(future_covariates, "to")
        self.target = target
        self.past_covariates = past_covariates
        self.future_covariates = future_covariates


class FakeTensor(np.ndarray):
    def to(self, device):
        return self


class FakeTorch:
    float32 = np.float32

    @staticmethod
    def as_tensor(value, dtype=None):
        return np.asarray(value, dtype=dtype).view(FakeTensor)


class FakeTirex2Model:
    def __init__(self):
        self.calls = []

    def forecast(self, timeseries, prediction_length, output_type, **kwargs):
        self.calls.append((timeseries, prediction_length, output_type, kwargs))
        # Native TiRex-2 layout: (n_targets, n_quantiles, horizon).
        return [
            np.broadcast_to(
                np.arange(1, 10, dtype=float)[None, :, None],
                (1, 9, prediction_length),
            ).copy()
            for _ in timeseries
        ]


def test_TiRex2Adapter_maps_covariates_and_native_quantiles(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tirex2",
        SimpleNamespace(TimeseriesType=FakeTimeseriesType),
    )
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    context = {"series": pd.Series([1, 2, 3, 4], index=index)}
    context_exog = {
        "series": pd.DataFrame(
            {"past": [10, 11, 12, 13], "future": [0, 1, 2, 3]}, index=index
        )
    }
    future_index = pd.date_range("2024-01-05", periods=2, freq="D")
    exog = {"series": pd.DataFrame({"future": [4, 5]}, index=future_index)}
    model = FakeTirex2Model()
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)

    result = adapter.predict(2, context, context_exog, exog, [0.1, 0.9])

    ts = model.calls[0][0][0]
    np.testing.assert_array_equal(ts.target, [[1, 2, 3, 4]])
    np.testing.assert_array_equal(ts.past_covariates, [[10, 11, 12, 13]])
    np.testing.assert_array_equal(ts.future_covariates, [[0, 1, 2, 3, 4, 5]])
    np.testing.assert_array_equal(result["series"], [[1, 9], [1, 9]])


def test_TiRex2Adapter_point_forecast_uses_median(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tirex2",
        SimpleNamespace(TimeseriesType=FakeTimeseriesType),
    )
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    model = FakeTirex2Model()
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=model)
    result = adapter.predict(
        2, {"s": pd.Series([1, 2, 3], index=index)}, None, None, None
    )
    np.testing.assert_array_equal(result["s"], [[5], [5]])


def test_TiRex2Adapter_rejects_managed_predict_kwargs():
    with pytest.raises(ValueError, match="prediction_length"):
        TiRex2Adapter(
            "NX-AI/TiRex-2",
            predict_kwargs={"prediction_length": 4},
        )


def test_TiRex2Adapter_accepts_tolerant_supported_quantiles(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tirex2",
        SimpleNamespace(TimeseriesType=FakeTimeseriesType),
    )
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    adapter = TiRex2Adapter("NX-AI/TiRex-2", model=FakeTirex2Model())
    result = adapter.predict(
        2,
        {"s": pd.Series([1, 2, 3], index=index)},
        None,
        None,
        [0.1000000005],
    )
    np.testing.assert_array_equal(result["s"], [[1], [1]])
