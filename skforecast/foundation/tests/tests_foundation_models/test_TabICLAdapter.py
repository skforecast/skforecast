# Unit test TabICLAdapter
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import TabICLAdapter
from .fixtures_adapters import (
    y, exog, y_wide, y_dict,
    FakeTabICLForecaster,
    prepare_fit_args, prepare_predict_args,
)


# ==============================================================================
# Tests TabICLAdapter.__init__
# ==============================================================================
def test_TabICLAdapter_init_default_params():
    """
    Test that default parameter values are set correctly and class-level
    attributes are properly initialised.
    """
    adapter = TabICLAdapter(model_id="soda-inria/tabicl")
    assert adapter.model_id == "soda-inria/tabicl"
    assert adapter.context_length == 4096
    assert adapter.point_estimate == "mean"
    assert adapter.tabicl_config == {}
    assert adapter.temporal_features is None
    assert adapter._model is None
    assert adapter.context_ is None
    assert adapter.context_exog_ is None
    assert adapter.is_fitted is False
    assert TabICLAdapter.allow_exog is True


def test_TabICLAdapter_init_custom_params_stored():
    """
    Test that custom constructor parameters are stored correctly.
    """
    adapter = TabICLAdapter(
        model_id="soda-inria/tabicl",
        context_length=512,
        point_estimate="median",
        tabicl_config={"n_estimators": 10},
        temporal_features=[],
    )
    assert adapter.context_length == 512
    assert adapter.point_estimate == "median"
    assert adapter.tabicl_config == {"n_estimators": 10}
    assert adapter.temporal_features == []


@pytest.mark.parametrize(
    "context_length",
    [0, -1, None, 1.5, "not_int"],
    ids=lambda cl: f"context_length={cl}"
)
def test_TabICLAdapter_init_ValueError_when_context_length_invalid(context_length):
    """
    Test that __init__ raises ValueError for non-positive-integer
    context_length values.
    """
    with pytest.raises(ValueError, match=re.escape("`context_length` must be a positive integer")):
        TabICLAdapter(model_id="soda-inria/tabicl", context_length=context_length)


@pytest.mark.parametrize(
    "point_estimate",
    ["mode", "sum", None, 123],
    ids=lambda pe: f"point_estimate={pe}"
)
def test_TabICLAdapter_init_ValueError_when_point_estimate_invalid(point_estimate):
    """
    Test that __init__ raises ValueError for unsupported point_estimate values.
    """
    with pytest.raises(ValueError, match=re.escape("`point_estimate` must be 'mean' or 'median'")):
        TabICLAdapter(model_id="soda-inria/tabicl", point_estimate=point_estimate)


# ==============================================================================
# Tests TabICLAdapter.get_params / set_params
# ==============================================================================
def test_TabICLAdapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns all expected keys with the values set at
    construction, and that tabicl_config returns None when empty.
    """
    adapter = TabICLAdapter(
        model_id="soda-inria/tabicl",
        context_length=512,
        point_estimate="median",
    )
    params = adapter.get_params()
    assert set(params.keys()) == {
        "model_id", "context_length", "point_estimate",
        "tabicl_config", "temporal_features",
    }
    assert params["model_id"] == "soda-inria/tabicl"
    assert params["context_length"] == 512
    assert params["point_estimate"] == "median"
    assert params["tabicl_config"] is None   # empty dict → None
    assert params["temporal_features"] is None


def test_TabICLAdapter_get_params_tabicl_config_non_empty():
    """
    Test that tabicl_config is returned as-is when non-empty.
    """
    adapter = TabICLAdapter(
        model_id="soda-inria/tabicl",
        tabicl_config={"n_estimators": 5},
    )
    params = adapter.get_params()
    assert params["tabicl_config"] == {"n_estimators": 5}


@pytest.mark.parametrize(
    "params, match",
    [
        ({"context_length": 0}, "`context_length` must be a positive integer"),
        ({"context_length": -5}, "`context_length` must be a positive integer"),
        ({"point_estimate": "mode"}, "`point_estimate` must be 'mean' or 'median'"),
        ({"unknown_param": 42}, "Invalid parameter"),
    ],
    ids=["context_length=0", "context_length=-5", "point_estimate=mode", "unknown_param"]
)
def test_TabICLAdapter_set_params_ValueError_when_invalid(params, match):
    """
    Test that set_params raises ValueError for invalid values or unknown keys.
    """
    adapter = TabICLAdapter(model_id="soda-inria/tabicl")
    with pytest.raises(ValueError, match=re.escape(match)):
        adapter.set_params(**params)


def test_TabICLAdapter_set_params_updates_and_returns_self():
    """
    Test that set_params updates parameters, resets the internal _model
    object, and returns self.
    """
    fake = FakeTabICLForecaster()
    adapter = TabICLAdapter(model_id="soda-inria/tabicl", model=fake)
    assert adapter._model is not None

    result = adapter.set_params(context_length=128, point_estimate="median")

    assert result is adapter
    assert adapter.context_length == 128
    assert adapter.point_estimate == "median"
    assert adapter._model is None  # reset when any param changes


def test_TabICLAdapter_set_params_tabicl_config_none_normalises_to_empty_dict():
    """
    Test that passing tabicl_config=None via set_params normalises
    the internal value to an empty dict.
    """
    adapter = TabICLAdapter(
        model_id="soda-inria/tabicl",
        tabicl_config={"n_estimators": 10},
    )
    adapter.set_params(tabicl_config=None)
    assert adapter.tabicl_config == {}


# ==============================================================================
# Tests TabICLAdapter.fit
# ==============================================================================
def test_TabICLAdapter_fit_single_series_stores_context():
    """
    Test fit on a single series: returns self, sets is_fitted=True,
    stores the context dict, and does not modify the input series.
    """
    adapter = TabICLAdapter(model_id="soda-inria/tabicl", context_length=20)
    y_copy = y.copy()
    context, context_exog = prepare_fit_args(y, context_length=20)

    result = adapter.fit(context=context, context_exog=context_exog)

    assert result is adapter
    assert adapter.is_fitted is True
    assert set(adapter.context_.keys()) == {"sales"}
    hist = adapter.context_["sales"]
    assert len(hist) == 20
    pd.testing.assert_series_equal(hist, y.iloc[-20:])
    pd.testing.assert_series_equal(y, y_copy)


@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_TabICLAdapter_fit_multi_series_stores_context(series_input):
    """
    Test fit on multi-series (wide DataFrame or dict): stores all series
    in context_ keyed by name.
    """
    adapter = TabICLAdapter(model_id="soda-inria/tabicl", context_length=15)
    context, context_exog = prepare_fit_args(series_input, context_length=15)

    adapter.fit(context=context, context_exog=context_exog)

    assert adapter.is_fitted is True
    assert set(adapter.context_.keys()) == {"s1", "s2"}
    for name, s in adapter.context_.items():
        assert isinstance(s, pd.Series)
        assert len(s) == 15


def test_TabICLAdapter_fit_stores_exog_and_handles_none_exog():
    """
    Test that exog is stored in context_exog_ and that None exog
    results in context_exog_ = None.
    """
    adapter = TabICLAdapter(model_id="soda-inria/tabicl", context_length=15)

    # Fit with exog
    context, context_exog = prepare_fit_args(y, exog=exog, context_length=15)
    adapter.fit(context=context, context_exog=context_exog)
    hist_exog = adapter.context_exog_["sales"]
    assert len(hist_exog) == 15

    # Fit without exog → context_exog_ is None
    ctx2, ctx_exog2 = prepare_fit_args(y)
    adapter.fit(context=ctx2, context_exog=ctx_exog2)
    assert adapter.context_exog_ is None


# ==============================================================================
# Tests TabICLAdapter.predict — single series
# ==============================================================================
def test_TabICLAdapter_predict_point_forecast_single_series():
    """
    Test point forecast (quantiles=None) on a single series: returns a dict
    with one entry, shape (steps, 1), and numeric values.
    """
    steps = 5
    adapter = TabICLAdapter(
        model_id="soda-inria/tabicl",
        model=FakeTabICLForecaster(),
    )
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    predictions = adapter.predict(
        steps=steps,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=None,
    )

    assert set(predictions.keys()) == {"sales"}
    arr = predictions["sales"]
    assert arr.shape == (steps, 1)
    assert np.issubdtype(arr.dtype, np.floating)


def test_TabICLAdapter_predict_quantile_forecast_single_series():
    """
    Test quantile forecast on a single series: returns (steps, n_quantiles)
    shape, and quantile values match those defined in FakeTabICLForecaster
    (each column value equals the quantile level).
    """
    steps = 6
    quantiles = [0.1, 0.5, 0.9]
    adapter = TabICLAdapter(
        model_id="soda-inria/tabicl",
        model=FakeTabICLForecaster(),
    )
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    predictions = adapter.predict(
        steps=steps,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=quantiles,
    )

    assert set(predictions.keys()) == {"sales"}
    arr = predictions["sales"]
    assert arr.shape == (steps, len(quantiles))
    # FakeTabICLForecaster returns q value itself for every step
    for col_idx, q in enumerate(quantiles):
        np.testing.assert_allclose(arr[:, col_idx], q)


# ==============================================================================
# Tests TabICLAdapter.predict — multi-series
# ==============================================================================
def test_TabICLAdapter_predict_multi_series_returns_all_keys():
    """
    Test that predict on multi-series returns one array per series with
    correct shape.
    """
    steps = 4
    adapter = TabICLAdapter(
        model_id="soda-inria/tabicl",
        model=FakeTabICLForecaster(),
    )
    context, context_exog = prepare_fit_args(y_dict)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    predictions = adapter.predict(
        steps=steps,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=[0.1, 0.9],
    )

    assert set(predictions.keys()) == {"s1", "s2"}
    for name, arr in predictions.items():
        assert arr.shape == (steps, 2)


# ==============================================================================
# Tests TabICLAdapter.predict — exog forwarding
# ==============================================================================
def test_TabICLAdapter_predict_exog_forwarded_to_context_df_and_future_df():
    """
    Test that context exog columns appear in context_df and future exog
    columns appear in future_df passed to the underlying forecaster.
    """
    steps = 3
    fake = FakeTabICLForecaster()
    adapter = TabICLAdapter(
        model_id="soda-inria/tabicl", model=fake
    )
    context, context_exog = prepare_fit_args(y, exog=exog, context_length=20)
    adapter.fit(context=context, context_exog=context_exog)

    # Prepare future exog (same columns, next `steps` rows)
    future_exog_df = exog.iloc[-steps:].copy()
    ctx, ctx_exog, future_exog = prepare_predict_args(
        adapter, steps, exog=future_exog_df
    )

    adapter.predict(
        steps=steps,
        context=ctx,
        context_exog=ctx_exog,
        exog=future_exog,
        quantiles=[0.5],
    )

    assert fake.last_context_df is not None
    assert fake.last_future_df is not None
    # context_df must contain the exog columns
    for col in exog.columns:
        assert col in fake.last_context_df.columns
    # future_df must contain the exog columns
    for col in exog.columns:
        assert col in fake.last_future_df.columns


# ==============================================================================
# Tests TabICLAdapter.predict — RangeIndex series
# ==============================================================================
def test_TabICLAdapter_predict_range_index_issues_warning_and_returns_arrays():
    """
    Test that predicting on a RangeIndex series issues a UserWarning about
    the synthetic DatetimeIndex and still returns arrays with correct shape.
    """
    steps = 4
    y_range = pd.Series(
        data=np.arange(30, dtype=float),
        index=pd.RangeIndex(30),
        name="ts",
    )
    fake = FakeTabICLForecaster()
    adapter = TabICLAdapter(model_id="soda-inria/tabicl", model=fake)
    context, context_exog = prepare_fit_args(y_range)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    with pytest.warns(UserWarning, match="non-DatetimeIndex"):
        predictions = adapter.predict(
            steps=steps,
            context=ctx,
            context_exog=ctx_exog,
            exog=future_exog,
            quantiles=[0.5],
        )

    assert "ts" in predictions
    assert predictions["ts"].shape == (steps, 1)


def test_TabICLAdapter_predict_range_index_timestamps_are_datetime():
    """
    Test that the context_df and future_df built from a RangeIndex series
    contain valid datetime timestamps.
    """
    steps = 3
    y_range = pd.Series(
        data=np.arange(20, dtype=float),
        index=pd.RangeIndex(20),
        name="ts",
    )
    fake = FakeTabICLForecaster()
    adapter = TabICLAdapter(model_id="soda-inria/tabicl", model=fake)
    context, context_exog = prepare_fit_args(y_range)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps)

    with pytest.warns(UserWarning):
        adapter.predict(
            steps=steps,
            context=ctx,
            context_exog=ctx_exog,
            exog=future_exog,
            quantiles=None,
        )

    assert pd.api.types.is_datetime64_any_dtype(fake.last_context_df["timestamp"])
    assert pd.api.types.is_datetime64_any_dtype(fake.last_future_df["timestamp"])
    # future timestamps must be strictly after context timestamps for same series
    ctx_max = fake.last_context_df["timestamp"].max()
    fut_min = fake.last_future_df["timestamp"].min()
    assert fut_min > ctx_max


# ==============================================================================
# Tests TabICLAdapter.predict — ImportError
# ==============================================================================
def test_TabICLAdapter_predict_ImportError_when_tabicl_not_installed(monkeypatch):
    """
    Test that predict raises ImportError with a clear install message when
    the tabicl package is not available and no _model is injected.
    """
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("tabicl"):
            raise ImportError("No module named 'tabicl'")
        return real_import(name, *args, **kwargs)

    adapter = TabICLAdapter(model_id="soda-inria/tabicl")
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context=context, context_exog=context_exog)
    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps=3)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="tabicl\\[forecast\\]"):
        adapter.predict(
            steps=3,
            context=ctx,
            context_exog=ctx_exog,
            exog=future_exog,
            quantiles=None,
        )


# ==============================================================================
# Tests _ADAPTER_REGISTRY and _resolve_adapter
# ==============================================================================
def test_TabICLAdapter_registered_in_registry():
    """
    Test that 'soda-inria/tabicl' resolves to TabICLAdapter via
    _resolve_adapter.
    """
    from skforecast.foundation._adapters import _resolve_adapter
    cls = _resolve_adapter("soda-inria/tabicl")
    assert cls is TabICLAdapter


def test_TabICLAdapter_used_by_FoundationModel():
    """
    Test that FoundationModel instantiates a TabICLAdapter when given
    the soda-inria/tabicl model_id.
    """
    from skforecast.foundation import FoundationModel
    fm = FoundationModel("soda-inria/tabicl")
    assert isinstance(fm.adapter, TabICLAdapter)
