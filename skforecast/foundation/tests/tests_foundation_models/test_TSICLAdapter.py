# Unit test TSICLAdapter
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import TSICLAdapter
from .fixtures_adapters import (
    y, exog, y_wide, y_dict, exog_shared,
    FakeTSICL,
    prepare_fit_args, prepare_predict_args
)


# ==============================================================================
# Tests TSICLAdapter.__init__
# ==============================================================================
def test_TSICLAdapter_init_default_params():
    """
    Test that default parameter values are set correctly and class-level
    attributes are properly initialised.
    """
    adapter = TSICLAdapter(model_id="taharnbl/TS-ICL")
    assert adapter.model_id == "taharnbl/TS-ICL"
    assert adapter.checkpoint_version == "tsicl-v1.ckpt"
    assert adapter.context_length == 4096
    assert adapter.device == "auto"
    assert adapter.allow_auto_download is True
    assert adapter._model is None
    assert adapter.context_ is None
    assert adapter.context_exog_ is None
    assert adapter.is_fitted is False
    assert TSICLAdapter.allow_exog is True


@pytest.mark.parametrize(
    "context_length",
    [0, -1, None, 1.5, "not_int"],
    ids=lambda cl: f"context_length={cl}"
)
def test_TSICLAdapter_init_ValueError_when_context_length_invalid(context_length):
    """
    Test that __init__ raises ValueError for non-positive-integer
    context_length values.
    """
    with pytest.raises(ValueError, match=re.escape("`context_length` must be a positive integer")):
        TSICLAdapter(
            model_id="taharnbl/TS-ICL", context_length=context_length
        )


def test_TSICLAdapter_init_custom_params_stored():
    """
    Test that custom constructor parameters are stored correctly.
    """
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL",
        checkpoint_version="tsicl-v2.ckpt",
        context_length=512,
        device="cpu",
        allow_auto_download=False,
    )
    assert adapter.checkpoint_version == "tsicl-v2.ckpt"
    assert adapter.context_length == 512
    assert adapter.device == "cpu"
    assert adapter.allow_auto_download is False


# ==============================================================================
# Tests TSICLAdapter.get_params / set_params
# ==============================================================================
def test_TSICLAdapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns all expected keys with the values set at
    construction.
    """
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL",
        context_length=512,
        device="cpu",
    )
    params = adapter.get_params()
    assert set(params.keys()) == {
        "model_id", "checkpoint_version", "context_length",
        "device", "allow_auto_download",
    }
    assert params["model_id"] == "taharnbl/TS-ICL"
    assert params["context_length"] == 512
    assert params["device"] == "cpu"
    assert params["allow_auto_download"] is True


@pytest.mark.parametrize(
    "params, match",
    [
        ({"context_length": 0}, "`context_length` must be a positive integer"),
        ({"context_length": -1}, "`context_length` must be a positive integer"),
        ({"unknown_param": 42}, "Invalid parameter"),
    ],
    ids=["context_length=0", "context_length=-1", "unknown_param"]
)
def test_TSICLAdapter_set_params_ValueError_when_invalid(params, match):
    """
    Test that set_params raises ValueError for invalid parameter values or
    unknown parameter names.
    """
    adapter = TSICLAdapter(model_id="taharnbl/TS-ICL")
    with pytest.raises(ValueError, match=re.escape(match)):
        adapter.set_params(**params)


@pytest.mark.parametrize(
    "param, value, resets_model",
    [
        ("checkpoint_version", "tsicl-v2.ckpt", True),
        ("allow_auto_download", False, True),
        ("model_id", "taharnbl/TS-ICL-large", False),
        ("device", "cpu", False),
        ("context_length", 128, False),
    ],
    ids=lambda x: str(x)
)
def test_TSICLAdapter_set_params_updates_and_resets_model(
    param, value, resets_model
):
    """
    Test that set_params updates the given parameter and resets _model
    only when checkpoint_version or allow_auto_download changes. Returns
    self.
    """
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL", model=FakeTSICL()
    )
    assert adapter._model is not None
    result = adapter.set_params(**{param: value})
    assert result is adapter
    if resets_model:
        assert adapter._model is None
    else:
        assert adapter._model is not None


# ==============================================================================
# Tests TSICLAdapter.fit
# ==============================================================================
def test_TSICLAdapter_fit_error_handling():
    """
    Test fit raises TypeError for unsupported series types, ValueError for
    empty dict, and TypeError for non-Series dict values.
    """
    adapter = TSICLAdapter(model_id="taharnbl/TS-ICL")

    with pytest.raises(TypeError):
        context, context_exog = prepare_fit_args(np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        context, context_exog = prepare_fit_args({})

    with pytest.raises(TypeError, match=re.escape("all series must be a named pandas Series")):
        context, context_exog = prepare_fit_args(
            {"s1": np.array([1.0, 2.0, 3.0])}
        )


@pytest.mark.parametrize(
    "context_length, expected_len",
    [(10, 10), (25, 25), (50, 50), (100, 50)],
    ids=lambda x: f"{x}"
)
def test_TSICLAdapter_fit_output_single_series(context_length, expected_len):
    """
    Test fit on a single series: returns self, sets is_fitted=True,
    stores history trimmed to context_length,
    and does not modify the input series.
    """
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL", context_length=context_length
    )
    y_copy = y.copy()
    context, context_exog = prepare_fit_args(y, context_length=context_length)
    result = adapter.fit(
        context=context, context_exog=context_exog
    )

    assert result is adapter
    assert adapter.is_fitted is True
    hist = next(iter(adapter.context_.values()))
    assert len(hist) == expected_len
    pd.testing.assert_series_equal(hist, y.iloc[-expected_len:])
    pd.testing.assert_series_equal(y, y_copy)


@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_TSICLAdapter_fit_output_multi_series(series_input):
    """
    Test fit on multi-series input (wide DataFrame or dict): sets is_fitted
    and stores a dict of Series keyed by series
    names, each trimmed to context_length.
    """
    context_length = 10
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL", context_length=context_length
    )
    context, context_exog = prepare_fit_args(series_input, context_length=context_length)
    adapter.fit(
        context=context, context_exog=context_exog
    )

    assert adapter.is_fitted is True
    assert set(adapter.context_.keys()) == {"s1", "s2"}
    for name, s in adapter.context_.items():
        assert isinstance(s, pd.Series)
        assert len(s) == context_length


def test_TSICLAdapter_fit_exog_stored_and_trimmed():
    """
    Test that exog is stored in context_exog_, trimmed to context_length,
    and that None exog maps all series to None.
    """
    context_length = 15
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL", context_length=context_length
    )

    # Fit with exog
    context, context_exog = prepare_fit_args(y, exog=exog, context_length=context_length)
    adapter.fit(
        context=context, context_exog=context_exog
    )
    hist_exog = next(iter(adapter.context_exog_.values()))
    assert len(hist_exog) == context_length
    pd.testing.assert_frame_equal(hist_exog, exog.iloc[-context_length:])

    # Fit without exog → context_exog_ is None
    ctx2, ctx_exog2 = prepare_fit_args(y)
    adapter.fit(
        context=ctx2, context_exog=ctx_exog2
    )
    assert adapter.context_exog_ is None


def test_TSICLAdapter_fit_multiseries_exog_broadcast_and_per_series():
    """
    Test that a shared exog DataFrame is broadcast to all series, and that
    a per-series exog dict stores correctly (present key → stored, absent
    key → None). Context_length trims exog per series.
    """
    context_length = 8
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL", context_length=context_length
    )

    # Shared exog broadcast
    context, context_exog = prepare_fit_args(y_dict, exog=exog_shared, context_length=context_length)
    adapter.fit(
        context=context, context_exog=context_exog
    )
    for name in y_dict:
        assert adapter.context_exog_[name] is not None
        assert len(adapter.context_exog_[name]) == context_length

    # Per-series exog dict: s1 has exog, s2 does not
    ctx2, ctx_exog2 = prepare_fit_args(
        y_dict, exog={"s1": exog_shared.copy()}
    )
    adapter.fit(
        context=ctx2, context_exog=ctx_exog2
    )
    assert adapter.context_exog_["s1"] is not None
    assert adapter.context_exog_["s2"] is None


# ==============================================================================
# Tests TSICLAdapter.predict — single series
# ==============================================================================
def test_TSICLAdapter_predict_point_forecast_single_series():
    """
    Test point forecast (quantiles=None) on a single series: returns a
    dict with a single array of shape (steps, 1), correct index, and
    values = 0.5 (FakeTSICL's quantile-equals-level convention with
    query_levels=[0.5]).
    """
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL", model=FakeTSICL()
    )
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=12)
    raw = adapter.predict(
        steps=12, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )

    assert isinstance(raw, dict)
    assert list(raw.keys()) == ["sales"]
    arr = raw["sales"]
    assert arr.shape == (12, 1)
    np.testing.assert_array_almost_equal(arr[:, 0], np.full(12, 0.5))


def test_TSICLAdapter_predict_quantile_forecast_single_series():
    """
    Test quantile forecast on a single series: returns dict with correct
    shape, and each quantile column equals its level (FakeTSICL property).
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL", model=FakeTSICL()
    )
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=5)
    raw = adapter.predict(
        steps=5, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=quantiles
    )

    arr = raw["sales"]
    assert arr.shape == (5, 3)
    for i, q in enumerate(quantiles):
        np.testing.assert_array_almost_equal(arr[:, i], np.full(5, q))


# ==============================================================================
# Tests TSICLAdapter.predict — multi-series
# ==============================================================================
@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_TSICLAdapter_predict_point_forecast_multi_series(series_input):
    """
    Test point forecast on multi-series: returns a dict with one array per
    series, each of shape (steps, 1) with value 0.5.
    """
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL", model=FakeTSICL()
    )
    ctx, ctx_exog = prepare_fit_args(series_input)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=5)
    raw = adapter.predict(
        steps=5, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )

    assert set(raw.keys()) == {"s1", "s2"}
    for name in ["s1", "s2"]:
        assert raw[name].shape == (5, 1)
        np.testing.assert_array_almost_equal(raw[name][:, 0], np.full(5, 0.5))


def test_TSICLAdapter_predict_quantile_forecast_multi_series():
    """
    Test quantile forecast on multi-series: returns a dict with one array
    per series, each of shape (steps, n_quantiles) with correct values.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL", model=FakeTSICL()
    )
    ctx, ctx_exog = prepare_fit_args(y_dict)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=4)
    raw = adapter.predict(
        steps=4, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=quantiles
    )

    for name in ["s1", "s2"]:
        assert raw[name].shape == (4, 3)
        for i, q in enumerate(quantiles):
            np.testing.assert_array_almost_equal(raw[name][:, i], np.full(4, q))


# ==============================================================================
# Tests TSICLAdapter.predict — model receives correct args
# ==============================================================================
def test_TSICLAdapter_predict_model_receives_correct_args():
    """
    Test that the model's forecast() receives the correct steps,
    quantile_levels, number of inputs, and context_length.
    """
    model = FakeTSICL()
    context_length = 30
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL", model=model, context_length=context_length
    )
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    # Point forecast → quantile_levels=[0.5], single input
    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=9)
    adapter.predict(
        steps=9, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    assert model.last_prediction_length == 9
    assert model.last_quantile_levels == [0.5]
    assert len(model.last_inputs) == 1
    assert model.last_context_length == context_length

    # Custom quantiles → forwarded
    adapter.predict(
        steps=3, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=[0.05, 0.95],
    )
    assert model.last_quantile_levels == [0.05, 0.95]


# ==============================================================================
# Tests TSICLAdapter.predict — exog forwarding
# ==============================================================================
def test_TSICLAdapter_predict_past_and_future_covariates_forwarded():
    """
    Test that past exog (from fit history) is forwarded as past_covariates,
    future exog is forwarded as future_covariates, and absent exog results
    in no covariates key.
    """
    model = FakeTSICL()
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL", model=model
    )

    # Fit with exog → predict should pass past_covariates
    ctx, ctx_exog = prepare_fit_args(y, exog=exog)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    future = pd.DataFrame(
        {"feat_a": np.arange(6, dtype=float)},
        index=pd.date_range("2024-03-01", periods=6, freq="ME")
    )
    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=6, exog=future)
    adapter.predict(
        steps=6, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    assert "past_covariates" in model.last_inputs[0]
    assert "future_covariates" in model.last_inputs[0]
    assert set(model.last_inputs[0]["past_covariates"].keys()) == {"feat_a", "feat_b"}
    assert len(model.last_inputs[0]["future_covariates"]["feat_a"]) == 6


def test_TSICLAdapter_predict_context_length_trims_history():
    """
    Test that the history passed to the model is trimmed to
    context_length when longer.
    """
    context_length = 10
    model = FakeTSICL()
    adapter = TSICLAdapter(
        model_id="taharnbl/TS-ICL",
        model=model,
        context_length=context_length
    )
    ctx, ctx_exog = prepare_fit_args(y, context_length=context_length)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=5)
    adapter.predict(
        steps=5, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    assert len(model.last_inputs[0]["target"]) == context_length


# ==============================================================================
# Tests TSICLAdapter._build_tsicl_input
# ==============================================================================
_past_exog_idx = pd.date_range("2020-01-01", periods=30, freq="ME")
_future_exog_idx = pd.date_range("2022-07-01", periods=12, freq="ME")
_adapter_for_build = TSICLAdapter(model_id="taharnbl/TS-ICL")
_target_30 = np.arange(30, dtype=float)


def test_TSICLAdapter_build_tsicl_input_target_only():
    """
    Test _build_tsicl_input returns only 'target' key when no exog is passed,
    and converts integer target to float32.
    """
    int_target = np.arange(10, dtype=int)
    result = _adapter_for_build._build_tsicl_input(context=int_target)
    assert set(result.keys()) == {"target"}
    assert result["target"].dtype == np.float32
    np.testing.assert_array_equal(result["target"], int_target.astype(np.float32))


@pytest.mark.parametrize(
    "past_exog, future_exog, expected_keys",
    [
        (
            pd.DataFrame(
                {"feat_a": np.arange(30, dtype=float)}, index=_past_exog_idx
            ),
            None,
            {"target", "past_covariates"}
        ),
        (
            pd.Series(
                np.arange(30, dtype=float), index=_past_exog_idx, name="feat_a"
            ),
            None,
            {"target", "past_covariates"}
        ),
        (
            None,
            pd.DataFrame(
                {"feat_a": np.arange(12, dtype=float)}, index=_future_exog_idx
            ),
            {"target", "future_covariates"}
        ),
        (
            pd.DataFrame(
                {"feat_a": np.arange(30, dtype=float)}, index=_past_exog_idx
            ),
            pd.DataFrame(
                {"feat_a": np.arange(12, dtype=float)}, index=_future_exog_idx
            ),
            {"target", "past_covariates", "future_covariates"}
        ),
    ],
    ids=["past_df", "past_series", "future_df", "both"]
)
def test_TSICLAdapter_build_tsicl_input_with_covariates(
    past_exog, future_exog, expected_keys
):
    """
    Test _build_tsicl_input includes the correct covariate keys depending
    on which exog arguments are provided (DataFrame, Series, or None).
    """
    result = _adapter_for_build._build_tsicl_input(
        context=_target_30, context_exog=past_exog, exog=future_exog
    )
    assert set(result.keys()) == expected_keys
    if "past_covariates" in result:
        assert "feat_a" in result["past_covariates"]
    if "future_covariates" in result:
        assert "feat_a" in result["future_covariates"]


@pytest.mark.parametrize(
    "col_data",
    [
        pd.Series(np.arange(30, dtype=int)),
        pd.Series(np.arange(30, dtype=float)),
        pd.Series([True, False, True] * 10),
        pd.array([1, 2, None, 4, 5] * 6, dtype="Int64"),
        pd.array([1.1, 2.2, None, 4.4, 5.5] * 6, dtype="Float64"),
        pd.array([True, False, None, True, False] * 6, dtype="boolean"),
    ],
    ids=[
        "int→float32", "float→float32", "bool→float32",
        "nullable_int→float32", "nullable_float→float32", "nullable_bool→float32",
    ]
)
def test_TSICLAdapter_build_tsicl_input_dtype_handling(col_data):
    """
    Test _to_covariate_array and _build_tsicl_input handle numeric dtype
    conversion: numeric/bool → float32, nullable pandas dtypes → float32
    with NaN.
    """
    if isinstance(col_data, (pd.arrays.IntegerArray, pd.arrays.FloatingArray,
                             pd.arrays.BooleanArray)):
        df = pd.DataFrame({"feat": col_data}, index=_past_exog_idx)
    else:
        df = pd.DataFrame({"feat": col_data.values}, index=_past_exog_idx)

    result = _adapter_for_build._build_tsicl_input(
        context=_target_30, context_exog=df
    )
    arr = result["past_covariates"]["feat"]
    assert arr.dtype == np.float32

    if isinstance(col_data, (pd.arrays.IntegerArray, pd.arrays.FloatingArray,
                             pd.arrays.BooleanArray)):
        assert np.isnan(arr[2])


@pytest.mark.parametrize(
    "col_data",
    [
        pd.Series(["sunny", "cloudy", "rainy"] * 10),
        pd.Categorical(["spring", "summer", "fall", "winter"] * 7 + ["spring", "summer"]),
    ],
    ids=["string_series", "categorical"]
)
def test_TSICLAdapter_build_tsicl_input_ValueError_non_numeric_covariate(col_data):
    """
    Test _build_tsicl_input raises ValueError when a covariate column is
    neither numeric nor boolean (TS-ICL only supports numeric covariates).
    """
    df = pd.DataFrame({"feat": col_data}, index=_past_exog_idx) if not isinstance(col_data, pd.Series) \
        else pd.DataFrame({"feat": col_data.values}, index=_past_exog_idx)

    with pytest.raises(ValueError, match="TSICLAdapter supports only numeric covariates"):
        _adapter_for_build._build_tsicl_input(context=_target_30, context_exog=df)


@pytest.mark.parametrize(
    "col_data, expected_error",
    [
        (np.arange(10, dtype=int), False),
        (np.arange(10, dtype=float), False),
        (np.array([True, False] * 5), False),
        (np.array(["sunny", "cloudy"] * 5), True),
    ],
    ids=["np_int", "np_float", "np_bool", "np_str"],
)
def test_TSICLAdapter_to_covariate_array_non_pandas_inputs(col_data, expected_error):
    """
    Test _to_covariate_array fallback path for non-pandas inputs: numpy
    numeric/bool arrays convert to float32, non-numeric arrays raise
    ValueError.
    """
    if expected_error:
        with pytest.raises(ValueError, match="TSICLAdapter supports only numeric covariates"):
            TSICLAdapter._to_covariate_array(col_data)
    else:
        arr = TSICLAdapter._to_covariate_array(col_data)
        assert arr.dtype == np.float32
