# Unit test TimestralAdapter
# ==============================================================================
import re
import sys
import types
import warnings
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.foundation._adapters import TimestralAdapter
from skforecast.exceptions import FeatureOutOfRangeWarning, LicenseWarning
from .fixtures_adapters import (
    y, y_wide, y_dict,
    FakeTimestralForecaster,
    TIMESTRAL_QUANTILES,
    prepare_fit_args, prepare_predict_args
)


# Helpers
# ==============================================================================
def make_adapter(**kwargs) -> TimestralAdapter:
    """
    Return a TimestralAdapter pre-loaded with FakeTimestralForecaster.
    """
    defaults = dict(
        model_id="mistralai/Timestral-28M",
        model=FakeTimestralForecaster()
    )
    defaults.update(kwargs)
    return TimestralAdapter(**defaults)


def fit_and_predict(adapter, steps, quantiles=None, exog=None):
    """
    Fit the adapter on `y` and predict `steps` ahead through the dict-based
    adapter API.
    """
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)
    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=steps, exog=exog)

    return adapter.predict(
        steps=steps, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=quantiles
    )


# ==============================================================================
# Tests TimestralAdapter.__init__
# ==============================================================================
def test_TimestralAdapter_init_default_params():
    """
    Test that default parameter values are set correctly and that the
    capability flags are class-level attributes, never shadowed on the
    instance.
    """
    adapter = TimestralAdapter(model_id="mistralai/Timestral-28M")
    assert adapter.model_id == "mistralai/Timestral-28M"
    assert adapter.context_length == 4096
    assert adapter.device == "auto"
    assert adapter.predict_kwargs == {}
    assert adapter._model is None
    assert adapter.context_ is None
    assert adapter.context_exog_ is None
    assert adapter.is_fitted is False
    assert TimestralAdapter.allow_exog is True
    assert TimestralAdapter.supports_past_only_covariates is True
    assert TimestralAdapter.supports_categorical_covariates is True
    assert TimestralAdapter.supports_heterogeneous_covariates is False
    assert TimestralAdapter.requires_equal_length_context is True
    assert TimestralAdapter.supports_nan_in_series is True
    assert "allow_exog" not in vars(adapter)
    assert TimestralAdapter.SUPPORTED_QUANTILES == [
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
    ]


@pytest.mark.parametrize(
    "value",
    [0, -1, None, "a"],
    ids=lambda x: str(x)
)
def test_TimestralAdapter_init_ValueError_when_invalid_context_length(value):
    """
    Test that __init__ raises ValueError for a non-positive-integer
    context_length.
    """
    err_msg = re.escape("`context_length` must be a positive integer")
    with pytest.raises(ValueError, match=err_msg):
        TimestralAdapter(model_id="mistralai/Timestral-28M", context_length=value)


@pytest.mark.parametrize(
    "model_id",
    [
        "mistralai/Mistral-7B-v0.1",
        "google/timesfm-3.0-pytorch",
        "autogluon/chronos-2-small",
    ],
    ids=lambda x: str(x)
)
def test_TimestralAdapter_init_ValueError_when_model_id_not_timestral(model_id):
    """
    Test that __init__ raises ValueError when model_id does not start with
    the Timestral prefix served by this adapter.
    """
    err_msg = re.escape(
        f"`model_id` must start with 'mistralai/Timestral' for TimestralAdapter. "
        f"Got {model_id!r}."
    )
    with pytest.raises(ValueError, match=err_msg):
        TimestralAdapter(model_id=model_id)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_context": 1024},
        {"per_shot_horizon_patches": 16},
        {"quantile_real_cap_k": 10},
        {"revision": "main"},
    ],
    ids=lambda x: str(x)
)
def test_TimestralAdapter_init_TypeError_when_loader_only_kwargs(kwargs):
    """
    Test that the loading options of TimestralForecaster.from_pretrained are
    not exposed as constructor parameters.
    """
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        TimestralAdapter(model_id="mistralai/Timestral-28M", **kwargs)


@pytest.mark.parametrize(
    "reserved_key",
    [
        "context", "prediction_length", "observation_mask", "past_covariates",
        "known_covariates", "categorical_past_covariates",
        "categorical_known_covariates",
    ],
)
def test_TimestralAdapter_init_ValueError_when_predict_kwargs_has_reserved_key(reserved_key):
    """
    Test that __init__ raises ValueError when predict_kwargs includes a key
    that the adapter manages internally.
    """
    err_msg = re.escape(
        f"`predict_kwargs` cannot include {[reserved_key]}. These arguments are "
        f"managed internally by TimestralAdapter."
    )
    with pytest.raises(ValueError, match=err_msg):
        TimestralAdapter(
            model_id="mistralai/Timestral-28M",
            predict_kwargs={reserved_key: "value"},
        )


def test_TimestralAdapter_init_predict_kwargs_stored_by_reference():
    """
    Test that predict_kwargs is stored by reference (not copied), so the
    same object is returned by get_params and the adapter stays compatible
    with sklearn.base.clone.
    """
    original = {"symmetric_averaging": False}
    adapter = TimestralAdapter(
        model_id="mistralai/Timestral-28M",
        predict_kwargs=original
    )
    assert adapter.predict_kwargs is original


# ==============================================================================
# Tests TimestralAdapter.get_params / set_params
# ==============================================================================
def test_TimestralAdapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns exactly the Timestral parameters with the
    correct values, and that predict_kwargs is None when empty.
    """
    adapter = TimestralAdapter(
        model_id="mistralai/Timestral-209M",
        context_length=512,
        device="cpu",
        predict_kwargs={"symmetric_averaging": False},
    )
    params = adapter.get_params()
    assert params == {
        "model_id":       "mistralai/Timestral-209M",
        "context_length": 512,
        "device":         "cpu",
        "predict_kwargs": {"symmetric_averaging": False},
    }

    adapter2 = TimestralAdapter(model_id="mistralai/Timestral-28M")
    assert adapter2.get_params()["predict_kwargs"] is None


@pytest.mark.parametrize(
    "params, match",
    [
        ({"context_length": -1}, "`context_length` must be a positive integer"),
        ({"unknown_param": 42}, "Invalid parameter(s) for TimestralAdapter"),
        ({"max_context": 1024}, "Invalid parameter(s) for TimestralAdapter"),
    ],
    ids=["context_length=-1", "unknown_param", "max_context"]
)
def test_TimestralAdapter_set_params_ValueError_when_invalid(params, match):
    """
    Test that set_params raises ValueError for invalid values, unknown
    parameter names, and the loading options that are not exposed.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match=re.escape(match)):
        adapter.set_params(**params)


def test_TimestralAdapter_set_params_ValueError_when_model_id_not_timestral():
    """
    Test that set_params rejects a model_id of another provider and leaves the
    adapter untouched (validation runs before anything is applied).
    """
    adapter = make_adapter()
    err_msg = re.escape(
        "`model_id` must start with 'mistralai/Timestral' for TimestralAdapter. "
        "Got 'mistralai/Mistral-7B-v0.1'."
    )
    with pytest.raises(ValueError, match=err_msg):
        adapter.set_params(model_id="mistralai/Mistral-7B-v0.1")

    assert adapter.model_id == "mistralai/Timestral-28M"
    assert adapter._model is not None


def test_TimestralAdapter_set_params_ValueError_when_predict_kwargs_has_reserved_key():
    """
    Test that set_params raises ValueError when predict_kwargs includes a
    key that the adapter manages internally.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match=re.escape("`predict_kwargs` cannot include")):
        adapter.set_params(predict_kwargs={"known_covariates": None})


@pytest.mark.parametrize(
    "param, value",
    [
        ("model_id", "mistralai/Timestral-209M"),
        ("device", "cuda"),
    ],
    ids=lambda x: str(x)
)
def test_TimestralAdapter_set_params_updates_and_resets_model(param, value):
    """
    Test that set_params updates model_id or device, resets _model (both are
    forwarded to TimestralForecaster.from_pretrained), returns self, and does
    not reset _model when the value is unchanged.
    """
    adapter = make_adapter(device="cpu")
    assert adapter._model is not None

    adapter.set_params(**{param: getattr(adapter, param)})
    assert adapter._model is not None

    result = adapter.set_params(**{param: value})
    assert result is adapter
    assert getattr(adapter, param) == value
    assert adapter._model is None


@pytest.mark.parametrize(
    "param, value",
    [
        ("context_length", 2048),
        ("predict_kwargs", {"symmetric_averaging": False}),
    ],
    ids=lambda x: str(x)
)
def test_TimestralAdapter_set_params_no_reset_for_non_reload_keys(param, value):
    """
    Test that changing context_length or predict_kwargs via set_params
    updates the attribute but does not reset the cached model, since neither
    is passed to TimestralForecaster.from_pretrained.
    """
    adapter = make_adapter()
    adapter.set_params(**{param: value})

    assert adapter._model is not None
    assert getattr(adapter, param) == value


# ==============================================================================
# Tests TimestralAdapter.fit
# ==============================================================================
@pytest.mark.parametrize(
    "context_length, expected_len",
    [(10, 10), (50, 50), (100, 50)],
    ids=lambda x: f"{x}"
)
def test_TimestralAdapter_fit_output_single_series(context_length, expected_len):
    """
    Test fit on a single series with exog: returns self, sets
    is_fitted=True, stores history trimmed to context_length and stores
    context_exog_, and does not modify the input series.
    """
    exog_df = pd.DataFrame({"feat": np.arange(50, dtype=float)}, index=y.index)
    adapter = make_adapter(context_length=context_length)
    y_copy = y.copy()
    ctx, ctx_exog = prepare_fit_args(y, exog=exog_df, context_length=context_length)
    result = adapter.fit(context=ctx, context_exog=ctx_exog)

    assert result is adapter
    assert adapter.is_fitted is True
    hist = next(iter(adapter.context_.values()))
    assert len(hist) == expected_len
    pd.testing.assert_series_equal(hist, y.iloc[-expected_len:])
    pd.testing.assert_series_equal(y, y_copy)
    pd.testing.assert_frame_equal(adapter.context_exog_["sales"], ctx_exog["sales"])


@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_TimestralAdapter_fit_output_multi_series(series_input):
    """
    Test fit on multi-series input: sets is_fitted=True, stores a dict of
    Series keyed by series names, each trimmed to context_length.
    """
    context_length = 10
    adapter = make_adapter(context_length=context_length)
    ctx, ctx_exog = prepare_fit_args(series_input, context_length=context_length)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    assert adapter.is_fitted is True
    assert set(adapter.context_.keys()) == {"s1", "s2"}
    for s in adapter.context_.values():
        assert isinstance(s, pd.Series)
        assert len(s) == context_length


# ==============================================================================
# Tests TimestralAdapter.predict
# ==============================================================================
@pytest.mark.parametrize(
    "bad_quantile",
    [0.33, 0.02, 0.12, 0.98, 1.1, -0.1],
    ids=lambda x: f"q={x}"
)
def test_TimestralAdapter_predict_ValueError_for_unsupported_quantile(bad_quantile):
    """
    Test predict raises ValueError for quantile levels not in
    SUPPORTED_QUANTILES.
    """
    adapter = make_adapter()
    err_msg = re.escape("Timestral only supports quantile levels")
    with pytest.raises(ValueError, match=err_msg):
        fit_and_predict(adapter, steps=3, quantiles=[0.5, bad_quantile])


def test_TimestralAdapter_predict_point_forecast_single_series():
    """
    Test point forecast (quantiles=None) on a single series: shape (steps, 1)
    with the median column of the fake (0.5), the context passed as a
    `(1, 1, n)` array and no covariates.
    """
    fake_model = FakeTimestralForecaster()
    adapter = make_adapter(model=fake_model)
    raw = fit_and_predict(adapter, steps=12)

    assert list(raw.keys()) == ["sales"]
    np.testing.assert_array_equal(raw["sales"], np.full((12, 1), 0.5))
    assert fake_model.last_context.shape == (1, 1, 50)
    np.testing.assert_array_equal(fake_model.last_context[0, 0], y.to_numpy())
    assert fake_model.last_prediction_length == 12
    assert fake_model.last_past_covariates is None
    assert fake_model.last_known_covariates is None
    assert fake_model.last_categorical_past_covariates is None
    assert fake_model.last_categorical_known_covariates is None


def test_TimestralAdapter_predict_quantile_forecast_single_series():
    """
    Test that the quantile columns are returned in the requested order.
    """
    raw = fit_and_predict(make_adapter(), steps=5, quantiles=[0.9, 0.01, 0.5])

    expected = np.tile(np.array([0.9, 0.01, 0.5]), (5, 1))
    np.testing.assert_array_almost_equal(raw["sales"], expected)


def test_TimestralAdapter_predict_all_supported_quantiles():
    """
    Test that the 21 supported quantile levels are accepted.
    """
    raw = fit_and_predict(
        make_adapter(), steps=3, quantiles=TimestralAdapter.SUPPORTED_QUANTILES
    )
    assert raw["sales"].shape == (3, 21)


def test_TimestralAdapter_predict_point_and_quantile_multi_series():
    """
    Test point and quantile forecasts on multi-series input: one array per
    series, in the order of `context`, from a single backend call.
    """
    fake_model = FakeTimestralForecaster()
    adapter = make_adapter(model=fake_model)
    ctx, ctx_exog = prepare_fit_args(y_dict)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=4)
    raw_point = adapter.predict(
        steps=4, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    raw_quantile = adapter.predict(
        steps=4, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=[0.1, 0.5, 0.9]
    )

    assert list(raw_point.keys()) == ["s1", "s2"]
    assert len(fake_model.calls) == 2
    assert fake_model.calls[0]["context"].shape == (2, 1, 30)
    for name in ["s1", "s2"]:
        np.testing.assert_array_equal(raw_point[name], np.full((4, 1), 0.5))
        assert raw_quantile[name].shape == (4, 3)


def test_TimestralAdapter_predict_no_horizon_ceiling():
    """
    Test that there is no horizon ceiling: predict succeeds for a horizon far
    larger than the context.
    """
    raw = fit_and_predict(make_adapter(), steps=2000)
    assert raw["sales"].shape == (2000, 1)


def test_TimestralAdapter_predict_forwards_predict_kwargs():
    """
    Test that predict_kwargs are forwarded verbatim to forecast.
    """
    fake_model = FakeTimestralForecaster()
    adapter = make_adapter(
        model=fake_model,
        predict_kwargs={"symmetric_averaging": False, "clip_nonnegative": False},
    )
    fit_and_predict(adapter, steps=3)

    assert fake_model.last_kwargs == {
        "symmetric_averaging": False, "clip_nonnegative": False
    }


def test_TimestralAdapter_predict_series_without_exog_key_has_no_covariates():
    """
    Test that a series with no entry in `context_exog` and `exog` (empty
    dicts, as in backtesting) is forecast with no covariates.
    """
    fake_model = FakeTimestralForecaster()
    adapter = make_adapter(model=fake_model)
    context = {"sales": y}
    adapter.fit(context=context, context_exog=None)

    raw = adapter.predict(
        steps=3, context=context, context_exog={}, exog={}, quantiles=None
    )

    assert raw["sales"].shape == (3, 1)
    assert fake_model.last_past_covariates is None
    assert fake_model.last_known_covariates is None
    assert fake_model.last_categorical_past_covariates is None
    assert fake_model.last_categorical_known_covariates is None


def test_TimestralAdapter_predict_handles_tensor_output():
    """
    Test that a torch-like tensor returned by the backend is converted to a
    NumPy array.
    """
    raw = fit_and_predict(
        make_adapter(model=FakeTimestralForecaster(as_tensor=True)),
        steps=3, quantiles=[0.1, 0.9]
    )

    assert isinstance(raw["sales"], np.ndarray)
    np.testing.assert_array_almost_equal(raw["sales"], np.tile([0.1, 0.9], (3, 1)))


# ==============================================================================
# Tests TimestralAdapter.predict: covariates
# ==============================================================================
def test_TimestralAdapter_predict_covariates_builds_past_and_known():
    """
    Test that columns only in context_exog are passed as past covariates
    (shape (1, 1, context)) and columns in exog as known covariates, the
    history followed by the future values (shape (1, 1, context + steps)).
    """
    context_length = 4
    steps = 2
    idx = y.index[-context_length:]
    future_idx = pd.date_range(idx[-1] + pd.DateOffset(months=1), periods=steps, freq="ME")
    context_exog = pd.DataFrame(
        {"known": [1.0, 2.0, 3.0, 4.0], "past_only": [10.0, 20.0, 30.0, 40.0]},
        index=idx,
    )
    future_exog = pd.DataFrame({"known": [5.0, 6.0]}, index=future_idx)

    fake_model = FakeTimestralForecaster()
    adapter = make_adapter(model=fake_model, context_length=context_length)
    ctx, ctx_exog = prepare_fit_args(y, exog=context_exog, context_length=context_length)
    adapter.fit(context=ctx, context_exog=ctx_exog)
    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=steps, exog=future_exog)
    adapter.predict(
        steps=steps, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )

    np.testing.assert_array_equal(
        fake_model.last_past_covariates, np.array([[[10.0, 20.0, 30.0, 40.0]]])
    )
    np.testing.assert_array_equal(
        fake_model.last_known_covariates,
        np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    )
    assert fake_model.last_categorical_past_covariates is None
    assert fake_model.last_categorical_known_covariates is None


def test_TimestralAdapter_predict_categorical_covariates_factorized():
    """
    Test that non-numeric columns are passed as categorical covariates, with
    codes shared by the history and the future and NaN for missing labels
    (a text known covariate and a category past covariate), while boolean and
    nullable numeric columns are passed as numeric covariates with NaN for the
    missing values.
    """
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    future_idx = pd.date_range("2024-01-05", periods=2, freq="D")
    context = {"s": pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="s")}
    context_exog = {
        "s": pd.DataFrame(
            {
                "color": ["a", "b", None, "a"],
                "cat":   pd.Categorical(["x", "y", "x", None]),
                "flag":  [True, False, True, False],
                "count": pd.array([1, None, 3, 4], dtype="Int64"),
                "level": pd.array([0.5, None, 1.5, 2.0], dtype="Float64"),
                "promo": pd.array([True, None, False, True], dtype="boolean"),
            },
            index=idx,
        )
    }
    exog = {
        "s": pd.DataFrame(
            {
                "color": ["c", "b"],
                "flag":  [True, True],
                "count": pd.array([5, None], dtype="Int64"),
                "level": pd.array([2.5, 3.0], dtype="Float64"),
                "promo": pd.array([None, False], dtype="boolean"),
            },
            index=future_idx,
        )
    }

    fake_model = FakeTimestralForecaster()
    adapter = make_adapter(model=fake_model)
    adapter.fit(context=context, context_exog=context_exog)
    adapter.predict(
        steps=2, context=context, context_exog=context_exog,
        exog=exog, quantiles=None
    )

    # Known covariates in the sorted order of the signature: count, flag,
    # level, promo (numeric) and color (categorical).
    np.testing.assert_array_equal(
        fake_model.last_known_covariates,
        np.array([[
            [1.0, np.nan, 3.0, 4.0, 5.0, np.nan],
            [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            [0.5, np.nan, 1.5, 2.0, 2.5, 3.0],
            [1.0, np.nan, 0.0, 1.0, np.nan, 0.0],
        ]])
    )
    np.testing.assert_array_equal(
        fake_model.last_categorical_known_covariates,
        np.array([[[0.0, 1.0, np.nan, 0.0, 2.0, 1.0]]])
    )
    np.testing.assert_array_equal(
        fake_model.last_categorical_past_covariates,
        np.array([[[0.0, 1.0, 0.0, np.nan]]])
    )
    assert fake_model.last_past_covariates is None


def test_TimestralAdapter_build_covariates_per_series():
    """
    Test that _build_covariates only returns the keys that have at least one
    column, and an empty dict when the series has no covariates.
    """
    ctx_idx = pd.date_range("2023-01-31", periods=3, freq="ME")
    future_idx = pd.date_range("2023-04-30", periods=2, freq="ME")
    context_exog = pd.DataFrame(
        {"known": [1.0, 2.0, 3.0], "past": [10.0, 20.0, 30.0]}, index=ctx_idx
    )
    exog_future = pd.DataFrame({"known": [4.0, 5.0]}, index=future_idx)

    covariates = TimestralAdapter._build_covariates(
        context_exog=context_exog, exog=exog_future,
        past_only_cols=("past",), fut_cols=("known",), categorical_cols=set(),
    )
    assert list(covariates) == ["past_covariates", "known_covariates"]
    np.testing.assert_array_equal(
        covariates["past_covariates"], np.array([[10.0, 20.0, 30.0]])
    )
    np.testing.assert_array_equal(
        covariates["known_covariates"], np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    )

    covariates = TimestralAdapter._build_covariates(
        context_exog=context_exog, exog=None,
        past_only_cols=(), fut_cols=(), categorical_cols=set(),
    )
    assert covariates == {}


def test_TimestralAdapter_predict_inf_in_numeric_covariate_becomes_nan():
    """
    Test that infinite values in a numeric covariate reach the backend as
    NaN, since Timestral only masks them in the target.
    """
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    future_idx = pd.date_range("2024-01-04", periods=1, freq="D")
    context = {"s": pd.Series([1.0, 2.0, 3.0], index=idx, name="s")}
    context_exog = {"s": pd.DataFrame({"x": [1.0, np.inf, 3.0]}, index=idx)}
    exog = {"s": pd.DataFrame({"x": [-np.inf]}, index=future_idx)}

    fake_model = FakeTimestralForecaster()
    adapter = make_adapter(model=fake_model)
    adapter.fit(context=context, context_exog=context_exog)
    adapter.predict(
        steps=1, context=context, context_exog=context_exog,
        exog=exog, quantiles=None
    )

    np.testing.assert_array_equal(
        fake_model.last_known_covariates, np.array([[[1.0, np.nan, 3.0, np.nan]]])
    )


def test_TimestralAdapter_predict_stacks_equal_length_series_in_one_call():
    """
    Test that series of the same length (as left-padded by FoundationModel)
    are forecast in a single call, with the covariate rows of every series in
    the sorted column order of the signature even when the series list their
    columns in a different order.
    """
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    future_idx = pd.date_range("2024-01-04", periods=2, freq="D")
    context = {
        "a": pd.Series([1.0, 2.0, 3.0], index=idx, name="a"),
        "b": pd.Series([4.0, 5.0, 6.0], index=idx, name="b"),
    }
    context_exog = {
        "a": pd.DataFrame({"p": [1.0, 2.0, 3.0], "k": [10.0, 20.0, 30.0]}, index=idx),
        "b": pd.DataFrame({"k": [40.0, 50.0, 60.0], "p": [4.0, 5.0, 6.0]}, index=idx),
    }
    exog = {
        "a": pd.DataFrame({"k": [70.0, 80.0]}, index=future_idx),
        "b": pd.DataFrame({"k": [90.0, 100.0]}, index=future_idx),
    }

    fake_model = FakeTimestralForecaster()
    adapter = make_adapter(model=fake_model)
    adapter.fit(context=context, context_exog=context_exog)
    raw = adapter.predict(
        steps=2, context=context, context_exog=context_exog,
        exog=exog, quantiles=None
    )

    assert list(raw) == ["a", "b"]
    assert len(fake_model.calls) == 1
    np.testing.assert_array_equal(
        fake_model.last_context, np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    )
    np.testing.assert_array_equal(
        fake_model.last_past_covariates, np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    )
    np.testing.assert_array_equal(
        fake_model.last_known_covariates,
        np.array([
            [[10.0, 20.0, 30.0, 70.0, 80.0]],
            [[40.0, 50.0, 60.0, 90.0, 100.0]],
        ])
    )


def test_FoundationModel_Timestral_predict_left_pads_and_groups_heterogeneous_exog():
    """
    Test through FoundationModel that series with different lengths and exog
    columns are forecast with one call per exog signature, that the shorter
    series of a call are left-padded with NaN (context and covariates), and
    that a series forecast alone with `levels` receives the same arrays as in
    the batch without the padding (same `steps`: with known covariates,
    Timestral changes the first steps when the horizon changes).
    """
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    future_idx = pd.date_range("2024-01-07", periods=2, freq="D")
    series = {
        "long":  pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx, name="long"),
        "short": pd.Series([7.0, 8.0, 9.0], index=idx[3:], name="short"),
        "none":  pd.Series([1.0, 1.0, 1.0, 1.0], index=idx[2:], name="none"),
    }
    exog_fit = {
        "long":  pd.DataFrame({"k": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=idx),
        "short": pd.DataFrame({"k": [7.0, 8.0, 9.0]}, index=idx[3:]),
    }
    exog_future = {
        "long":  pd.DataFrame({"k": [10.0, 11.0]}, index=future_idx),
        "short": pd.DataFrame({"k": [12.0, 13.0]}, index=future_idx),
    }

    fake_model = FakeTimestralForecaster()
    model = FoundationModel(model_id="mistralai/Timestral-28M", model=fake_model)
    model.fit(series=series, exog=exog_fit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = model.predict(steps=2, exog=exog_future)

    assert list(predictions["level"].unique()) == ["long", "short", "none"]
    assert len(fake_model.calls) == 2
    call_exog, call_none = fake_model.calls
    np.testing.assert_array_equal(
        call_exog["context"],
        np.array([
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            [[np.nan, np.nan, np.nan, 7.0, 8.0, 9.0]],
        ])
    )
    np.testing.assert_array_equal(
        call_exog["known_covariates"],
        np.array([
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0]],
            [[np.nan, np.nan, np.nan, 7.0, 8.0, 9.0, 12.0, 13.0]],
        ])
    )
    np.testing.assert_array_equal(
        call_none["context"], np.array([[[1.0, 1.0, 1.0, 1.0]]])
    )
    assert call_none["known_covariates"] is None

    model.predict(steps=2, levels=["short"], exog=exog_future)
    call_alone = fake_model.calls[-1]
    np.testing.assert_array_equal(call_alone["context"], np.array([[[7.0, 8.0, 9.0]]]))
    np.testing.assert_array_equal(
        call_alone["known_covariates"],
        np.array([[[7.0, 8.0, 9.0, 12.0, 13.0]]])
    )


# ==============================================================================
# Tests TimestralAdapter.predict: quantile index mapping
# ==============================================================================
def test_TimestralAdapter_predict_quantile_index_mapping_uses_model_quantile_grid():
    """
    Test that quantile columns are selected by matching against the quantile
    levels returned by the backend: with a reversed grid, 0.01 and 0.99 are
    still selected by value.
    """
    fake_model = FakeTimestralForecaster(quantile_levels=TIMESTRAL_QUANTILES[::-1])
    raw = fit_and_predict(make_adapter(model=fake_model), steps=3, quantiles=[0.01, 0.99])

    np.testing.assert_array_almost_equal(raw["sales"], np.tile([0.01, 0.99], (3, 1)))


def test_TimestralAdapter_predict_ValueError_when_quantile_missing_from_model_grid():
    """
    Test that predict raises ValueError when a requested quantile (valid
    against SUPPORTED_QUANTILES) is not in the quantile levels returned by
    the backend.
    """
    grid = [q for q in TIMESTRAL_QUANTILES if q != 0.5]
    adapter = make_adapter(model=FakeTimestralForecaster(quantile_levels=grid))
    err_msg = re.escape("Quantile 0.5 not found in the model's quantile grid")
    with pytest.raises(ValueError, match=err_msg):
        fit_and_predict(adapter, steps=3, quantiles=[0.5])


# ==============================================================================
# Tests TimestralAdapter.predict: issue mistralai/timestral#7
# ==============================================================================
@pytest.mark.parametrize(
    "history, future, warns",
    [
        ([2.5] * 6, [2.56, 2.56], True),
        ([np.nan, 2.5, 2.5, np.nan, 2.5, 2.5], [2.5, 2.6], True),
        ([2.5] * 6, [2.5, 2.5], False),
        ([2.5, 2.6, 2.5, 2.6, 2.5, 2.6], [2.7, 2.7], False),
        ([False] * 6, [True, True], True),
        (["a"] * 6, ["b", "b"], False),
    ],
    ids=["constant_then_change", "constant_with_nan", "constant_future",
         "varying_history", "bool_indicator", "categorical"],
)
def test_TimestralAdapter_predict_FeatureOutOfRangeWarning_when_known_covariate_constant_in_context(
    history, future, warns
):
    """
    Test that predict warns, naming the (series, column) pairs, when a
    numeric or boolean known covariate is constant over the context and
    changes over the horizon, and does not warn when the future keeps the same
    value, when the history already varies or when the column is categorical.
    """
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    future_idx = pd.date_range("2024-01-07", periods=2, freq="D")
    context = {"s": pd.Series(np.arange(6, dtype=float), index=idx, name="s")}
    context_exog = {"s": pd.DataFrame({"price": history}, index=idx)}
    exog = {"s": pd.DataFrame({"price": future}, index=future_idx)}

    adapter = make_adapter()
    adapter.fit(context=context, context_exog=context_exog)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.predict(
            steps=2, context=context, context_exog=context_exog,
            exog=exog, quantiles=None
        )
    caught = [w for w in caught if issubclass(w.category, FeatureOutOfRangeWarning)]

    if warns:
        assert len(caught) == 1
        assert str(caught[0].message).startswith(
            "Known covariates that are constant over the context and change "
            "over the forecast horizon, as (series, column): [('s', 'price')]. "
            "Timestral can return very large quantiles in this case "
            "(mistralai/timestral#7). Setting their historical values to NaN "
            "avoids it."
        )
    else:
        assert caught == []


def test_ForecasterFoundation_Timestral_FeatureOutOfRangeWarning_points_to_user_call():
    """
    Test that, through ForecasterFoundation.predict, the warning of issue
    mistralai/timestral#7 points to the line of the user's call.
    """
    exog = pd.DataFrame(
        {"price": [2.5] * 50 + [2.56] * 3},
        index=pd.date_range("2020-01-01", periods=53, freq="ME"),
    )
    forecaster = ForecasterFoundation(
        estimator=FoundationModel(model_id="mistralai/Timestral-28M")
    )
    forecaster.estimator.adapter._model = FakeTimestralForecaster()
    forecaster.fit(series=y, exog=exog.iloc[:50])

    with pytest.warns(FeatureOutOfRangeWarning) as record:
        forecaster.predict(steps=3, exog=exog.iloc[50:])

    assert record[0].filename == __file__


# ==============================================================================
# Tests TimestralAdapter._load_model
# ==============================================================================
class _RecordingForecaster(FakeTimestralForecaster):
    """
    FakeTimestralForecaster that records the arguments of from_pretrained.
    """
    last_args = None

    @classmethod
    def from_pretrained(cls, path_or_repo_id, device="cpu", **kwargs):
        cls.last_args = (path_or_repo_id, device, kwargs)
        return cls()


def _mock_timestral_module():
    module = types.ModuleType("timestral")
    module.TimestralForecaster = _RecordingForecaster
    return module


def test_TimestralAdapter_load_model_LicenseWarning_and_forwards_device(monkeypatch):
    """
    Test that _load_model issues a LicenseWarning, forwards the model_id and
    the resolved device to TimestralForecaster.from_pretrained, and is a
    no-op once the model is loaded. The `timestral` module is mocked so no
    network call happens.
    """
    monkeypatch.setitem(sys.modules, "timestral", _mock_timestral_module())
    adapter = TimestralAdapter(model_id="mistralai/Timestral-5M", device="cpu")

    warn_msg = re.escape(
        "The weights for 'mistralai/Timestral-5M' are released under Timestral "
        "Modified MIT License (no rights for companies with a monthly revenue "
        "above USD 20 million). Review the license terms before commercial or "
        "production use. See "
        "https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE."
    )
    with pytest.warns(LicenseWarning, match=warn_msg):
        adapter._load_model()
    assert isinstance(adapter._model, _RecordingForecaster)
    assert _RecordingForecaster.last_args == ("mistralai/Timestral-5M", "cpu", {})

    loaded = adapter._model
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter._load_model()
    assert adapter._model is loaded
    assert caught == []


def test_TimestralAdapter_load_model_ImportError_when_timestral_not_installed_no_LicenseWarning(
    monkeypatch
):
    """
    Test that _load_model raises an ImportError with the install command
    (the PEP 508 requirement of `backend_package`) when `timestral` is not
    installed, and that no LicenseWarning is issued since the weights are
    never loaded.
    """
    monkeypatch.setitem(sys.modules, "timestral", None)
    adapter = TimestralAdapter(model_id="mistralai/Timestral-28M")
    err_msg = re.escape(
        "timestral is required for TimestralAdapter. Install it with "
        "`pip install \"timestral @ git+https://github.com/mistralai/timestral.git\"`."
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ImportError, match=err_msg):
            adapter._load_model()
    assert not any(issubclass(w.category, LicenseWarning) for w in caught)


@pytest.mark.parametrize("device", ["auto", "mps", "mps:0"])
def test_TimestralAdapter_load_model_mps_falls_back_to_cpu_with_warning(
    monkeypatch, device
):
    """
    Test that MPS, selected by "auto" or requested explicitly, falls back to
    CPU with a UserWarning, since Timestral does not run on MPS
    (mistralai/timestral#6).
    """
    monkeypatch.setitem(sys.modules, "timestral", _mock_timestral_module())
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    adapter = TimestralAdapter(model_id="mistralai/Timestral-28M", device=device)

    warn_msg = re.escape(
        "MPS device is not supported by Timestral because the timestral library "
        "uses float64 operations internally. Falling back to CPU."
    )
    with patch.dict("sys.modules", {"torch": mock_torch}):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LicenseWarning)
            with pytest.warns(UserWarning, match=warn_msg):
                adapter._load_model()

    assert _RecordingForecaster.last_args[1] == "cpu"


def test_TimestralAdapter_load_model_noop_when_already_set():
    """
    Test that _load_model keeps a model passed to the constructor.
    """
    fake_model = FakeTimestralForecaster()
    adapter = make_adapter(model=fake_model)
    adapter._load_model()
    assert adapter._model is fake_model


# ==============================================================================
# Tests ForecasterFoundation with Timestral
# ==============================================================================
@pytest.mark.parametrize(
    "interval, expected",
    [
        ([0.1, 0.9], (0.1, 0.9)),
        ([0.05, 0.95], (0.05, 0.95)),
        ([0.01, 0.99], (0.01, 0.99)),
        (0.9, (0.05, 0.95)),
        (0.98, (0.01, 0.99)),
    ],
    ids=lambda x: str(x),
)
def test_ForecasterFoundation_Timestral_predict_interval_supported_levels(
    interval, expected
):
    """
    Test predict_interval with levels of the Timestral grid, also when they
    come from a float coverage (0.9 gives 0.04999999999999999 and 0.98 gives
    0.010000000000000009), which must fall within the tolerances of
    `_validate_supported_quantiles` and `_match_quantile_indices`.
    """
    forecaster = ForecasterFoundation(
        estimator=FoundationModel(model_id="mistralai/Timestral-28M")
    )
    forecaster.estimator.adapter._model = FakeTimestralForecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_interval(steps=3, interval=interval)

    assert list(result.columns) == ["level", "pred", "lower_bound", "upper_bound"]
    np.testing.assert_array_almost_equal(result["pred"].to_numpy(), [0.5] * 3)
    np.testing.assert_array_almost_equal(result["lower_bound"].to_numpy(), [expected[0]] * 3)
    np.testing.assert_array_almost_equal(result["upper_bound"].to_numpy(), [expected[1]] * 3)
