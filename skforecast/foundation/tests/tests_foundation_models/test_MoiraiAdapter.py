# Unit test MoiraiAdapter
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import MoiraiAdapter
from .fixtures_adapters import (
    y, y_wide, y_dict,
    FakeMoirai2Forecast,
    prepare_fit_args, prepare_predict_args
)


# Helpers
# ==============================================================================
def make_adapter(**kwargs) -> MoiraiAdapter:
    """
    Return a MoiraiAdapter pre-loaded with FakeMoirai2Forecast.
    """
    adapter = MoiraiAdapter(
        model_id=kwargs.pop("model_id", "Salesforce/moirai-2.0-R-small"),
        **kwargs
    )
    adapter._forecast_obj = FakeMoirai2Forecast()
    return adapter


# ==============================================================================
# Tests MoiraiAdapter.__init__
# ==============================================================================
def test_MoiraiAdapter_init_default_params():
    """
    Test that default parameter values are set correctly and class-level
    attributes are properly initialised.
    """
    adapter = MoiraiAdapter(model_id="Salesforce/moirai-2.0-R-small")
    assert adapter.model_id == "Salesforce/moirai-2.0-R-small"
    assert adapter.context_length == 2048
    assert adapter.device == "auto"
    assert adapter._module is None
    assert adapter._forecast_obj is None
    assert adapter.context_ is None
    assert adapter.is_fitted is False
    assert MoiraiAdapter.allow_exog is False
    assert MoiraiAdapter.SUPPORTED_QUANTILES == [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ]


@pytest.mark.parametrize(
    "context_length",
    [0, -1, None],
    ids=lambda cl: f"context_length={cl}"
)
def test_MoiraiAdapter_init_ValueError_when_context_length_invalid(context_length):
    """
    Test that __init__ raises ValueError for non-positive-integer
    context_length values.
    """
    with pytest.raises(ValueError, match=re.escape("`context_length` must be a positive integer")):
        MoiraiAdapter(
            model_id="Salesforce/moirai-2.0-R-small",
            context_length=context_length
        )


def test_MoiraiAdapter_init_custom_module_stored():
    """
    Test that a pre-loaded module is stored in _module.
    """
    fake_module = object()
    adapter = MoiraiAdapter(
        model_id="Salesforce/moirai-2.0-R-small", module=fake_module
    )
    assert adapter._module is fake_module


# ==============================================================================
# Tests MoiraiAdapter.get_params / set_params
# ==============================================================================
def test_MoiraiAdapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns all expected keys with correct values.
    """
    adapter = MoiraiAdapter(
        model_id="Salesforce/moirai-2.0-R-base", context_length=1024
    )
    params = adapter.get_params()
    assert set(params.keys()) == {"model_id", "context_length", "device"}
    assert params["model_id"] == "Salesforce/moirai-2.0-R-base"
    assert params["context_length"] == 1024
    assert params["device"] == "auto"


@pytest.mark.parametrize(
    "params, match",
    [
        ({"context_length": 0}, "`context_length` must be a positive integer"),
        ({"unknown_param": 42}, "Invalid parameter"),
    ],
    ids=["context_length=0", "unknown_param"]
)
def test_MoiraiAdapter_set_params_ValueError_when_invalid(params, match):
    """
    Test that set_params raises ValueError for invalid values or unknown
    parameter names.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match=re.escape(match)):
        adapter.set_params(**params)


@pytest.mark.parametrize(
    "param, value",
    [
        ("model_id", "Salesforce/moirai-2.0-R-large"),
        ("context_length", 512),
        ("device", "cpu"),
    ],
    ids=lambda x: str(x)
)
def test_MoiraiAdapter_set_params_updates_and_resets_module(param, value):
    """
    Test that set_params updates the given parameter, resets _module and
    _forecast_obj, and returns self.
    """
    adapter = make_adapter()
    adapter._module = object()
    result = adapter.set_params(**{param: value})
    assert result is adapter
    assert adapter._module is None
    assert adapter._forecast_obj is None


# ==============================================================================
# Tests MoiraiAdapter.fit
# ==============================================================================
def test_MoiraiAdapter_fit_error_handling():
    """
    Test fit raises TypeError for unsupported series types and ValueError
    for empty dict.
    """
    adapter = make_adapter()
    with pytest.raises(TypeError):
        prepare_fit_args(np.arange(50))
    with pytest.raises(ValueError):
        prepare_fit_args({})


@pytest.mark.parametrize(
    "context_length, expected_len",
    [(10, 10), (20, 20), (50, 50), (100, 50)],
    ids=lambda x: f"{x}"
)
def test_MoiraiAdapter_fit_output_single_series(context_length, expected_len):
    """
    Test fit on a single series: returns self, sets is_fitted=True,
    stores history trimmed to context_length,
    and does not modify the input series.
    """
    adapter = make_adapter(context_length=context_length)
    y_copy = y.copy()
    ctx, ctx_exog = prepare_fit_args(y, context_length=context_length)
    result = adapter.fit(
        context=ctx, context_exog=ctx_exog
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
def test_MoiraiAdapter_fit_output_multi_series(series_input):
    """
    Test fit on multi-series input: sets is_fitted=True,
    stores a dict of Series keyed by series names,
    each trimmed to context_length.
    """
    context_length = 10
    adapter = make_adapter(context_length=context_length)
    ctx, ctx_exog = prepare_fit_args(series_input, context_length=context_length)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    assert adapter.is_fitted is True
    assert set(adapter.context_.keys()) == {"s1", "s2"}
    for name, s in adapter.context_.items():
        assert isinstance(s, pd.Series)
        assert len(s) == context_length


def test_MoiraiAdapter_fit_exog_ignored_silently():
    """
    Test that passing exog to fit completes successfully (exog handling is
    done upstream by FoundationModel).
    """
    exog_df = pd.DataFrame({"feat": np.arange(50, dtype=float)}, index=y.index)
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y, exog=exog_df)
    adapter.fit(context=ctx, context_exog=ctx_exog)
    assert adapter.is_fitted is True


# ==============================================================================
# Tests MoiraiAdapter.predict — error handling
# ==============================================================================
@pytest.mark.parametrize(
    "bad_quantile",
    [0.05, 0.15, 0.25, 0.95, 1.1, -0.1],
    ids=lambda x: f"q={x}"
)
def test_MoiraiAdapter_predict_ValueError_for_unsupported_quantile(bad_quantile):
    """
    Test predict raises ValueError for quantile levels not in
    SUPPORTED_QUANTILES.
    """
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    with pytest.raises(ValueError, match=re.escape("Moirai only supports quantile levels")):
        adapter.predict(
            steps=3, context=ctx_p, context_exog=ctx_exog_p,
            exog=exog_p, quantiles=[0.5, bad_quantile],
            
        )


# ==============================================================================
# Tests MoiraiAdapter.predict — single series
# ==============================================================================
def test_MoiraiAdapter_predict_point_forecast_single_series():
    """
    Test point forecast (quantiles=None) on a single series: returns dict
    with one key, shape (steps, 1), values = 0.5 (FakeMoirai2Forecast
    q_idx=4 → 0.5).
    """
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=6)
    raw = adapter.predict(
        steps=6, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )

    assert list(raw.keys()) == ["sales"]
    arr = raw["sales"]
    assert arr.shape == (6, 1)
    np.testing.assert_array_almost_equal(arr[:, 0], np.full(6, 0.5))


def test_MoiraiAdapter_predict_quantile_forecast_single_series():
    """
    Test quantile forecast on a single series: returns dict with correct
    shape and values matching FakeMoirai2Forecast output.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = make_adapter()
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


def test_MoiraiAdapter_predict_all_supported_quantiles():
    """
    Test that all 9 supported quantile levels are accepted without error.
    """
    adapter = make_adapter()
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    raw = adapter.predict(
        steps=3, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p,
        quantiles=MoiraiAdapter.SUPPORTED_QUANTILES,
        
    )
    assert raw["sales"].shape == (3, 9)


# ==============================================================================
# Tests MoiraiAdapter.predict — multi-series
# ==============================================================================
@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_MoiraiAdapter_predict_point_forecast_multi_series(series_input):
    """
    Test point forecast on multi-series: returns dict with one array per
    series, each of shape (steps, 1) with value 0.5.
    """
    adapter = make_adapter()
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


def test_MoiraiAdapter_predict_quantile_forecast_multi_series():
    """
    Test quantile forecast on multi-series: returns dict with one array per
    series, each of shape (steps, n_quantiles) with correct values.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = make_adapter()
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
# Tests MoiraiAdapter.predict — pipeline receives correct args
# ==============================================================================
def test_MoiraiAdapter_predict_model_receives_correct_args():
    """
    Test that the model's predict receives the correct number of inputs and
    each input has shape (T, 1). Steps forwarded via hparams_context.
    """
    fake_forecast = FakeMoirai2Forecast()
    adapter = make_adapter()
    adapter._forecast_obj = fake_forecast
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=7)
    adapter.predict(
        steps=7, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    assert fake_forecast._last_steps == 7
    assert len(fake_forecast.last_inputs) == 1
    assert fake_forecast.last_inputs[0].shape == (len(y), 1)


def test_MoiraiAdapter_predict_multiseries_batched_inputs():
    """
    Test that all series are batched into a single predict call (one array
    per series in inputs_list) with shape (T, 1).
    """
    fake_forecast = FakeMoirai2Forecast()
    adapter = make_adapter()
    adapter._forecast_obj = fake_forecast
    ctx, ctx_exog = prepare_fit_args(y_wide)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    adapter.predict(
        steps=3, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    assert len(fake_forecast.last_inputs) == 2
    for arr in fake_forecast.last_inputs:
        assert arr.ndim == 2
        assert arr.shape[1] == 1


def test_MoiraiAdapter_predict_context_length_trims_history():
    """
    Test that the history passed to the model is trimmed to context_length.
    """
    context_length = 10
    fake_forecast = FakeMoirai2Forecast()
    adapter = make_adapter(context_length=context_length)
    adapter._forecast_obj = fake_forecast
    ctx, ctx_exog = prepare_fit_args(y, context_length=context_length)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=5)
    adapter.predict(
        steps=5, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    assert fake_forecast.last_inputs[0].shape == (context_length, 1)
