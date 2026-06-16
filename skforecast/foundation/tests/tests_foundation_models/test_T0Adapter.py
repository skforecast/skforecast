# Unit test T0Adapter
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import T0Adapter
from .fixtures_adapters import (
    y, exog,
    FakeT0Forecaster,
    prepare_fit_args, prepare_predict_args
)


# ==============================================================================
# Tests T0Adapter.__init__
# ==============================================================================
def test_T0Adapter_init_default_params():
    """
    Test that default parameter values are set correctly and class-level
    attributes are properly initialised.
    """
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    assert adapter.model_id == "theforecastingcompany/t0-alpha"
    assert adapter.context_length == 8192
    assert adapter.device_map == "auto"
    assert adapter.torch_dtype is None
    assert adapter._model is None
    assert adapter.context_ is None
    assert adapter.context_exog_ is None
    assert adapter.is_fitted is False
    assert T0Adapter.allow_exog is True


@pytest.mark.parametrize(
    "context_length",
    [0, -1, None, 1.5, "not_int"],
    ids=lambda cl: f"context_length={cl}"
)
def test_T0Adapter_init_ValueError_when_context_length_invalid(context_length):
    """
    Test that __init__ raises ValueError for non-positive-integer
    context_length values.
    """
    with pytest.raises(ValueError, match=re.escape("`context_length` must be a positive integer")):
        T0Adapter(model_id="theforecastingcompany/t0-alpha", context_length=context_length)


# ==============================================================================
# Tests T0Adapter.get_params / set_params
# ==============================================================================
def test_T0Adapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns all expected keys with the values set at
    construction.
    """
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", context_length=512)
    assert adapter.get_params() == {
        'model_id':       "theforecastingcompany/t0-alpha",
        'context_length': 512,
        'device_map':     "auto",
        'torch_dtype':    None,
    }


def test_T0Adapter_set_params_resets_model_on_loader_keys():
    """
    Test that set_params clears the loaded model when a key baked into the
    model (model_id, device_map, torch_dtype) changes.
    """
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=FakeT0Forecaster())
    adapter.set_params(context_length=256)
    assert adapter._model is not None
    adapter.set_params(device_map="cpu")
    assert adapter._model is None
    assert adapter.context_length == 256
    assert adapter.device_map == "cpu"


def test_T0Adapter_set_params_ValueError_on_invalid_key():
    """
    Test that set_params raises ValueError for unknown parameters.
    """
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    with pytest.raises(ValueError, match=re.escape("Invalid parameter(s) for T0Adapter")):
        adapter.set_params(not_a_param=1)


# ==============================================================================
# Tests T0Adapter.fit
# ==============================================================================
def test_T0Adapter_fit_stores_context():
    """
    Test that fit stores the context and exog dicts and flips is_fitted.
    """
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha")
    context, context_exog = prepare_fit_args(y, exog)
    adapter.fit(context, context_exog)
    assert adapter.is_fitted is True
    assert "sales" in adapter.context_
    assert "sales" in adapter.context_exog_


# ==============================================================================
# Tests T0Adapter.predict
# ==============================================================================
def test_T0Adapter_predict_returns_steps_by_n_quantiles():
    """
    Test that predict returns one (steps, n_quantiles) array per series with
    columns ordered to match the requested quantiles.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context, context_exog)

    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps=5)
    preds = adapter.predict(
        steps=5, context=ctx, context_exog=ctx_exog, exog=future_exog,
        quantiles=[0.9, 0.1, 0.5],
    )

    assert preds["sales"].shape == (5, 3)
    # FakeT0Forecast sets each column equal to its (sorted) quantile level;
    # the adapter must reorder columns back to the requested [0.9, 0.1, 0.5].
    np.testing.assert_allclose(preds["sales"][0], [0.9, 0.1, 0.5])
    # T0 was queried with sorted, unique levels.
    assert fake.last_quantiles == [0.1, 0.5, 0.9]


def test_T0Adapter_predict_none_quantiles_returns_median():
    """
    Test that quantiles=None produces a single median (0.5) column.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context, context_exog)

    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps=4)
    preds = adapter.predict(
        steps=4, context=ctx, context_exog=ctx_exog, exog=future_exog,
        quantiles=None,
    )

    assert preds["sales"].shape == (4, 1)
    assert fake.last_quantiles == [0.5]


def test_T0Adapter_predict_builds_future_covariates_from_past_and_future():
    """
    Test that exog past values (context_exog) and future values (exog) are
    concatenated into T0's [1, n_covariates, context_length + steps] stream.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    context, context_exog = prepare_fit_args(y, exog)
    adapter.fit(context, context_exog)

    steps = 6
    future_exog_values = exog.iloc[:steps].copy()
    ctx, ctx_exog, future_exog = prepare_predict_args(
        adapter, steps=steps, context=y, context_exog=exog, exog=future_exog_values
    )
    adapter.predict(
        steps=steps, context=ctx, context_exog=ctx_exog, exog=future_exog,
        quantiles=[0.5],
    )

    fc = fake.last_future_covariates
    context_length = len(y)
    assert fc.shape == (1, 2, context_length + steps)
    # Context portion equals the historical exog; horizon portion the future.
    np.testing.assert_allclose(fc[0, 0, :context_length], exog["feat_a"].to_numpy())
    np.testing.assert_allclose(fc[0, 0, context_length:], future_exog_values["feat_a"].to_numpy())


def test_T0Adapter_predict_no_exog_passes_none_covariates():
    """
    Test that a series without future exog is forecast with no covariates.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context, context_exog)

    ctx, ctx_exog, future_exog = prepare_predict_args(adapter, steps=3)
    adapter.predict(
        steps=3, context=ctx, context_exog=ctx_exog, exog=future_exog,
        quantiles=[0.5],
    )
    assert fake.last_future_covariates is None


def test_T0Adapter_predict_ValueError_on_non_numeric_covariate():
    """
    Test that a non-numeric future covariate raises a clear ValueError.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    context, context_exog = prepare_fit_args(y)
    adapter.fit(context, context_exog)

    steps = 3
    bad_exog = pd.DataFrame({"feat_str": ["a", "b", "c"]})
    ctx, ctx_exog, future_exog = prepare_predict_args(
        adapter, steps=steps, context=y, exog=bad_exog
    )
    with pytest.raises(ValueError, match=re.escape("T0Adapter supports only numeric covariates")):
        adapter.predict(
            steps=steps, context=ctx, context_exog=ctx_exog, exog=future_exog,
            quantiles=[0.5],
        )


# ==============================================================================
# Tests T0Adapter._load_model
# ==============================================================================
def test_T0Adapter_load_model_noop_when_already_set():
    """
    Test that _load_model is a no-op when a model is already present.
    """
    fake = FakeT0Forecaster()
    adapter = T0Adapter(model_id="theforecastingcompany/t0-alpha", model=fake)
    adapter._load_model()
    assert adapter._model is fake
