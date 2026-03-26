# Unit test FoundationalModels
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundational._adapters import (
    Chronos2Adapter,
    TimesFM25Adapter,
    _resolve_adapter,
    _ADAPTER_REGISTRY,
)
from skforecast.foundational._foundational_model import FoundationalModel

from ..tests_forecaster_foundational.fixtures_forecaster_foundational import (
    FakePipeline,
)


# Fixtures
# ==============================================================================
y = pd.Series(
    data=np.arange(50, dtype=float),
    index=pd.date_range("2020-01-01", periods=50, freq="ME"),
    name="sales",
)

exog = pd.DataFrame(
    {"feat_a": np.arange(50, dtype=float)},
    index=y.index,
)

data = np.array([
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
], dtype=np.float64)
data = pd.Series(
    data,
    index=pd.date_range(start="1949-01", periods=len(data), freq="MS"),
    name="y",
)


# Tests FoundationalModels.__init__
# ==============================================================================
def test_FoundationalModels_init_creates_Chronos2Adapter():
    """
    Test that FoundationalModels creates an internal Chronos2Adapter on init.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    assert isinstance(m.adapter, Chronos2Adapter)


def test_FoundationalModels_init_model_id_stored_in_adapter():
    """
    Test that the model_id string is forwarded to the adapter.
    """
    model_id = "autogluon/chronos-2-small"
    m = FoundationalModel(model_id)
    assert m.adapter.model_id == model_id


def test_FoundationalModels_init_kwargs_forwarded_to_adapter():
    """
    Test that keyword arguments (context_length, device_map) are forwarded to the adapter.
    """
    m = FoundationalModel(
        "autogluon/chronos-2-small", context_length=128, device_map="cpu"
    )
    assert m.adapter.context_length == 128
    assert m.adapter.device_map == "cpu"


# Tests FoundationalModels.is_fitted
# ==============================================================================
def test_FoundationalModels_is_fitted_is_false_before_fit():
    """
    Test that is_fitted is False before calling fit.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    assert m.is_fitted is False


def test_FoundationalModels_is_fitted_is_true_after_fit():
    """
    Test that is_fitted is True after fit has been called.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    m.fit(series=y)
    assert m.is_fitted is True


def test_FoundationalModels_init_context_length_attribute_default():
    """
    context_length attribute mirrors adapter default (2048).
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    assert m.context_length == 2048
    assert m.context_length == m.adapter.context_length


def test_FoundationalModels_init_context_length_attribute_when_set():
    """
    context_length attribute reflects the value passed at construction.
    """
    m = FoundationalModel("autogluon/chronos-2-small", context_length=128)
    assert m.context_length == 128
    assert m.context_length == m.adapter.context_length


def test_FoundationalModels_set_params_updates_context_length_attribute():
    """
    set_params keeps context_length attribute in sync with the adapter.
    """
    m = FoundationalModel("autogluon/chronos-2-small", context_length=512)
    assert m.context_length == 512

    m.set_params(context_length=256)
    assert m.context_length == 256
    assert m.adapter.context_length == 256


# Tests FoundationalModels.fit
# ==============================================================================
def test_FoundationalModels_fit_returns_self():
    """
    Test that fit returns the FoundationalModels instance.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    result = m.fit(series=y)
    assert result is m


def test_FoundationalModels_fit_delegates_to_adapter():
    """
    Test that fit stores history in the underlying adapter.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    m.fit(series=y)
    assert m.adapter._history is not None
    assert len(m.adapter._history) == len(y)


def test_FoundationalModels_fit_raises_TypeError_when_series_is_invalid_type():
    """
    Test that fit raises TypeError when y is not a pd.Series, pd.DataFrame, or dict.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    err_msg = re.escape(
        "`series` must be a pd.Series, a wide pd.DataFrame (one column "
        "per series), or a dict[str, pd.Series]. "
        f"Got {type([1, 2, 3])}."
    )
    with pytest.raises(TypeError, match=err_msg):
        m.fit(series=[1, 2, 3])


def test_FoundationalModels_fit_raises_ValueError_when_series_has_nan():
    """
    Test that fit raises ValueError when y contains NaN values (via check_y).
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    y_nan = y.copy()
    y_nan.iloc[3] = np.nan
    err_msg = re.escape("`series` has missing values.")
    with pytest.raises(ValueError, match=err_msg):
        m.fit(series=y_nan)


# Tests FoundationalModels.predict
# ==============================================================================
def test_FoundationalModels_predict_returns_series_for_point_forecast():
    """
    Test that predict returns a pandas Series when quantiles=None.
    """
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)
    result = m.predict(steps=5)
    assert isinstance(result, pd.Series)


def test_FoundationalModels_predict_series_has_correct_length():
    """
    Test that the point forecast Series has exactly `steps` observations.
    """
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)
    result = m.predict(steps=12)
    assert len(result) == 12


def test_FoundationalModels_predict_returns_dataframe_for_quantiles():
    """
    Test that predict returns a pandas DataFrame when quantiles are requested.
    """
    quantiles = [0.1, 0.5, 0.9]
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)
    result = m.predict(steps=5, quantiles=quantiles)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["q_0.1", "q_0.5", "q_0.9"]


def test_FoundationalModels_predict_uses_last_window():
    """
    Test that predict respects last_window: the forecast index must follow
    from last_window's index.
    """
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y)

    last_window = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2025-01-01", periods=10, freq="ME"),
        name="sales",
    )
    result = m.predict(steps=3, last_window=last_window)
    expected_start = last_window.index[-1] + last_window.index.freq
    expected_index = pd.date_range(start=expected_start, periods=3, freq=last_window.index.freq)
    pd.testing.assert_index_equal(result.index, expected_index)


def test_FoundationalModels_predict_passes_future_exog_to_pipeline():
    """
    Test that future exog passed to predict is forwarded as future_covariates.
    """
    pipeline = FakePipeline()
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=pipeline)
    m.fit(series=y)

    future = pd.DataFrame(
        {"feat_a": np.arange(6, dtype=float)},
        index=pd.date_range("2024-03-01", periods=6, freq="ME"),
    )
    m.predict(steps=6, exog=future)
    assert "future_covariates" in pipeline.last_inputs[0]


# Tests FoundationalModels.predict — exact predicted values
# ==============================================================================
def test_FoundationalModels_predict_point_forecast_exact_values():
    """
    Test that the point forecast contains exactly 0.5 for all steps when
    fitted on the air-passengers series. FakePipeline always returns the
    quantile level as a constant, so the median (0.5) is expected.
    The forecast index must immediately follow the last training date.
    """
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=data)
    result = m.predict(steps=12)
    expected_index = pd.date_range("1961-01-01", periods=12, freq="MS")
    expected_values = np.full(12, 0.5)
    pd.testing.assert_index_equal(result.index, expected_index)
    np.testing.assert_array_almost_equal(result.to_numpy(), expected_values)


def test_FoundationalModels_predict_quantile_forecast_exact_values():
    """
    Test that each quantile column contains the expected constant values when
    fitted on the air-passengers series. FakePipeline returns `q_level` for
    every step, so q_0.1 must be all 0.1, q_0.5 all 0.5, q_0.9 all 0.9.
    The forecast index must immediately follow the last training date.
    """
    quantiles = [0.1, 0.5, 0.9]
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=data)
    result = m.predict(steps=12, quantiles=quantiles)
    expected_index = pd.date_range("1961-01-01", periods=12, freq="MS")
    pd.testing.assert_index_equal(result.index, expected_index)
    for q in quantiles:
        np.testing.assert_array_almost_equal(
            result[f"q_{q}"].to_numpy(), np.full(12, q)
        )


# Fixtures for multi-series tests
# ==============================================================================
_idx_ms = pd.date_range("2020-01-01", periods=30, freq="ME")
_y_s1_ms = pd.Series(np.arange(30, dtype=float), index=_idx_ms, name="s1")
_y_s2_ms = pd.Series(np.arange(30, 60, dtype=float), index=_idx_ms, name="s2")
_y_wide_ms = pd.DataFrame({"s1": _y_s1_ms, "s2": _y_s2_ms})
_y_dict_ms = {"s1": _y_s1_ms.copy(), "s2": _y_s2_ms.copy()}


# Tests FoundationalModels — cross_learning parameter
# ==============================================================================
def test_FoundationalModels_init_cross_learning_forwarded_to_adapter():
    """
    Test that cross_learning=True is forwarded to the underlying adapter.
    """
    m = FoundationalModel("autogluon/chronos-2-small", cross_learning=True)
    assert m.adapter.cross_learning is True


def test_FoundationalModels_init_cross_learning_default_is_false():
    """
    Test that cross_learning defaults to False when not specified.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    assert m.adapter.cross_learning is False


# Tests FoundationalModels.fit — multi-series
# ==============================================================================
def test_FoundationalModels_fit_multiseries_wide_dataframe_sets_is_multiseries():
    """
    Test that fit on a wide DataFrame delegates correctly and sets _is_multiseries.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    m.fit(series=_y_wide_ms)
    assert m.adapter._is_multiseries is True
    assert m.is_fitted is True


def test_FoundationalModels_fit_multiseries_dict_sets_is_multiseries():
    """
    Test that fit on a dict of Series delegates correctly and sets _is_multiseries.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    m.fit(series=_y_dict_ms)
    assert m.adapter._is_multiseries is True


def test_FoundationalModels_fit_multiseries_returns_self():
    """
    Test that fit in multi-series mode returns the FoundationalModels instance.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    result = m.fit(series=_y_dict_ms)
    assert result is m


def test_FoundationalModels_fit_multiseries_stores_history_dict():
    """
    Test that fit stores a dict in the adapter's _history for multi-series input.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    m.fit(series=_y_dict_ms)
    assert isinstance(m.adapter._history, dict)
    assert list(m.adapter._history.keys()) == ["s1", "s2"]


# Tests FoundationalModels.predict — multi-series
# ==============================================================================
def test_FoundationalModels_predict_multiseries_point_returns_long_dataframe():
    """
    Test that predict returns a long pd.DataFrame with columns ["level", "pred"]
    for a multi-series point forecast.
    """
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=_y_dict_ms)
    result = m.predict(steps=6)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred"]


def test_FoundationalModels_predict_multiseries_point_correct_length():
    """
    Test that the point forecast long DataFrame has exactly steps * n_series rows.
    """
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=_y_wide_ms)
    result = m.predict(steps=10)
    assert len(result) == 10 * 2  # steps * n_series


def test_FoundationalModels_predict_multiseries_quantile_returns_long_dataframe():
    """
    Test that predict returns a long pd.DataFrame for a multi-series quantile
    forecast, with columns ["level", "q_0.1", "q_0.5", "q_0.9"].
    """
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=_y_dict_ms)
    result = m.predict(steps=5, quantiles=[0.1, 0.5, 0.9])
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "q_0.1", "q_0.5", "q_0.9"]
    assert len(result) == 5 * 2  # steps * n_series


def test_FoundationalModels_predict_multiseries_cross_learning_forwarded_to_pipeline():
    """
    Test that cross_learning is forwarded all the way to predict_quantiles.
    """
    pipeline = FakePipeline()
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=pipeline, cross_learning=True)
    m.fit(series=_y_dict_ms)
    m.predict(steps=3)
    assert pipeline.last_kwargs.get("cross_learning") is True


def test_FoundationalModels_predict_multiseries_last_window_wide_dataframe():
    """
    Test that a wide DataFrame passed as last_window is used in multi-series mode.
    The forecast index must follow from last_window's index.
    """
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=_y_dict_ms)
    new_idx = pd.date_range("2025-01-01", periods=10, freq="ME")
    lw = pd.DataFrame(
        {"s1": np.arange(10, dtype=float), "s2": np.arange(10, 20, dtype=float)},
        index=new_idx,
    )
    result = m.predict(steps=4, last_window=lw)
    expected_start = new_idx[-1] + new_idx.freq
    expected_index = pd.date_range(start=expected_start, periods=4, freq=new_idx.freq)
    pd.testing.assert_index_equal(result.index.unique(), expected_index)


def test_FoundationalModels_predict_multiseries_last_window_dict():
    """
    Test that a dict[str, pd.Series] passed as last_window is handled correctly.
    Output must be a long DataFrame with columns ["level", "pred"].
    """
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=_y_dict_ms)
    new_idx = pd.date_range("2025-06-01", periods=8, freq="ME")
    lw_dict = {
        "s1": pd.Series(np.arange(8, dtype=float), index=new_idx, name="s1"),
        "s2": pd.Series(np.arange(8, 16, dtype=float), index=new_idx, name="s2"),
    }
    result = m.predict(steps=3, last_window=lw_dict)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred"]


def test_FoundationalModels_predict_multiseries_exact_point_values():
    """
    Test that point forecast values equal 0.5 for all steps and series
    (FakePipeline returns the quantile level as the value; median = 0.5).
    """
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=_y_dict_ms)
    result = m.predict(steps=5)
    for name in ["s1", "s2"]:
        subset = result[result["level"] == name]
        np.testing.assert_array_almost_equal(subset["pred"].to_numpy(), np.full(5, 0.5))


# Tests FoundationalModel.get_params / set_params
# ==============================================================================

def test_FoundationalModels_get_params_returns_expected_keys():
    """
    Test that get_params returns a dict with the canonical set of parameter
    names and does NOT expose internal attributes like 'pipeline'.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    params = m.get_params()
    assert set(params.keys()) == {
        "model",
        "cross_learning",
        "context_length",
        "device_map",
        "torch_dtype",
        "predict_kwargs",
    }
    assert "pipeline" not in params


def test_FoundationalModels_get_params_returns_correct_values():
    """
    Test that get_params reflects the values passed at construction time.
    """
    m = FoundationalModel(
        "autogluon/chronos-2-small",
        context_length=64,
        cross_learning=True,
    )
    params = m.get_params()
    assert params["model"] == "autogluon/chronos-2-small"
    assert params["context_length"] == 64
    assert params["cross_learning"] is True


def test_FoundationalModels_set_params_invalid_key_raises_ValueError():
    """
    Test that set_params raises ValueError when given a key that is not a valid
    constructor parameter.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    err_msg = re.escape("Invalid parameter")
    with pytest.raises(ValueError, match=err_msg):
        m.set_params(invalid_param=True)


def test_FoundationalModels_init_model_id_attribute():
    """
    model_id attribute mirrors adapter.model_id at construction.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    assert m.model_id == "autogluon/chronos-2-small"
    assert m.model_id == m.adapter.model_id


def test_FoundationalModels_set_params_updates_model_id_attribute():
    """
    set_params keeps model_id attribute in sync with the adapter.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    assert m.model_id == "autogluon/chronos-2-small"
    m.set_params(model_id="autogluon/chronos-2-tiny")
    assert m.model_id == "autogluon/chronos-2-tiny"
    assert m.adapter.model_id == "autogluon/chronos-2-tiny"


# Tests _resolve_adapter / _ADAPTER_REGISTRY
# ==============================================================================
def test_resolve_adapter_returns_Chronos2Adapter_for_chronos_prefix():
    """
    Test that _resolve_adapter returns Chronos2Adapter for any model ID
    starting with 'autogluon/chronos'.
    """
    assert _resolve_adapter("autogluon/chronos-2-small") is Chronos2Adapter
    assert _resolve_adapter("autogluon/chronos-2-large") is Chronos2Adapter


def test_resolve_adapter_raises_ValueError_for_unknown_prefix():
    """
    Test that _resolve_adapter raises ValueError with a clear message when
    no registered prefix matches the model ID.
    """
    err_msg = re.escape("No adapter found for model 'unknown/my-model'.")
    with pytest.raises(ValueError, match=err_msg):
        _resolve_adapter("unknown/my-model")


def test_resolve_adapter_error_lists_registered_prefixes():
    """
    Test that the ValueError message for an unknown model ID lists the
    known prefixes so the user knows what is supported.
    """
    with pytest.raises(ValueError, match="Registered prefixes"):
        _resolve_adapter("unknown/model")


def test_FoundationalModel_init_raises_ValueError_for_unknown_model():
    """
    Test that FoundationalModel raises ValueError when the model ID does
    not match any registered adapter prefix.
    """
    with pytest.raises(ValueError, match="No adapter found"):
        FoundationalModel("unknown/unsupported-model")


def test_ADAPTER_REGISTRY_contains_chronos_key():
    """
    Test that _ADAPTER_REGISTRY has the 'autogluon/chronos' prefix mapped
    to Chronos2Adapter.
    """
    assert "autogluon/chronos" in _ADAPTER_REGISTRY
    assert _ADAPTER_REGISTRY["autogluon/chronos"] is Chronos2Adapter


def test_resolve_adapter_returns_TimesFM25Adapter_for_timesfm_prefix():
    """
    Test that _resolve_adapter returns TimesFM25Adapter for any model ID
    starting with 'google/timesfm'.
    """
    assert _resolve_adapter("google/timesfm-2.5-200m-pytorch") is TimesFM25Adapter
    assert _resolve_adapter("google/timesfm-2.5-200m-flax") is TimesFM25Adapter


def test_ADAPTER_REGISTRY_contains_timesfm_key():
    """
    Test that _ADAPTER_REGISTRY has the 'google/timesfm' prefix mapped
    to TimesFM25Adapter.
    """
    assert "google/timesfm" in _ADAPTER_REGISTRY
    assert _ADAPTER_REGISTRY["google/timesfm"] is TimesFM25Adapter


# Tests Chronos2Adapter.get_params
# ==============================================================================
def test_Chronos2Adapter_get_params_returns_expected_keys():
    """
    Test that get_params returns a dict with all constructor parameter keys
    and does NOT include private attributes like '_pipeline'.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    params = adapter.get_params()
    assert set(params.keys()) == {
        "model_id", "cross_learning", "context_length",
        "device_map", "torch_dtype", "predict_kwargs",
    }
    assert "_pipeline" not in params
    assert "pipeline" not in params


def test_Chronos2Adapter_get_params_returns_correct_values():
    """
    Test that get_params reflects the values set at construction time.
    """
    adapter = Chronos2Adapter(
        model_id="autogluon/chronos-2-small",
        context_length=512,
        cross_learning=True,
        device_map="cpu",
    )
    params = adapter.get_params()
    assert params["model_id"] == "autogluon/chronos-2-small"
    assert params["context_length"] == 512
    assert params["cross_learning"] is True
    assert params["device_map"] == "cpu"


def test_Chronos2Adapter_get_params_predict_kwargs_is_none_when_empty():
    """
    Test that get_params returns None for predict_kwargs when the adapter
    was created without any predict_kwargs (normalised to {} internally).
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    assert adapter.get_params()["predict_kwargs"] is None


# Tests Chronos2Adapter.set_params
# ==============================================================================
def test_Chronos2Adapter_set_params_updates_cross_learning():
    """
    Test that set_params correctly updates cross_learning without resetting
    the pipeline (cross_learning is not a pipeline-reset key).
    """
    adapter = Chronos2Adapter(
        model_id="autogluon/chronos-2-small", pipeline=FakePipeline()
    )
    adapter.set_params(cross_learning=True)
    assert adapter.cross_learning is True
    assert adapter._pipeline is not None  # pipeline NOT reset


def test_Chronos2Adapter_set_params_model_id_resets_pipeline():
    """
    Test that changing model_id via set_params clears the cached pipeline,
    since it will need to be reloaded for the new model.
    """
    adapter = Chronos2Adapter(
        model_id="autogluon/chronos-2-small", pipeline=FakePipeline()
    )
    assert adapter._pipeline is not None
    adapter.set_params(model_id="autogluon/chronos-2-large")
    assert adapter._pipeline is None
    assert adapter.model_id == "autogluon/chronos-2-large"


def test_Chronos2Adapter_set_params_device_map_resets_pipeline():
    """
    Test that changing device_map resets the pipeline.
    """
    adapter = Chronos2Adapter(
        model_id="autogluon/chronos-2-small", pipeline=FakePipeline()
    )
    adapter.set_params(device_map="cpu")
    assert adapter._pipeline is None
    assert adapter.device_map == "cpu"


def test_Chronos2Adapter_set_params_invalid_key_raises_ValueError():
    """
    Test that set_params raises ValueError for unrecognised keys.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    with pytest.raises(ValueError, match="Invalid parameter"):
        adapter.set_params(bad_key=42)


def test_Chronos2Adapter_set_params_predict_kwargs_none_normalises_to_empty_dict():
    """
    Test that passing predict_kwargs=None via set_params normalises
    the internal value to an empty dict (consistent with __init__ behaviour).
    """
    adapter = Chronos2Adapter(
        model_id="autogluon/chronos-2-small",
        predict_kwargs={"num_samples": 20},
    )
    adapter.set_params(predict_kwargs=None)
    assert adapter.predict_kwargs == {}


# Tests FoundationalModel.set_params — model → model_id translation
# ==============================================================================
def test_FoundationalModels_set_params_model_translates_to_model_id():
    """
    Test that set_params(model=...) correctly translates to model_id on the
    underlying adapter.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    m.set_params(model="autogluon/chronos-2-large")
    assert m.adapter.model_id == "autogluon/chronos-2-large"


def test_FoundationalModels_set_params_model_resets_pipeline():
    """
    Test that set_params(model=...) resets the adapter's cached pipeline.
    """
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    assert m.adapter._pipeline is not None
    m.set_params(model="autogluon/chronos-2-large")
    assert m.adapter._pipeline is None


def test_FoundationalModels_set_params_non_pipeline_key_no_pipeline_reset():
    """
    Test that set_params with a non-pipeline key (cross_learning) does not
    reset the cached pipeline.
    """
    m = FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.set_params(cross_learning=True)
    assert m.adapter.cross_learning is True
    assert m.adapter._pipeline is not None


def test_FoundationalModels_set_params_error_message_says_FoundationalModel():
    """
    Test that an invalid-key error from set_params names 'FoundationalModel',
    not the internal 'Chronos2Adapter'.
    """
    m = FoundationalModel("autogluon/chronos-2-small")
    with pytest.raises(ValueError) as exc_info:
        m.set_params(bad_param=1)
    assert "FoundationalModel" in str(exc_info.value)
    assert "Chronos2Adapter" not in str(exc_info.value)


