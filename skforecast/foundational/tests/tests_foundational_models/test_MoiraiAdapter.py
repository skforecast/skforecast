# Unit tests MoiraiAdapter
# ==============================================================================
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.foundational._adapters import MoiraiAdapter


# Fixtures and helpers
# ==============================================================================
_idx = pd.date_range("2020-01-01", periods=50, freq="ME")
y = pd.Series(np.arange(50, dtype=float), index=_idx, name="sales")

_idx_ms = pd.date_range("2020-01-01", periods=30, freq="ME")
_ys1 = pd.Series(np.arange(30, dtype=float), index=_idx_ms, name="s1")
_ys2 = pd.Series(np.arange(30, 60, dtype=float), index=_idx_ms, name="s2")
_y_wide = pd.DataFrame({"s1": _ys1, "s2": _ys2})
_y_dict = {"s1": _ys1.copy(), "s2": _ys2.copy()}


class FakeMoirai2Forecast:
    """
    Fake Moirai2Forecast for testing without the uni2ts / torch dependency.

    `predict(past_target)` returns `np.ndarray` of shape
    `(n, 9, steps)` where `raw[i, q_idx, :]` equals `(q_idx + 1) / 10`
    i.e. the same value as the corresponding quantile level.

    Records the last call arguments for inspection.
    """

    def __init__(self):
        self.last_inputs = None
        self._last_steps = None

    class _HparamsCtx:
        """Context manager that sets prediction_length on the forecast object."""

        def __init__(self, forecast_obj, prediction_length):
            self._obj = forecast_obj
            self._pl = prediction_length

        def __enter__(self):
            self._obj._last_steps = self._pl
            return self._obj

        def __exit__(self, *args):
            pass

    def hparams_context(self, prediction_length):
        return self._HparamsCtx(self, prediction_length)

    def predict(self, past_target):
        self.last_inputs = past_target
        n = len(past_target)
        steps = self._last_steps
        # q_idx i → value (i+1)/10  (0.1, 0.2, …, 0.9)
        raw = np.zeros((n, 9, steps), dtype=float)
        for q_idx in range(9):
            raw[:, q_idx, :] = (q_idx + 1) / 10.0
        return raw


def make_adapter(**kwargs) -> MoiraiAdapter:
    """Return a MoiraiAdapter pre-loaded with FakeMoirai2Forecast."""
    adapter = MoiraiAdapter(
        model_id=kwargs.pop("model_id", "Salesforce/moirai-2.0-R-small"),
        **kwargs,
    )
    adapter._forecast_obj = FakeMoirai2Forecast()
    return adapter


# Tests MoiraiAdapter.__init__
# ==============================================================================
def test_MoiraiAdapter_init_default_params():
    """
    Test that default parameter values are set correctly.
    """
    adapter = MoiraiAdapter(model_id="Salesforce/moirai-2.0-R-small")
    assert adapter.model_id == "Salesforce/moirai-2.0-R-small"
    assert adapter.context_length == 2048
    assert adapter._module is None
    assert adapter._forecast_obj is None
    assert adapter._history is None
    assert adapter._is_fitted is False
    assert adapter._is_multiseries is False


def test_MoiraiAdapter_allow_exogenous_is_False():
    """
    allow_exogenous class attribute is False (covariates not supported).
    """
    assert MoiraiAdapter.allow_exogenous is False
    assert make_adapter().allow_exogenous is False


def test_MoiraiAdapter_init_custom_context_length():
    """
    Test that a custom context_length is stored correctly.
    """
    adapter = MoiraiAdapter(model_id="Salesforce/moirai-2.0-R-small", context_length=512)
    assert adapter.context_length == 512


def test_MoiraiAdapter_init_custom_module_stored():
    """
    Test that a pre-loaded module is stored in _module.
    """
    fake_module = object()
    adapter = MoiraiAdapter(model_id="Salesforce/moirai-2.0-R-small", module=fake_module)
    assert adapter._module is fake_module


def test_MoiraiAdapter_init_raises_ValueError_when_context_length_is_zero():
    """
    Test that __init__ raises ValueError when context_length is 0.
    """
    with pytest.raises(ValueError, match="`context_length` must be a positive integer"):
        MoiraiAdapter(model_id="Salesforce/moirai-2.0-R-small", context_length=0)


def test_MoiraiAdapter_init_raises_ValueError_when_context_length_is_negative():
    """
    Test that __init__ raises ValueError when context_length is negative.
    """
    with pytest.raises(ValueError, match="`context_length` must be a positive integer"):
        MoiraiAdapter(model_id="Salesforce/moirai-2.0-R-small", context_length=-1)


def test_MoiraiAdapter_init_raises_ValueError_when_context_length_is_None():
    """
    Test that __init__ raises ValueError when context_length is None.
    """
    with pytest.raises(ValueError, match="`context_length` must be a positive integer"):
        MoiraiAdapter(model_id="Salesforce/moirai-2.0-R-small", context_length=None)


def test_MoiraiAdapter_SUPPORTED_QUANTILES_has_nine_levels():
    """
    Test that SUPPORTED_QUANTILES contains the 9 expected levels.
    """
    expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert MoiraiAdapter.SUPPORTED_QUANTILES == expected


# Tests MoiraiAdapter.get_params / set_params
# ==============================================================================
def test_MoiraiAdapter_get_params_returns_expected_keys():
    """
    Test that get_params returns only the keys model_id and context_length.
    """
    adapter = MoiraiAdapter(model_id="Salesforce/moirai-2.0-R-small")
    assert set(adapter.get_params().keys()) == {"model_id", "context_length"}


def test_MoiraiAdapter_get_params_round_trip():
    """
    Test that get_params reflects the values set at construction.
    """
    adapter = MoiraiAdapter(
        model_id="Salesforce/moirai-2.0-R-base", context_length=1024
    )
    params = adapter.get_params()
    assert params["model_id"] == "Salesforce/moirai-2.0-R-base"
    assert params["context_length"] == 1024


def test_MoiraiAdapter_set_params_returns_self():
    """
    Test that set_params returns the adapter instance.
    """
    adapter = make_adapter()
    result = adapter.set_params(context_length=512)
    assert result is adapter


def test_MoiraiAdapter_set_params_updates_context_length():
    """
    Test that set_params updates context_length.
    """
    adapter = make_adapter()
    adapter.set_params(context_length=512)
    assert adapter.context_length == 512


def test_MoiraiAdapter_set_params_updates_model_id():
    """
    Test that set_params updates model_id.
    """
    adapter = make_adapter()
    adapter.set_params(model_id="Salesforce/moirai-2.0-R-large")
    assert adapter.model_id == "Salesforce/moirai-2.0-R-large"


def test_MoiraiAdapter_set_params_resets_module_and_forecast_obj_on_model_id_change():
    """
    Test that set_params resets _module and _forecast_obj when model_id changes.
    """
    adapter = make_adapter()
    adapter._module = object()
    adapter.set_params(model_id="Salesforce/moirai-2.0-R-large")
    assert adapter._module is None
    assert adapter._forecast_obj is None


def test_MoiraiAdapter_set_params_resets_module_and_forecast_obj_on_context_length_change():
    """
    Test that set_params resets _module and _forecast_obj when context_length changes.
    """
    adapter = make_adapter()
    adapter._module = object()
    adapter.set_params(context_length=256)
    assert adapter._module is None
    assert adapter._forecast_obj is None


def test_MoiraiAdapter_set_params_raises_ValueError_on_invalid_context_length():
    """
    Test that set_params raises ValueError when context_length is invalid.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match="`context_length` must be a positive integer"):
        adapter.set_params(context_length=0)


def test_MoiraiAdapter_set_params_raises_ValueError_on_unknown_param():
    """
    Test that set_params raises ValueError for unrecognised parameter names.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match="Invalid parameter"):
        adapter.set_params(unknown_param=42)


# Tests MoiraiAdapter.fit
# ==============================================================================
def test_MoiraiAdapter_fit_sets_is_fitted():
    """
    Test that fit sets _is_fitted to True.
    """
    adapter = make_adapter()
    assert adapter._is_fitted is False
    adapter.fit(series=y)
    assert adapter._is_fitted is True


def test_MoiraiAdapter_fit_returns_self():
    """
    Test that fit returns the adapter instance.
    """
    adapter = make_adapter()
    assert adapter.fit(series=y) is adapter


def test_MoiraiAdapter_fit_stores_single_series_history():
    """
    Test that fit stores the series in _history for single-series mode.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    assert isinstance(adapter._history, pd.Series)
    pd.testing.assert_series_equal(adapter._history, y)


def test_MoiraiAdapter_fit_trims_history_to_context_length():
    """
    Test that fit trims _history to the last context_length observations.
    """
    context_length = 20
    adapter = make_adapter(context_length=context_length)
    adapter.fit(series=y)
    assert len(adapter._history) == context_length
    pd.testing.assert_series_equal(adapter._history, y.iloc[-context_length:])


def test_MoiraiAdapter_fit_history_is_copy_not_view():
    """
    Test that _history is a copy, not a view of the input series.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    original_first = adapter._history.iloc[0]
    y.iloc[0] = 9999.0
    assert adapter._history.iloc[0] == original_first
    y.iloc[0] = 0.0  # restore


def test_MoiraiAdapter_fit_sets_is_multiseries_false_for_series():
    """
    Test that _is_multiseries is False after fitting on a pd.Series.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    assert adapter._is_multiseries is False


def test_MoiraiAdapter_fit_sets_is_multiseries_true_for_dataframe():
    """
    Test that _is_multiseries is True after fitting on a wide DataFrame.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    assert adapter._is_multiseries is True


def test_MoiraiAdapter_fit_sets_is_multiseries_true_for_dict():
    """
    Test that _is_multiseries is True after fitting on a dict of Series.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_dict)
    assert adapter._is_multiseries is True


def test_MoiraiAdapter_fit_multiseries_stores_history_dict():
    """
    Test that fit stores a dict of Series as _history in multi-series mode.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    assert isinstance(adapter._history, dict)
    assert set(adapter._history.keys()) == {"s1", "s2"}


def test_MoiraiAdapter_fit_multiseries_trims_each_series_to_context_length():
    """
    Test that fit trims each series in _history to context_length in
    multi-series mode.
    """
    context_length = 10
    adapter = make_adapter(context_length=context_length)
    adapter.fit(series=_y_wide)
    for name, s in adapter._history.items():
        assert len(s) == context_length, f"Series '{name}' not trimmed"


def test_MoiraiAdapter_fit_issues_IgnoredArgumentWarning_for_exog():
    """
    Test that passing exog to fit issues an IgnoredArgumentWarning and completes
    successfully.
    """
    exog = pd.DataFrame({"feat": np.arange(50, dtype=float)}, index=y.index)
    adapter = make_adapter()
    with pytest.warns(IgnoredArgumentWarning, match="MoiraiAdapter does not support covariates"):
        adapter.fit(series=y, exog=exog)
    assert adapter._is_fitted is True


def test_MoiraiAdapter_fit_raises_TypeError_on_unsupported_series_type():
    """
    Test that fit raises TypeError when series is neither pd.Series,
    pd.DataFrame nor dict.
    """
    adapter = make_adapter()
    with pytest.raises(TypeError):
        adapter.fit(series=np.arange(50))


def test_MoiraiAdapter_fit_raises_ValueError_when_series_dict_is_empty():
    """
    Test that fit raises ValueError when an empty dict is passed.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match="at least one series"):
        adapter.fit(series={})


# Tests MoiraiAdapter.predict — error handling
# ==============================================================================
def test_MoiraiAdapter_predict_raises_ValueError_when_not_fitted_and_no_last_window():
    """
    Test predict raises ValueError when adapter is not fitted and no
    last_window is passed.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match="Call `fit` before `predict`, or pass `last_window`"):
        adapter.predict(steps=5)


@pytest.mark.parametrize(
    "steps",
    [0, -1, -10],
    ids=lambda x: f"steps_{x}",
)
def test_MoiraiAdapter_predict_raises_ValueError_when_steps_not_positive(steps):
    """
    Test predict raises ValueError when steps is 0 or negative.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    with pytest.raises(ValueError, match="`steps` must be a positive integer"):
        adapter.predict(steps=steps)


@pytest.mark.parametrize(
    "bad_quantile",
    [0.05, 0.15, 0.25, 0.95, 1.1, -0.1],
    ids=lambda x: f"q_{x}",
)
def test_MoiraiAdapter_predict_raises_ValueError_for_unsupported_quantile(bad_quantile):
    """
    Test predict raises ValueError for quantile levels not in SUPPORTED_QUANTILES.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    with pytest.raises(ValueError, match="Moirai-2 only supports quantile levels"):
        adapter.predict(steps=3, quantiles=[0.5, bad_quantile])


def test_MoiraiAdapter_predict_issues_warning_when_exog_provided():
    """
    Test predict issues IgnoredArgumentWarning when exog is not None.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.predict(steps=3, exog=pd.Series([1, 2, 3]))
    assert any(issubclass(w.category, IgnoredArgumentWarning) for w in caught)


def test_MoiraiAdapter_predict_issues_warning_when_last_window_exog_provided():
    """
    Test predict issues IgnoredArgumentWarning when last_window_exog is not None.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.predict(steps=3, last_window_exog=pd.Series([1, 2, 3]))
    assert any(issubclass(w.category, IgnoredArgumentWarning) for w in caught)


# Tests MoiraiAdapter.predict — point forecast (quantiles=None)
# ==============================================================================
def test_MoiraiAdapter_predict_returns_series_as_point_forecast():
    """
    Test predict returns a pd.Series when quantiles is None.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=5)
    assert isinstance(result, pd.Series)


def test_MoiraiAdapter_predict_series_correct_length():
    """
    Test the returned Series has exactly `steps` observations.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    assert len(adapter.predict(steps=12)) == 12


def test_MoiraiAdapter_predict_series_preserves_name():
    """
    Test the returned Series carries the same name as the training series.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    assert adapter.predict(steps=3).name == "sales"


def test_MoiraiAdapter_predict_series_correct_index():
    """
    Test the returned Series index starts right after the last training date
    and spans exactly `steps` periods with the same frequency.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=3)
    expected_start = y.index[-1] + y.index.freq
    expected_index = pd.date_range(start=expected_start, periods=3, freq=y.index.freq)
    pd.testing.assert_index_equal(result.index, expected_index)


def test_MoiraiAdapter_predict_point_forecast_values():
    """
    Test that point forecast (median, q=0.5) values equal 0.5.
    FakeMoirai2Forecast sets raw[i, q_idx, :] = (q_idx + 1) / 10,
    so q_idx=4 (q=0.5) gives 0.5.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=6)
    np.testing.assert_array_almost_equal(result.to_numpy(), np.full(6, 0.5))


def test_MoiraiAdapter_predict_passes_correct_steps_to_inference():
    """
    Test that the steps value is forwarded correctly to _run_inference.
    """
    fake_forecast = FakeMoirai2Forecast()
    adapter = make_adapter()
    adapter._forecast_obj = fake_forecast
    adapter.fit(series=y)
    adapter.predict(steps=7)
    assert fake_forecast._last_steps == 7


def test_MoiraiAdapter_predict_passes_single_input_to_model():
    """
    Test that exactly one input array is passed to predict in single-series mode.
    """
    fake_forecast = FakeMoirai2Forecast()
    adapter = make_adapter()
    adapter._forecast_obj = fake_forecast
    adapter.fit(series=y)
    adapter.predict(steps=5)
    assert len(fake_forecast.last_inputs) == 1


def test_MoiraiAdapter_predict_input_shape_is_T_by_1():
    """
    Test that the input array passed to predict has shape (T, 1).
    """
    fake_forecast = FakeMoirai2Forecast()
    adapter = make_adapter()
    adapter._forecast_obj = fake_forecast
    adapter.fit(series=y)
    adapter.predict(steps=5)
    assert fake_forecast.last_inputs[0].shape == (len(y), 1)


# Tests MoiraiAdapter.predict — quantile forecast
# ==============================================================================
def test_MoiraiAdapter_predict_returns_dataframe_when_quantiles_provided():
    """
    Test predict returns a pd.DataFrame when quantiles is specified.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=5, quantiles=[0.1, 0.5, 0.9])
    assert isinstance(result, pd.DataFrame)


def test_MoiraiAdapter_predict_dataframe_has_correct_columns():
    """
    Test that quantile DataFrame columns follow the q_<level> naming convention.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=5, quantiles=[0.1, 0.5, 0.9])
    assert list(result.columns) == ["q_0.1", "q_0.5", "q_0.9"]


def test_MoiraiAdapter_predict_dataframe_correct_length():
    """
    Test that the quantile DataFrame has exactly `steps` rows.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=7, quantiles=[0.1, 0.5, 0.9])
    assert len(result) == 7


def test_MoiraiAdapter_predict_dataframe_correct_index():
    """
    Test quantile DataFrame index starts right after the last training date.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=4, quantiles=[0.1, 0.5, 0.9])
    expected_start = y.index[-1] + y.index.freq
    expected_index = pd.date_range(start=expected_start, periods=4, freq=y.index.freq)
    pd.testing.assert_index_equal(result.index, expected_index)


def test_MoiraiAdapter_predict_quantile_values_correct():
    """
    Test quantile DataFrame values match FakeMoirai2Forecast output.
    FakeMoirai2Forecast sets raw[i, q_idx, :] = (q_idx + 1) / 10:
      q_0.1 → q_idx=0 → 0.1; q_0.5 → q_idx=4 → 0.5; q_0.9 → q_idx=8 → 0.9.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=5, quantiles=quantiles)
    for q_idx, q in enumerate([0.1, 0.5, 0.9]):
        expected_val = [0.1, 0.5, 0.9][q_idx]
        np.testing.assert_array_almost_equal(
            result[f"q_{q}"].to_numpy(), np.full(5, expected_val)
        )


def test_MoiraiAdapter_predict_all_supported_quantiles():
    """
    Test that all supported quantile levels are accepted without error and
    produce correctly named columns.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=3, quantiles=MoiraiAdapter.SUPPORTED_QUANTILES)
    expected_cols = [f"q_{q}" for q in MoiraiAdapter.SUPPORTED_QUANTILES]
    assert list(result.columns) == expected_cols


# Tests MoiraiAdapter.predict — last_window behaviour
# ==============================================================================
def test_MoiraiAdapter_predict_uses_last_window_forecast_index():
    """
    Test that the forecast index follows from last_window's index, not
    from stored history.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    last_window = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2025-01-01", periods=10, freq="ME"),
        name="sales",
    )
    result = adapter.predict(steps=3, last_window=last_window)
    expected_start = last_window.index[-1] + last_window.index.freq
    expected_index = pd.date_range(start=expected_start, periods=3, freq=last_window.index.freq)
    pd.testing.assert_index_equal(result.index, expected_index)


def test_MoiraiAdapter_predict_trims_last_window_to_context_length():
    """
    Test that a last_window longer than context_length is trimmed to
    context_length before inference.
    """
    context_length = 10
    fake_forecast = FakeMoirai2Forecast()
    adapter = make_adapter(context_length=context_length)
    adapter._forecast_obj = fake_forecast
    adapter.fit(series=y)
    long_window = pd.Series(
        np.arange(40, dtype=float),
        index=pd.date_range("2023-01-01", periods=40, freq="ME"),
        name="sales",
    )
    adapter.predict(steps=5, last_window=long_window)
    assert fake_forecast.last_inputs[0].shape == (context_length, 1)


def test_MoiraiAdapter_predict_without_fit_and_with_last_window():
    """
    Test that predict works without prior fit when last_window is provided.
    """
    adapter = make_adapter()
    last_window = pd.Series(
        np.arange(20, dtype=float),
        index=pd.date_range("2020-01-01", periods=20, freq="ME"),
        name="sales",
    )
    result = adapter.predict(steps=4, last_window=last_window)
    assert isinstance(result, pd.Series)
    assert len(result) == 4


# Tests MoiraiAdapter.predict — multi-series
# ==============================================================================
def test_MoiraiAdapter_predict_multiseries_point_returns_long_dataframe():
    """
    Test predict returns a long pd.DataFrame with columns
    ["level", "pred"] for a multi-series point forecast.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=6)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred"]


def test_MoiraiAdapter_predict_multiseries_level_column_contains_series_names():
    """
    Test that the "level" column contains each series name.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=3)
    assert set(result["level"].unique()) == {"s1", "s2"}


def test_MoiraiAdapter_predict_multiseries_point_correct_row_count():
    """
    Test that the multi-series point forecast has n_steps × n_series rows.
    """
    steps = 5
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=steps)
    assert len(result) == steps * 2


def test_MoiraiAdapter_predict_multiseries_point_correct_index():
    """
    Test that the multi-series point forecast index repeats each forecast
    timestamp once per series.
    """
    steps = 4
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=steps)
    expected_start = _y_wide.index[-1] + _y_wide.index.freq
    forecast_ts = pd.date_range(
        start=expected_start, periods=steps, freq=_y_wide.index.freq
    )
    expected_index = np.repeat(forecast_ts, 2)
    np.testing.assert_array_equal(result.index, expected_index)


def test_MoiraiAdapter_predict_multiseries_point_values_correct():
    """
    Test multi-series point forecast values (q=0.5 → 0.5 from fake model).
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=3)
    np.testing.assert_array_almost_equal(
        result["pred"].to_numpy(), np.full(6, 0.5)
    )


def test_MoiraiAdapter_predict_multiseries_from_dict():
    """
    Test multi-series point forecast works when history was fitted from a dict.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_dict)
    result = adapter.predict(steps=4)
    assert list(result.columns) == ["level", "pred"]
    assert len(result) == 4 * 2


def test_MoiraiAdapter_predict_multiseries_quantile_columns():
    """
    Test that multi-series quantile forecast has ["level", q_columns] columns.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=3, quantiles=quantiles)
    assert list(result.columns) == ["level", "q_0.1", "q_0.5", "q_0.9"]


def test_MoiraiAdapter_predict_multiseries_quantile_values_correct():
    """
    Test multi-series quantile values match FakeMoirai2Forecast output.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=3, quantiles=quantiles)
    for col, expected_val in zip(["q_0.1", "q_0.5", "q_0.9"], [0.1, 0.5, 0.9]):
        np.testing.assert_array_almost_equal(
            result[col].to_numpy(), np.full(6, expected_val)
        )


def test_MoiraiAdapter_predict_multiseries_passes_batched_inputs():
    """
    Test that all series are batched into a single predict call (one
    array per series in inputs_list).
    """
    fake_forecast = FakeMoirai2Forecast()
    adapter = make_adapter()
    adapter._forecast_obj = fake_forecast
    adapter.fit(series=_y_wide)
    adapter.predict(steps=3)
    assert len(fake_forecast.last_inputs) == 2


def test_MoiraiAdapter_predict_multiseries_input_shape_is_T_by_1():
    """
    Test that each batched input array has shape (T, 1).
    """
    fake_forecast = FakeMoirai2Forecast()
    adapter = make_adapter()
    adapter._forecast_obj = fake_forecast
    adapter.fit(series=_y_wide)
    adapter.predict(steps=3)
    for arr in fake_forecast.last_inputs:
        assert arr.ndim == 2
        assert arr.shape[1] == 1


def test_MoiraiAdapter_predict_multiseries_with_last_window_dict():
    """
    Test multi-series predict with last_window as a dict uses it as context.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    new_idx = pd.date_range("2025-06-01", periods=8, freq="ME")
    lw = {
        "s1": pd.Series(np.ones(8), index=new_idx, name="s1"),
        "s2": pd.Series(np.ones(8) * 2, index=new_idx, name="s2"),
    }
    result = adapter.predict(steps=3, last_window=lw)
    expected_start = new_idx[-1] + new_idx.freq
    expected_ts = pd.date_range(start=expected_start, periods=3, freq=new_idx.freq)
    expected_index = np.repeat(expected_ts, 2)
    np.testing.assert_array_equal(result.index, expected_index)


def test_MoiraiAdapter_predict_multiseries_with_last_window_wide_dataframe():
    """
    Test multi-series predict with last_window as a wide DataFrame.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    new_idx = pd.date_range("2025-06-01", periods=5, freq="ME")
    lw = pd.DataFrame({"s1": np.ones(5), "s2": np.ones(5) * 2}, index=new_idx)
    result = adapter.predict(steps=4, last_window=lw)
    assert list(result.columns) == ["level", "pred"]
    assert len(result) == 4 * 2


def test_MoiraiAdapter_predict_multiseries_trims_last_window_to_context_length():
    """
    Test that a last_window longer than context_length is trimmed in
    multi-series mode.
    """
    context_length = 5
    fake_forecast = FakeMoirai2Forecast()
    adapter = make_adapter(context_length=context_length)
    adapter._forecast_obj = fake_forecast
    adapter.fit(series=_y_wide)
    long_idx = pd.date_range("2020-01-01", periods=20, freq="ME")
    lw = {
        "s1": pd.Series(np.ones(20), index=long_idx, name="s1"),
        "s2": pd.Series(np.ones(20) * 2, index=long_idx, name="s2"),
    }
    adapter.predict(steps=3, last_window=lw)
    for arr in fake_forecast.last_inputs:
        assert arr.shape == (context_length, 1)


def test_MoiraiAdapter_predict_multiseries_level_order_tiles_series_names():
    """
    Test that within each timestep block the level order matches the series
    order (np.tile behaviour: [s1, s2, s1, s2, …]).
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=3)
    level_values = result["level"].tolist()
    # steps=3, n_series=2 → [s1, s2, s1, s2, s1, s2]
    assert level_values == ["s1", "s2", "s1", "s2", "s1", "s2"]


def test_MoiraiAdapter_fit_raises_ValueError_when_series_has_nan():
    """
    Test that fit raises ValueError when the series contains NaN values.
    """
    adapter = make_adapter()
    y_nan = y.copy()
    y_nan.iloc[5] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        adapter.fit(series=y_nan)


def test_MoiraiAdapter_fit_multiseries_raises_ValueError_for_nan_in_series():
    """
    Test that fit raises ValueError when any series in the dict contains NaN.
    """
    adapter = make_adapter()
    y_nan_dict = {"s1": _ys1.copy(), "s2": _ys2.copy()}
    y_nan_dict["s2"].iloc[5] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        adapter.fit(series=y_nan_dict)


def test_MoiraiAdapter_fit_multiseries_series_names_from_dataframe_columns():
    """
    Test that the keys of _history match the column names of the input DataFrame.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    assert list(adapter._history.keys()) == list(_y_wide.columns)


def test_MoiraiAdapter_fit_multiseries_series_names_from_dict_keys():
    """
    Test that the keys of _history match the keys of the input dict.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_dict)
    assert list(adapter._history.keys()) == list(_y_dict.keys())


def test_MoiraiAdapter_fit_multiseries_refitting_single_series_resets_flag():
    """
    Test that re-fitting on a single Series after a multi-series fit resets
    _is_multiseries to False and stores a Series in _history.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    assert adapter._is_multiseries is True
    adapter.fit(series=y)
    assert adapter._is_multiseries is False
    assert isinstance(adapter._history, pd.Series)
