# Unit tests TimesFM25Adapter
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.foundational._adapters import TimesFM25Adapter


# Fixtures and helpers
# ==============================================================================
y = pd.Series(
    data=np.arange(50, dtype=float),
    index=pd.date_range("2020-01-01", periods=50, freq="ME"),
    name="sales",
)

air_passengers = np.array([
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
    air_passengers,
    index=pd.date_range(start="1949-01", periods=len(air_passengers), freq="MS"),
    name="y",
)

_idx_ms = pd.date_range("2020-01-01", periods=30, freq="ME")
_ys1 = pd.Series(np.arange(30, dtype=float), index=_idx_ms, name="s1")
_ys2 = pd.Series(np.arange(30, 60, dtype=float), index=_idx_ms, name="s2")
_y_wide = pd.DataFrame({"s1": _ys1, "s2": _ys2})
_y_dict = {"s1": _ys1.copy(), "s2": _ys2.copy()}


class FakeTimesFM25Model:
    """
    Fake TimesFM 2.5 model for testing without torch or timesfm dependency.

    ``forecast(horizon, inputs)`` returns:
      - ``point_forecast`` : ``np.zeros((n, horizon))``
      - ``quantile_forecast``: shape ``(n, horizon, 10)`` where ``[..., i]``
        equals ``i / 10.0`` (so index 0 = 0.0 mean, index 1 = 0.1, …,
        index 9 = 0.9).

    Records last call arguments for inspection.
    """

    def __init__(self):
        self.last_horizon = None
        self.last_inputs = None
        self.forecast_config = None

    @classmethod
    def from_pretrained(cls, model_id):
        return cls()

    def compile(self, forecast_config):
        self.forecast_config = forecast_config

    def forecast(self, horizon, inputs):
        self.last_horizon = horizon
        self.last_inputs = inputs
        n = len(inputs)
        point_forecast = np.zeros((n, horizon))
        quantile_vals = np.array([i / 10.0 for i in range(10)])
        quantile_forecast = np.broadcast_to(
            quantile_vals[np.newaxis, np.newaxis, :],
            (n, horizon, 10),
        ).copy()
        return point_forecast, quantile_forecast


def make_adapter(**kwargs) -> TimesFM25Adapter:
    """Return a TimesFM25Adapter pre-loaded with FakeTimesFM25Model."""
    defaults = dict(model_id="google/timesfm-2.5-200m-pytorch", model=FakeTimesFM25Model())
    defaults.update(kwargs)
    return TimesFM25Adapter(**defaults)


# Tests TimesFM25Adapter.__init__
# ==============================================================================
def test_TimesFM25Adapter_init_default_params():
    """
    Test that default parameter values are set correctly.
    """
    adapter = TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch")
    assert adapter.model_id == "google/timesfm-2.5-200m-pytorch"
    assert adapter.context_length == 512
    assert adapter.max_horizon == 512
    assert adapter.forecast_config_kwargs == {}
    assert adapter._model is None
    assert adapter._history is None
    assert adapter._is_fitted is False
    assert adapter._is_multiseries is False


def test_TimesFM25Adapter_init_raises_ValueError_when_context_length_not_positive():
    """
    Test that __init__ raises ValueError when context_length <= 0.
    """
    with pytest.raises(ValueError, match="`context_length` must be a positive integer"):
        TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch", context_length=0)


def test_TimesFM25Adapter_init_raises_ValueError_when_context_length_is_None():
    """
    Test that __init__ raises ValueError when context_length is None.
    """
    with pytest.raises(ValueError, match="`context_length` must be a positive integer"):
        TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch", context_length=None)


def test_TimesFM25Adapter_init_raises_ValueError_when_max_horizon_not_positive():
    """
    Test that __init__ raises ValueError when max_horizon <= 0.
    """
    with pytest.raises(ValueError, match="`max_horizon` must be a positive integer"):
        TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch", max_horizon=0)


def test_TimesFM25Adapter_init_raises_ValueError_when_max_horizon_is_None():
    """
    Test that __init__ raises ValueError when max_horizon is None.
    """
    with pytest.raises(ValueError, match="`max_horizon` must be a positive integer"):
        TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch", max_horizon=None)


def test_TimesFM25Adapter_init_forecast_config_kwargs_stored_as_copy():
    """
    Test that forecast_config_kwargs is stored as an independent copy.
    """
    original = {"normalize_inputs": True}
    adapter = TimesFM25Adapter(
        model_id="google/timesfm-2.5-200m-pytorch",
        forecast_config_kwargs=original,
    )
    original["extra"] = "should_not_appear"
    assert "extra" not in adapter.forecast_config_kwargs


# Tests TimesFM25Adapter.get_params / set_params
# ==============================================================================
def test_TimesFM25Adapter_get_params_returns_expected_keys():
    """
    Test that get_params returns all expected keys.
    """
    adapter = TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch")
    params = adapter.get_params()
    assert set(params.keys()) == {"model_id", "context_length", "max_horizon", "forecast_config_kwargs"}


def test_TimesFM25Adapter_get_params_round_trip():
    """
    Test that get_params returns the values set at construction.
    """
    adapter = TimesFM25Adapter(
        model_id="google/timesfm-2.5-200m-pytorch",
        context_length=256,
        max_horizon=128,
        forecast_config_kwargs={"normalize_inputs": True},
    )
    params = adapter.get_params()
    assert params["model_id"] == "google/timesfm-2.5-200m-pytorch"
    assert params["context_length"] == 256
    assert params["max_horizon"] == 128
    assert params["forecast_config_kwargs"] == {"normalize_inputs": True}


def test_TimesFM25Adapter_get_params_forecast_config_kwargs_is_None_when_empty():
    """
    Test that get_params returns None for forecast_config_kwargs when not set.
    """
    adapter = TimesFM25Adapter(model_id="google/timesfm-2.5-200m-pytorch")
    assert adapter.get_params()["forecast_config_kwargs"] is None


def test_TimesFM25Adapter_set_params_updates_context_length():
    """
    Test that set_params updates context_length correctly.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    adapter.set_params(context_length=128)
    assert adapter.context_length == 128


def test_TimesFM25Adapter_set_params_updates_max_horizon():
    """
    Test that set_params updates max_horizon correctly.
    """
    adapter = make_adapter()
    adapter.set_params(max_horizon=256)
    assert adapter.max_horizon == 256


def test_TimesFM25Adapter_set_params_returns_self():
    """
    Test that set_params returns the adapter instance.
    """
    adapter = make_adapter()
    result = adapter.set_params(context_length=64)
    assert result is adapter


def test_TimesFM25Adapter_set_params_resets_model_on_model_id_change():
    """
    Test that set_params resets _model when model_id changes.
    """
    adapter = make_adapter()
    assert adapter._model is not None
    adapter.set_params(model_id="google/timesfm-2.5-200m-pytorch-v2")
    assert adapter._model is None


def test_TimesFM25Adapter_set_params_resets_model_on_context_length_change():
    """
    Test that set_params resets _model when context_length changes.
    """
    adapter = make_adapter()
    assert adapter._model is not None
    adapter.set_params(context_length=256)
    assert adapter._model is None


def test_TimesFM25Adapter_set_params_resets_model_on_max_horizon_change():
    """
    Test that set_params resets _model when max_horizon changes.
    """
    adapter = make_adapter()
    assert adapter._model is not None
    adapter.set_params(max_horizon=128)
    assert adapter._model is None


def test_TimesFM25Adapter_set_params_resets_model_on_forecast_config_kwargs_change():
    """
    Test that set_params resets _model when forecast_config_kwargs changes.
    """
    adapter = make_adapter()
    assert adapter._model is not None
    adapter.set_params(forecast_config_kwargs={"normalize_inputs": True})
    assert adapter._model is None


def test_TimesFM25Adapter_set_params_raises_ValueError_on_invalid_context_length():
    """
    Test that set_params raises ValueError when context_length is invalid.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match="`context_length` must be a positive integer"):
        adapter.set_params(context_length=-1)


def test_TimesFM25Adapter_set_params_raises_ValueError_on_invalid_max_horizon():
    """
    Test that set_params raises ValueError when max_horizon is invalid.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match="`max_horizon` must be a positive integer"):
        adapter.set_params(max_horizon=0)


def test_TimesFM25Adapter_set_params_raises_ValueError_on_unknown_param():
    """
    Test that set_params raises ValueError for unrecognised parameter names.
    """
    adapter = make_adapter()
    with pytest.raises(ValueError, match="Invalid parameter"):
        adapter.set_params(unknown_param=42)


# Tests TimesFM25Adapter.fit
# ==============================================================================
def test_TimesFM25Adapter_fit_sets_is_fitted():
    """
    Test that fit sets _is_fitted to True.
    """
    adapter = make_adapter()
    assert adapter._is_fitted is False
    adapter.fit(series=y)
    assert adapter._is_fitted is True


def test_TimesFM25Adapter_fit_returns_self():
    """
    Test that fit returns the adapter instance.
    """
    adapter = make_adapter()
    result = adapter.fit(series=y)
    assert result is adapter


def test_TimesFM25Adapter_fit_stores_single_series_history():
    """
    Test that fit stores a copy of the series as _history (single-series).
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    assert isinstance(adapter._history, pd.Series)
    pd.testing.assert_series_equal(adapter._history, y)


def test_TimesFM25Adapter_fit_trims_history_to_context_length():
    """
    Test that fit trims _history to the last context_length observations.
    """
    context_length = 20
    adapter = make_adapter(context_length=context_length)
    adapter.fit(series=y)
    assert len(adapter._history) == context_length
    pd.testing.assert_series_equal(adapter._history, y.iloc[-context_length:])


def test_TimesFM25Adapter_fit_does_not_share_reference_with_input():
    """
    Test that _history is a copy, not a view of the input series.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    y_orig_values = adapter._history.values.copy()
    y.iloc[0] = 9999.0
    np.testing.assert_array_equal(adapter._history.values, y_orig_values)
    y.iloc[0] = 0.0  # restore


def test_TimesFM25Adapter_fit_sets_is_multiseries_false_for_series():
    """
    Test that _is_multiseries is False after fitting on a pd.Series.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    assert adapter._is_multiseries is False


def test_TimesFM25Adapter_fit_sets_is_multiseries_true_for_dataframe():
    """
    Test that _is_multiseries is True after fitting on a wide DataFrame.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    assert adapter._is_multiseries is True


def test_TimesFM25Adapter_fit_sets_is_multiseries_true_for_dict():
    """
    Test that _is_multiseries is True after fitting on a dict of Series.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_dict)
    assert adapter._is_multiseries is True


def test_TimesFM25Adapter_fit_multiseries_stores_history_dict():
    """
    Test that fit stores a dict of Series as _history in multi-series mode.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    assert isinstance(adapter._history, dict)
    assert set(adapter._history.keys()) == {"s1", "s2"}


def test_TimesFM25Adapter_fit_multiseries_trims_each_series_to_context_length():
    """
    Test that fit trims each series in _history to context_length.
    """
    context_length = 10
    adapter = make_adapter(context_length=context_length)
    adapter.fit(series=_y_wide)
    for name, s in adapter._history.items():
        assert len(s) == context_length, f"Series '{name}' not trimmed"


def test_TimesFM25Adapter_fit_ignores_exog_silently():
    """
    Test that passing exog to fit does not raise and is silently ignored.
    """
    exog = pd.DataFrame({"feat": np.arange(50, dtype=float)}, index=y.index)
    adapter = make_adapter()
    adapter.fit(series=y, exog=exog)
    assert adapter._is_fitted is True


def test_TimesFM25Adapter_fit_raises_TypeError_on_unsupported_series_type():
    """
    Test that fit raises TypeError when series is neither pd.Series,
    pd.DataFrame nor dict.
    """
    adapter = make_adapter()
    with pytest.raises(TypeError):
        adapter.fit(series=np.arange(50))


# Tests TimesFM25Adapter.predict — error handling
# ==============================================================================
def test_TimesFM25Adapter_predict_raises_ValueError_when_not_fitted_and_no_last_window():
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
def test_TimesFM25Adapter_predict_raises_ValueError_when_steps_not_positive(steps):
    """
    Test predict raises ValueError when steps is 0 or negative.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    with pytest.raises(ValueError, match="`steps` must be a positive integer"):
        adapter.predict(steps=steps)


def test_TimesFM25Adapter_predict_raises_ValueError_when_steps_exceed_max_horizon():
    """
    Test predict raises ValueError when steps > max_horizon.
    """
    adapter = make_adapter(max_horizon=10)
    adapter.fit(series=y)
    err_msg = re.escape("`steps` (15) exceeds `max_horizon` (10).")
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(steps=15)


@pytest.mark.parametrize(
    "bad_quantile",
    [0.05, 0.15, 0.25, 0.95, 1.1, -0.1],
    ids=lambda x: f"q_{x}",
)
def test_TimesFM25Adapter_predict_raises_ValueError_for_unsupported_quantile(bad_quantile):
    """
    Test predict raises ValueError for quantile levels not in SUPPORTED_QUANTILES.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    with pytest.raises(ValueError, match="TimesFM 2.5 only supports quantile levels"):
        adapter.predict(steps=3, quantiles=[0.5, bad_quantile])


def test_TimesFM25Adapter_predict_issues_warning_when_exog_provided():
    """
    Test predict issues IgnoredArgumentWarning when exog is not None.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    future_exog = pd.DataFrame({"f": np.zeros(3)}, index=pd.date_range("2024-03-01", periods=3, freq="ME"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.predict(steps=3, exog=future_exog)
    assert any(issubclass(w.category, IgnoredArgumentWarning) for w in caught)


def test_TimesFM25Adapter_predict_issues_warning_when_last_window_exog_provided():
    """
    Test predict issues IgnoredArgumentWarning when last_window_exog is not None.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    lw_exog = pd.DataFrame({"f": np.zeros(5)}, index=y.index[-5:])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.predict(steps=3, last_window_exog=lw_exog)
    assert any(issubclass(w.category, IgnoredArgumentWarning) for w in caught)


# Tests TimesFM25Adapter.predict — point forecast (quantiles=None)
# ==============================================================================
def test_TimesFM25Adapter_predict_returns_series_as_point_forecast():
    """
    Test predict returns a pd.Series when quantiles is None.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=5)
    assert isinstance(result, pd.Series)


def test_TimesFM25Adapter_predict_series_correct_length():
    """
    Test the returned Series has exactly `steps` observations.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=12)
    assert len(result) == 12


def test_TimesFM25Adapter_predict_series_preserves_name():
    """
    Test the returned Series carries the same name as the training series.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=3)
    assert result.name == "sales"


def test_TimesFM25Adapter_predict_series_correct_index():
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


def test_TimesFM25Adapter_predict_point_forecast_values():
    """
    Test that point forecast values equal 0.0 for all steps.
    FakeTimesFM25Model returns zeros as point_forecast.
    """
    adapter = make_adapter()
    adapter.fit(series=data)
    result = adapter.predict(steps=12)
    expected_index = pd.date_range("1961-01-01", periods=12, freq="MS")
    pd.testing.assert_index_equal(result.index, expected_index)
    np.testing.assert_array_equal(result.to_numpy(), np.zeros(12))


def test_TimesFM25Adapter_predict_passes_correct_horizon_to_model():
    """
    Test that the horizon passed to the model equals steps.
    """
    fake_model = FakeTimesFM25Model()
    adapter = make_adapter(model=fake_model)
    adapter.fit(series=y)
    adapter.predict(steps=7)
    assert fake_model.last_horizon == 7


def test_TimesFM25Adapter_predict_passes_single_input_to_model():
    """
    Test that exactly one input array is passed to the model in single-series mode.
    """
    fake_model = FakeTimesFM25Model()
    adapter = make_adapter(model=fake_model)
    adapter.fit(series=y)
    adapter.predict(steps=5)
    assert len(fake_model.last_inputs) == 1


# Tests TimesFM25Adapter.predict — quantile forecast
# ==============================================================================
def test_TimesFM25Adapter_predict_returns_dataframe_when_quantiles_provided():
    """
    Test predict returns a pd.DataFrame when quantiles is specified.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=5, quantiles=[0.1, 0.5, 0.9])
    assert isinstance(result, pd.DataFrame)


def test_TimesFM25Adapter_predict_dataframe_has_correct_columns():
    """
    Test that quantile DataFrame columns follow the q_<level> naming convention.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=5, quantiles=quantiles)
    assert list(result.columns) == ["q_0.1", "q_0.5", "q_0.9"]


def test_TimesFM25Adapter_predict_dataframe_correct_length():
    """
    Test that the quantile DataFrame has exactly `steps` rows.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=7, quantiles=[0.1, 0.5, 0.9])
    assert len(result) == 7


def test_TimesFM25Adapter_predict_dataframe_correct_index():
    """
    Test quantile DataFrame index starts right after the last training date.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=4, quantiles=[0.1, 0.5, 0.9])
    expected_start = y.index[-1] + y.index.freq
    expected_index = pd.date_range(start=expected_start, periods=4, freq=y.index.freq)
    pd.testing.assert_index_equal(result.index, expected_index)


def test_TimesFM25Adapter_predict_quantile_values_correct():
    """
    Test quantile DataFrame values match FakeTimesFM25Model output.
    FakeTimesFM25Model returns i/10 at index i:
      q_0.1 → index 1 → 0.1; q_0.5 → index 5 → 0.5; q_0.9 → index 9 → 0.9.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = make_adapter()
    adapter.fit(series=data)
    result = adapter.predict(steps=12, quantiles=quantiles)
    expected_index = pd.date_range("1961-01-01", periods=12, freq="MS")
    pd.testing.assert_index_equal(result.index, expected_index)
    for q in quantiles:
        np.testing.assert_array_almost_equal(
            result[f"q_{q}"].to_numpy(), np.full(12, q)
        )


def test_TimesFM25Adapter_predict_all_supported_quantiles():
    """
    Test that all supported quantile levels are accepted without error and
    produce correctly named columns.
    """
    adapter = make_adapter()
    adapter.fit(series=y)
    result = adapter.predict(steps=3, quantiles=TimesFM25Adapter.SUPPORTED_QUANTILES)
    expected_cols = [f"q_{q}" for q in TimesFM25Adapter.SUPPORTED_QUANTILES]
    assert list(result.columns) == expected_cols


# Tests TimesFM25Adapter.predict — last_window behaviour
# ==============================================================================
def test_TimesFM25Adapter_predict_uses_last_window_when_provided():
    """
    Test that last_window is used as context instead of stored _history.
    The forecast index must follow from last_window's index.
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


def test_TimesFM25Adapter_predict_trims_last_window_to_context_length():
    """
    Test that a last_window longer than context_length is trimmed before
    inference. The model must receive only context_length observations.
    """
    context_length = 10
    fake_model = FakeTimesFM25Model()
    adapter = make_adapter(model=fake_model, context_length=context_length)
    adapter.fit(series=y)

    long_window = pd.Series(
        np.arange(40, dtype=float),
        index=pd.date_range("2023-01-01", periods=40, freq="ME"),
        name="sales",
    )
    adapter.predict(steps=5, last_window=long_window)
    assert len(fake_model.last_inputs[0]) == context_length


def test_TimesFM25Adapter_predict_without_fit_and_with_last_window():
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


# Tests TimesFM25Adapter.predict — multi-series
# ==============================================================================
def test_TimesFM25Adapter_predict_multiseries_point_returns_long_dataframe():
    """
    Test that predict returns a long pd.DataFrame with columns
    [\"level\", \"pred\"] for a multi-series point forecast.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=6)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred"]


def test_TimesFM25Adapter_predict_multiseries_level_column_contains_series_names():
    """
    Test that the \"level\" column contains each series name repeated for
    each step.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=3)
    assert set(result["level"].unique()) == {"s1", "s2"}


def test_TimesFM25Adapter_predict_multiseries_point_correct_row_count():
    """
    Test that the multi-series point forecast has n_steps × n_series rows.
    """
    steps = 5
    n_series = 2
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=steps)
    assert len(result) == steps * n_series


def test_TimesFM25Adapter_predict_multiseries_point_correct_index():
    """
    Test that the multi-series point forecast index repeats each forecast
    timestamp once per series.
    """
    steps = 4
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=steps)
    expected_start = _y_wide.index[-1] + _y_wide.index.freq
    forecast_ts = pd.date_range(start=expected_start, periods=steps, freq=_y_wide.index.freq)
    expected_index = np.repeat(forecast_ts, 2)
    np.testing.assert_array_equal(result.index, expected_index)


def test_TimesFM25Adapter_predict_multiseries_from_dict():
    """
    Test multi-series point forecast works when history was fitted from a dict.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_dict)
    result = adapter.predict(steps=4)
    assert list(result.columns) == ["level", "pred"]
    assert len(result) == 4 * 2


def test_TimesFM25Adapter_predict_multiseries_quantile_columns():
    """
    Test that multi-series quantile forecast has [\"level\", q_columns] columns.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=3, quantiles=quantiles)
    assert list(result.columns) == ["level", "q_0.1", "q_0.5", "q_0.9"]


def test_TimesFM25Adapter_predict_multiseries_quantile_values_correct():
    """
    Test multi-series quantile values match FakeTimesFM25Model output.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = make_adapter()
    adapter.fit(series=_y_wide)
    result = adapter.predict(steps=3, quantiles=quantiles)
    for q in quantiles:
        np.testing.assert_array_almost_equal(
            result[f"q_{q}"].to_numpy(), np.full(6, q)
        )


def test_TimesFM25Adapter_predict_multiseries_with_last_window_dict():
    """
    Test multi-series predict with last_window as a dict uses it as context
    and produces the forecast index based on that window.
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


def test_TimesFM25Adapter_predict_multiseries_with_last_window_wide_dataframe():
    """
    Test multi-series predict with last_window as a wide DataFrame.
    """
    adapter = make_adapter()
    adapter.fit(series=_y_wide)

    new_idx = pd.date_range("2025-06-01", periods=5, freq="ME")
    lw = pd.DataFrame(
        {"s1": np.ones(5), "s2": np.ones(5) * 2},
        index=new_idx,
    )
    result = adapter.predict(steps=4, last_window=lw)
    assert list(result.columns) == ["level", "pred"]
    assert len(result) == 4 * 2


def test_TimesFM25Adapter_predict_multiseries_trims_last_window_to_context_length():
    """
    Test that a last_window longer than context_length is trimmed in
    multi-series mode.
    """
    context_length = 5
    fake_model = FakeTimesFM25Model()
    adapter = make_adapter(model=fake_model, context_length=context_length)
    adapter.fit(series=_y_wide)

    long_idx = pd.date_range("2020-01-01", periods=20, freq="ME")
    lw = {"s1": pd.Series(np.ones(20), index=long_idx, name="s1"),
          "s2": pd.Series(np.ones(20) * 2, index=long_idx, name="s2")}
    adapter.predict(steps=3, last_window=lw)
    for arr in fake_model.last_inputs:
        assert len(arr) == context_length
