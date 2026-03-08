# Unit test FoundationalModels
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundational._foundational_models import (
    Chronos2Adapter,
    FoundationalModels,
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


class FakePipeline:
    """
    Fake Chronos-2 pipeline for testing without torch or chronos dependency.
    Stores last call arguments for inspection.
    """

    def __init__(self):
        self.last_inputs = None
        self.last_prediction_length = None
        self.last_quantile_levels = None

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        self.last_inputs = inputs
        self.last_prediction_length = prediction_length
        self.last_quantile_levels = quantile_levels

        n_q = len(quantile_levels)
        q_values = np.array(quantile_levels, dtype=float)
        arr = np.broadcast_to(q_values, (1, prediction_length, n_q)).copy()
        return [arr], [np.zeros((1, prediction_length))]


# Tests FoundationalModels.__init__
# ==============================================================================
def test_FoundationalModels_init_creates_Chronos2Adapter():
    """
    Test that FoundationalModels creates an internal Chronos2Adapter on init.
    """
    m = FoundationalModels("autogluon/chronos-2-small")
    assert isinstance(m.adapter, Chronos2Adapter)


def test_FoundationalModels_init_model_id_stored_in_adapter():
    """
    Test that the model_id string is forwarded to the adapter.
    """
    model_id = "autogluon/chronos-2-small"
    m = FoundationalModels(model_id)
    assert m.adapter.model_id == model_id


def test_FoundationalModels_init_kwargs_forwarded_to_adapter():
    """
    Test that keyword arguments (context_length, device_map) are forwarded to the adapter.
    """
    m = FoundationalModels(
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
    m = FoundationalModels("autogluon/chronos-2-small")
    assert m.is_fitted is False


def test_FoundationalModels_is_fitted_is_true_after_fit():
    """
    Test that is_fitted is True after fit has been called.
    """
    m = FoundationalModels("autogluon/chronos-2-small")
    m.fit(y=y)
    assert m.is_fitted is True


# Tests FoundationalModels.fit
# ==============================================================================
def test_FoundationalModels_fit_returns_self():
    """
    Test that fit returns the FoundationalModels instance.
    """
    m = FoundationalModels("autogluon/chronos-2-small")
    result = m.fit(y=y)
    assert result is m


def test_FoundationalModels_fit_delegates_to_adapter():
    """
    Test that fit stores history in the underlying adapter.
    """
    m = FoundationalModels("autogluon/chronos-2-small")
    m.fit(y=y)
    assert m.adapter._history is not None
    assert len(m.adapter._history) == len(y)


def test_FoundationalModels_fit_raises_TypeError_when_y_is_not_series():
    """
    Test that fit raises TypeError when y is not a pandas Series (via check_y).
    """
    m = FoundationalModels("autogluon/chronos-2-small")
    err_msg = re.escape(
        "`y` must be a pandas Series with a DatetimeIndex or a RangeIndex. "
        f"Found {type([1, 2, 3])}."
    )
    with pytest.raises(TypeError, match=err_msg):
        m.fit(y=[1, 2, 3])


def test_FoundationalModels_fit_raises_ValueError_when_y_has_nan():
    """
    Test that fit raises ValueError when y contains NaN values (via check_y).
    """
    m = FoundationalModels("autogluon/chronos-2-small")
    y_nan = y.copy()
    y_nan.iloc[3] = np.nan
    err_msg = re.escape("`y` has missing values.")
    with pytest.raises(ValueError, match=err_msg):
        m.fit(y=y_nan)


# Tests FoundationalModels.predict
# ==============================================================================
def test_FoundationalModels_predict_returns_series_for_point_forecast():
    """
    Test that predict returns a pandas Series when quantiles=None.
    """
    m = FoundationalModels("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(y=y)
    result = m.predict(steps=5)
    assert isinstance(result, pd.Series)


def test_FoundationalModels_predict_series_has_correct_length():
    """
    Test that the point forecast Series has exactly `steps` observations.
    """
    m = FoundationalModels("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(y=y)
    result = m.predict(steps=12)
    assert len(result) == 12


def test_FoundationalModels_predict_returns_dataframe_for_quantiles():
    """
    Test that predict returns a pandas DataFrame when quantiles are requested.
    """
    quantiles = [0.1, 0.5, 0.9]
    m = FoundationalModels("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(y=y)
    result = m.predict(steps=5, quantiles=quantiles)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["q_0.1", "q_0.5", "q_0.9"]


def test_FoundationalModels_predict_uses_last_window():
    """
    Test that predict respects last_window: the forecast index must follow
    from last_window's index.
    """
    m = FoundationalModels("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(y=y)

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
    m = FoundationalModels("autogluon/chronos-2-small", pipeline=pipeline)
    m.fit(y=y)

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
    m = FoundationalModels("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(y=data)
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
    m = FoundationalModels("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(y=data)
    result = m.predict(steps=12, quantiles=quantiles)
    expected_index = pd.date_range("1961-01-01", periods=12, freq="MS")
    pd.testing.assert_index_equal(result.index, expected_index)
    for q in quantiles:
        np.testing.assert_array_almost_equal(
            result[f"q_{q}"].to_numpy(), np.full(12, q)
        )
