# Unit test predict_interval ForecasterFoundational
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.foundational import ForecasterFoundational

# Fixtures
from .fixtures_forecaster_foundational import (
    make_forecaster,
    y,
    y_lw,
    exog,
    exog_lw,
    exog_predict,
    df_exog,
    df_exog_predict,
)


# Tests predict_interval
# ==============================================================================

def test_predict_interval_returns_DataFrame():
    """
    predict_interval() returns a pandas DataFrame.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_interval(steps=5)
    assert isinstance(result, pd.DataFrame)


def test_predict_interval_has_correct_columns():
    """
    predict_interval() returns a DataFrame with columns [pred, lower_bound, upper_bound].
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_interval(steps=5)
    assert list(result.columns) == ["pred", "lower_bound", "upper_bound"]


def test_predict_interval_correct_length():
    """
    predict_interval() returns exactly `steps` rows.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_interval(steps=5)
    assert len(result) == 5


def test_predict_interval_index_follows_training_series():
    """
    The forecast index continues directly from the training series index.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_interval(steps=5)
    expected_index = pd.date_range(
        start=y.index[-1] + y.index.freq,
        periods=5,
        freq=y.index.freq,
    )
    pd.testing.assert_index_equal(result.index, expected_index)


def test_predict_interval_values_from_fake_pipeline():
    """
    With FakePipeline the interval values equal the quantile levels:
    - pred  == 0.5  (median)
    - lower == lower_q   (e.g. 0.1 for interval=[10, 90])
    - upper == upper_q   (e.g. 0.9 for interval=[10, 90])
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_interval(steps=3, interval=[10, 90])

    np.testing.assert_array_almost_equal(result["pred"].values, [0.5, 0.5, 0.5])
    np.testing.assert_array_almost_equal(result["lower_bound"].values, [0.1, 0.1, 0.1])
    np.testing.assert_array_almost_equal(result["upper_bound"].values, [0.9, 0.9, 0.9])


def test_predict_interval_custom_percentiles():
    """
    Custom interval percentiles are forwarded to the pipeline correctly.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_interval(steps=3, interval=[5, 95])

    # FakePipeline: each value equals its quantile level.
    np.testing.assert_array_almost_equal(result["lower_bound"].values, [0.05, 0.05, 0.05])
    np.testing.assert_array_almost_equal(result["upper_bound"].values, [0.95, 0.95, 0.95])


def test_predict_interval_NotFittedError_when_not_fitted_and_no_last_window():
    """
    Raise NotFittedError when forecaster is not fitted and no last_window is given.
    """
    forecaster = make_forecaster()

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `predict_interval()`."
    )
    with pytest.raises(NotFittedError, match=err_msg):
        forecaster.predict_interval(steps=5)


def test_predict_interval_with_last_window_bypasses_is_fitted():
    """
    predict_interval() with last_window works even when is_fitted=False.
    """
    forecaster = make_forecaster()
    result = forecaster.predict_interval(steps=3, last_window=y)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_predict_interval_with_exog():
    """
    predict_interval() with future exog passes it to the pipeline.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)
    result = forecaster.predict_interval(steps=5, exog=exog_predict)
    assert len(result) == 5
    assert list(result.columns) == ["pred", "lower_bound", "upper_bound"]


def test_predict_interval_with_last_window_and_last_window_exog():
    """
    predict_interval() accepts last_window and last_window_exog.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)
    result = forecaster.predict_interval(
        steps=3,
        exog=exog_predict.iloc[:3],
        last_window=y_lw,
        last_window_exog=exog_lw,
    )
    assert len(result) == 3
    assert list(result.columns) == ["pred", "lower_bound", "upper_bound"]


@pytest.mark.parametrize(
    "interval",
    [[10, 90], [5, 95], [25, 75]],
    ids=lambda v: f"interval={v}",
)
def test_predict_interval_lower_le_pred_le_upper(interval):
    """
    For FakePipeline outputs: lower_bound <= pred <= upper_bound for all rows.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_interval(steps=5, interval=interval)

    assert (result["lower_bound"] <= result["pred"]).all()
    assert (result["pred"] <= result["upper_bound"]).all()


def test_predict_interval_exact_index_when_last_window_provided():
    """
    When last_window is provided, the forecast index must start from
    last_window.index[-1] + freq.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_interval(steps=4, last_window=y_lw)
    expected_index = pd.date_range(
        start=y_lw.index[-1] + y_lw.index.freq,
        periods=4,
        freq=y_lw.index.freq,
    )
    pd.testing.assert_index_equal(result.index, expected_index)


def test_predict_interval_exact_values_when_last_window_provided():
    """
    With FakePipeline, predict_interval using last_window returns the expected
    quantile values regardless of context content.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_interval(steps=3, interval=[10, 90], last_window=y_lw)
    np.testing.assert_array_almost_equal(result["pred"].values, [0.5, 0.5, 0.5])
    np.testing.assert_array_almost_equal(result["lower_bound"].values, [0.1, 0.1, 0.1])
    np.testing.assert_array_almost_equal(result["upper_bound"].values, [0.9, 0.9, 0.9])


# Tests interval validation
# ==============================================================================

def test_predict_interval_raises_when_interval_has_one_element():
    """
    predict_interval() raises ValueError when interval has only one element.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    with pytest.raises(ValueError, match="`interval` must be a sequence of exactly two values"):
        forecaster.predict_interval(steps=5, interval=[10])


def test_predict_interval_raises_when_interval_has_three_elements():
    """
    predict_interval() raises ValueError when interval has three elements.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    with pytest.raises(ValueError, match="`interval` must be a sequence of exactly two values"):
        forecaster.predict_interval(steps=5, interval=[10, 50, 90])


def test_predict_interval_raises_when_lower_greater_than_upper():
    """
    predict_interval() raises ValueError when lower >= upper (reversed interval).
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    with pytest.raises(ValueError, match="0 <= lower < upper <= 100"):
        forecaster.predict_interval(steps=5, interval=[90, 10])


def test_predict_interval_raises_when_lower_equals_upper():
    """
    predict_interval() raises ValueError when lower == upper.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    with pytest.raises(ValueError, match="0 <= lower < upper <= 100"):
        forecaster.predict_interval(steps=5, interval=[50, 50])


def test_predict_interval_raises_when_lower_is_negative():
    """
    predict_interval() raises ValueError when lower < 0.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    with pytest.raises(ValueError, match="0 <= lower < upper <= 100"):
        forecaster.predict_interval(steps=5, interval=[-5, 90])


def test_predict_interval_raises_when_upper_exceeds_100():
    """
    predict_interval() raises ValueError when upper > 100.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    with pytest.raises(ValueError, match="0 <= lower < upper <= 100"):
        forecaster.predict_interval(steps=5, interval=[10, 105])


# Tests exog trimming
# ==============================================================================

def test_predict_interval_exog_trimmed_to_steps():
    """
    predict_interval() trims exog to the first `steps` rows so users can
    safely pass a longer DataFrame (e.g. the entire test set).
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)
    long_exog = pd.DataFrame(
        {"feat_a": np.arange(70, 80, dtype=float)},
        index=pd.date_range("2024-07-31", periods=10, freq="ME"),
    )
    # steps=5 means only the first 5 rows of long_exog (10 rows) should be used
    result = forecaster.predict_interval(steps=5, exog=long_exog)
    assert len(result) == 5
