# Unit test predict_interval ForecasterFoundation
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.foundation import ForecasterFoundation

# Fixtures
from .fixtures_forecaster_foundation import (
    make_forecaster,
    y,
    y_lw,
    exog,
    exog_lw,
    exog_predict,
    exog_predict_lw,
    series_df,
    lw_df,
)


# Tests predict_interval — errors
# ==============================================================================

def test_predict_interval_NotFittedError_when_not_fitted():
    """
    Raise NotFittedError when forecaster is not fitted, regardless of whether
    last_window is provided.
    """
    forecaster = make_forecaster()

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `predict_interval()`."
    )
    with pytest.raises(NotFittedError, match=err_msg):
        forecaster.predict_interval(steps=5)

    with pytest.raises(NotFittedError):
        forecaster.predict_interval(steps=3, last_window=y)


@pytest.mark.parametrize(
    "interval",
    [[10], [10, 50, 90], [90, 10], [50, 50], [-5, 90], [10, 105]],
    ids=[
        "one_element",
        "three_elements",
        "lower_greater_than_upper",
        "lower_equals_upper",
        "lower_negative",
        "upper_exceeds_100",
    ],
)
def test_predict_interval_ValueError_when_invalid_interval(interval):
    """
    predict_interval() raises ValueError for invalid interval specifications.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    with pytest.raises(ValueError):
        forecaster.predict_interval(steps=5, interval=interval)


# Tests predict_interval — single series basic output
# ==============================================================================

def test_predict_interval_output_when_single_series():
    """
    predict_interval() returns a DataFrame with correct columns, length,
    values, and index continuation from the training series.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_interval(steps=5, interval=[10, 90])

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred", "lower_bound", "upper_bound"]
    assert len(result) == 5
    np.testing.assert_array_almost_equal(result["pred"].values, [0.5] * 5)
    np.testing.assert_array_almost_equal(result["lower_bound"].values, [0.1] * 5)
    np.testing.assert_array_almost_equal(result["upper_bound"].values, [0.9] * 5)

    expected_index = pd.date_range(
        start=y.index[-1] + y.index.freq,
        periods=5,
        freq=y.index.freq,
    )
    pd.testing.assert_index_equal(result.index, expected_index)


def test_predict_interval_custom_percentiles():
    """
    Custom interval percentiles are forwarded to the pipeline correctly.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_interval(steps=3, interval=[5, 95])

    np.testing.assert_array_almost_equal(
        result["lower_bound"].values, [0.05] * 3
    )
    np.testing.assert_array_almost_equal(
        result["upper_bound"].values, [0.95] * 3
    )


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


# Tests predict_interval — multi-series basic output
# ==============================================================================

def test_predict_interval_output_when_multiseries():
    """
    predict_interval() with a multi-series fit returns correct columns,
    row count, and level values.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_df)
    result = forecaster.predict_interval(steps=5)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred", "lower_bound", "upper_bound"]
    assert len(result) == 5 * 2
    expected_levels = np.tile(["s1", "s2"], 5)
    np.testing.assert_array_equal(result["level"].values, expected_levels)
    assert (result["lower_bound"] <= result["pred"]).all()
    assert (result["pred"] <= result["upper_bound"]).all()


def test_predict_interval_with_levels_filter():
    """
    predict_interval() with `levels` filters the output to the specified series.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_df)

    result_list = forecaster.predict_interval(steps=5, levels=["s2"])
    assert set(result_list["level"].unique()) == {"s2"}
    assert len(result_list) == 5

    result_str = forecaster.predict_interval(steps=5, levels="s1")
    assert set(result_str["level"].unique()) == {"s1"}


# Tests predict_interval — last_window and exog
# ==============================================================================

def test_predict_interval_with_last_window_and_exog():
    """
    predict_interval() with last_window, last_window_exog, and exog returns
    the correct output. Exog longer than steps is trimmed automatically.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)

    # With exog
    result = forecaster.predict_interval(steps=5, exog=exog_predict)
    assert len(result) == 5
    assert list(result.columns) == ["level", "pred", "lower_bound", "upper_bound"]

    # With last_window + last_window_exog
    result_lw = forecaster.predict_interval(
        steps=3,
        exog=exog_predict_lw.iloc[:3],
        last_window=y_lw,
        last_window_exog=exog_lw,
    )
    assert len(result_lw) == 3

    # Exog trimmed to steps
    long_exog = pd.DataFrame(
        {"feat_a": np.arange(70, 80, dtype=float)},
        index=pd.date_range("2024-03-31", periods=10, freq="ME"),
    )
    result_trimmed = forecaster.predict_interval(steps=5, exog=long_exog)
    assert len(result_trimmed) == 5

    # last_window index and values (no exog)
    forecaster_no_exog = make_forecaster()
    forecaster_no_exog.fit(series=y)
    result_idx = forecaster_no_exog.predict_interval(
        steps=4, interval=[10, 90], last_window=y_lw
    )
    expected_index = pd.date_range(
        start=y_lw.index[-1] + y_lw.index.freq,
        periods=4,
        freq=y_lw.index.freq,
    )
    pd.testing.assert_index_equal(result_idx.index, expected_index)
    np.testing.assert_array_almost_equal(result_idx["pred"].values, [0.5] * 4)


def test_predict_interval_with_last_window_multiseries():
    """
    predict_interval() with multi-series last_window returns correct output.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_df)
    result = forecaster.predict_interval(steps=5, last_window=lw_df)
    assert list(result.columns) == ["level", "pred", "lower_bound", "upper_bound"]
