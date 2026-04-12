# Unit test predict ForecasterFoundation
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
    df_exog,
    df_exog_predict,
    series_df,
    series_dict,
    lw_df,
    lw_dict,
)


# Tests predict — errors
# ==============================================================================

def test_predict_NotFittedError_when_not_fitted():
    """
    Raise NotFittedError when forecaster is not fitted, regardless of whether
    last_window is provided.
    """
    forecaster = make_forecaster()

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `predict()`."
    )
    with pytest.raises(NotFittedError, match=err_msg):
        forecaster.predict(steps=5)

    with pytest.raises(NotFittedError):
        forecaster.predict(steps=3, last_window=y)


# Tests predict — single series basic output
# ==============================================================================

def test_predict_output_when_single_series():
    """
    predict() returns a DataFrame with correct columns, length, values,
    and index continuation from the training series.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    predictions = forecaster.predict(steps=5)

    assert isinstance(predictions, pd.DataFrame)
    assert list(predictions.columns) == ["level", "pred"]
    assert len(predictions) == 5
    # FakePipeline returns quantile level as value; median = 0.5
    np.testing.assert_array_almost_equal(
        predictions["pred"].values, [0.5] * 5
    )
    expected_index = pd.date_range(
        start=y.index[-1] + y.index.freq,
        periods=5,
        freq=y.index.freq,
    )
    pd.testing.assert_index_equal(predictions.index, expected_index)


# Tests predict — multi-series basic output
# ==============================================================================

def test_predict_output_when_multiseries():
    """
    predict() with a multi-series fit returns a DataFrame with n_steps × n_series
    rows, correct columns, and level values tiling series names.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_df)
    result = forecaster.predict(steps=5)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred"]
    assert len(result) == 5 * 2
    expected_levels = np.tile(["s1", "s2"], 5)
    np.testing.assert_array_equal(result["level"].values, expected_levels)


@pytest.mark.parametrize(
    "levels, expected_levels, expected_len",
    [
        (["s1"], {"s1"}, 5),
        ("s2", {"s2"}, 5),
    ],
    ids=["levels_list", "levels_str"],
)
def test_predict_with_levels_filter(levels, expected_levels, expected_len):
    """
    predict() with `levels` filters the output to the specified series.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_df)
    result = forecaster.predict(steps=5, levels=levels)
    assert set(result["level"].unique()) == expected_levels
    assert len(result) == expected_len


# Tests predict — last_window
# ==============================================================================

def test_predict_output_when_last_window_provided():
    """
    When last_window is provided, predict() uses it as context. The forecast
    index starts from last_window.index[-1] + freq. context_length trimming
    must not affect the output index or values.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    predictions = forecaster.predict(steps=4, last_window=y_lw)

    expected_index = pd.date_range(
        start=y_lw.index[-1] + y_lw.index.freq,
        periods=4,
        freq=y_lw.index.freq,
    )
    pd.testing.assert_index_equal(predictions.index, expected_index)
    np.testing.assert_array_almost_equal(
        predictions["pred"].values, [0.5] * 4
    )

    # context_length trimming must not change output
    forecaster_short = make_forecaster(context_length=10)
    forecaster_short.fit(series=y)
    predictions_short = forecaster_short.predict(steps=4, last_window=y_lw)
    pd.testing.assert_index_equal(predictions_short.index, expected_index)
    np.testing.assert_array_almost_equal(
        predictions_short["pred"].values, [0.5] * 4
    )


@pytest.mark.parametrize(
    "last_window",
    [lw_df, lw_dict],
    ids=["DataFrame", "dict"],
)
def test_predict_with_last_window_multiseries(last_window):
    """
    predict() with multi-series last_window (DataFrame or dict) returns
    a DataFrame with correct columns.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_df)
    result = forecaster.predict(steps=5, last_window=last_window)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["level", "pred"]


def test_predict_unfitted_with_last_window_raises_not_fitted_error():
    """
    predict() raises NotFittedError even when last_window is provided
    and the forecaster has not been fitted.
    """
    forecaster = make_forecaster()
    with pytest.raises(NotFittedError):
        forecaster.predict(steps=3, last_window=lw_df)


# Tests predict — exog
# ==============================================================================

def test_predict_with_exog_and_last_window_exog():
    """
    predict() accepts exog and last_window_exog, returning the correct output.
    Exog longer than steps is trimmed automatically.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)

    # Basic exog
    predictions = forecaster.predict(steps=5, exog=exog_predict)
    assert len(predictions) == 5

    # With last_window + last_window_exog
    predictions_lw = forecaster.predict(
        steps=3,
        exog=exog_predict_lw.iloc[:3],
        last_window=y_lw,
        last_window_exog=exog_lw,
    )
    assert len(predictions_lw) == 3
    assert isinstance(predictions_lw, pd.DataFrame)

    # Exog trimmed to steps
    long_exog = pd.DataFrame(
        {"feat_a": np.arange(70, 80, dtype=float)},
        index=pd.date_range("2024-03-31", periods=10, freq="ME"),
    )
    result = forecaster.predict(steps=5, exog=long_exog)
    assert len(result) == 5


# Tests predict — does not modify input
# ==============================================================================

def test_predict_does_not_modify_input():
    """
    predict() must not modify the training series or the future exog.
    """
    forecaster = make_forecaster()
    y_copy = y.copy()
    exog_copy = exog_predict.copy()
    forecaster.fit(series=y, exog=exog)
    forecaster.predict(steps=5, exog=exog_predict)
    pd.testing.assert_series_equal(y, y_copy)
    pd.testing.assert_frame_equal(exog_predict, exog_copy)
