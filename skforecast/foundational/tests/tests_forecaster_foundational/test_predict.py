# Unit test predict ForecasterFoundational
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


# Tests predict
# ==============================================================================

def test_predict_returns_Series():
    """
    predict() returns a pandas Series.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    predictions = forecaster.predict(steps=5)
    assert isinstance(predictions, pd.Series)


def test_predict_output_length():
    """
    predict() returns exactly `steps` values.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    predictions = forecaster.predict(steps=5)
    assert len(predictions) == 5


def test_predict_name_is_pred():
    """
    The returned Series has name='pred'.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    predictions = forecaster.predict(steps=5)
    assert predictions.name == "pred"


def test_predict_index_follows_training_series():
    """
    The prediction index continues directly from the training series index.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    predictions = forecaster.predict(steps=5)
    expected_index = pd.date_range(
        start=y.index[-1] + y.index.freq,
        periods=5,
        freq=y.index.freq,
    )
    pd.testing.assert_index_equal(predictions.index, expected_index)


def test_predict_values_from_fake_pipeline():
    """
    With FakePipeline the point forecast equals the median quantile (0.5).
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    predictions = forecaster.predict(steps=3)
    # FakePipeline returns quantile level as value; median = 0.5.
    np.testing.assert_array_almost_equal(predictions.values, [0.5, 0.5, 0.5])


def test_predict_NotFittedError_when_not_fitted_and_no_last_window():
    """
    Raise NotFittedError when forecaster is not fitted and no last_window is given.
    """
    forecaster = make_forecaster()

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `predict()`."
    )
    with pytest.raises(NotFittedError, match=err_msg):
        forecaster.predict(steps=5)


def test_predict_with_last_window_bypasses_is_fitted():
    """
    predict() with last_window works even when is_fitted=False.
    """
    forecaster = make_forecaster()
    # Do not call fit; pass last_window directly.
    predictions = forecaster.predict(steps=3, last_window=y)
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == 3


def test_predict_does_not_modify_y():
    """
    predict() must not modify the training series.
    """
    forecaster = make_forecaster()
    y_copy = y.copy()
    forecaster.fit(series=y)
    forecaster.predict(steps=5)
    pd.testing.assert_series_equal(y, y_copy)


def test_predict_does_not_modify_exog():
    """
    predict() must not modify the future exog.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)
    exog_predict_copy = exog_predict.copy()
    forecaster.predict(steps=5, exog=exog_predict)
    pd.testing.assert_frame_equal(exog_predict, exog_predict_copy)


def test_predict_with_exog_returns_correct_length():
    """
    predict() with exog returns the correct number of steps.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)
    predictions = forecaster.predict(steps=5, exog=exog_predict)
    assert len(predictions) == 5


def test_predict_with_last_window_and_last_window_exog():
    """
    predict() accepts last_window and last_window_exog simultaneously.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)
    predictions = forecaster.predict(
        steps=3,
        exog=exog_predict.iloc[:3],
        last_window=y_lw,
        last_window_exog=exog_lw,
    )
    assert len(predictions) == 3
    assert isinstance(predictions, pd.Series)


def test_predict_exact_index_when_last_window_provided():
    """
    When last_window is provided, the forecast index must start from
    last_window.index[-1] + freq, not from the training series end.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    # y_lw.index[-1] is later than y.index[-1]
    predictions = forecaster.predict(steps=4, last_window=y_lw)
    expected_index = pd.date_range(
        start=y_lw.index[-1] + y_lw.index.freq,
        periods=4,
        freq=y_lw.index.freq,
    )
    pd.testing.assert_index_equal(predictions.index, expected_index)


def test_predict_exact_values_when_last_window_provided():
    """
    With FakePipeline, predict() using last_window still returns 0.5 for every
    step regardless of context content.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    predictions = forecaster.predict(steps=3, last_window=y_lw)
    np.testing.assert_array_almost_equal(predictions.values, [0.5, 0.5, 0.5])


def test_predict_exact_index_with_context_length():
    """
    context_length trimming must not affect the output index: forecasts still
    start from the position after the last observation in `last_window`.
    """
    forecaster = make_forecaster(context_length=10)
    forecaster.fit(series=y)
    predictions = forecaster.predict(steps=4, last_window=y_lw)
    expected_index = pd.date_range(
        start=y_lw.index[-1] + y_lw.index.freq,
        periods=4,
        freq=y_lw.index.freq,
    )
    pd.testing.assert_index_equal(predictions.index, expected_index)


def test_predict_exact_values_with_context_length():
    """
    context_length trimming must not change the FakePipeline output values.
    """
    forecaster = make_forecaster(context_length=10)
    forecaster.fit(series=y)
    predictions = forecaster.predict(steps=3, last_window=y_lw)
    np.testing.assert_array_almost_equal(predictions.values, [0.5, 0.5, 0.5])


# Test exog trimming
# ==============================================================================

def test_predict_exog_trimmed_to_steps():
    """
    predict() trims exog to the first `steps` rows so users can safely
    pass a longer DataFrame (e.g. the entire test set).
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)
    long_exog = pd.DataFrame(
        {"feat_a": np.arange(70, 80, dtype=float)},
        index=pd.date_range("2024-07-31", periods=10, freq="ME"),
    )
    # steps=5 means only the first 5 rows of long_exog (10 rows) should be used
    result = forecaster.predict(steps=5, exog=long_exog)
    assert len(result) == 5
