# Unit test predict_quantiles ForecasterFoundational
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
    exog_predict_lw,
    df_exog,
    df_exog_predict,
)


# Tests predict_quantiles
# ==============================================================================

def test_predict_quantiles_returns_DataFrame():
    """
    predict_quantiles() returns a pandas DataFrame.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(steps=5)
    assert isinstance(result, pd.DataFrame)


def test_predict_quantiles_default_columns():
    """
    Default quantiles=[0.1, 0.5, 0.9] produce columns q_0.1, q_0.5, q_0.9.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(steps=5)
    assert list(result.columns) == ["q_0.1", "q_0.5", "q_0.9"]


def test_predict_quantiles_custom_quantiles():
    """
    Custom quantiles produce the matching column names.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(steps=5, quantiles=[0.05, 0.5, 0.95])
    assert list(result.columns) == ["q_0.05", "q_0.5", "q_0.95"]


def test_predict_quantiles_correct_length():
    """
    predict_quantiles() returns exactly `steps` rows.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(steps=7)
    assert len(result) == 7


def test_predict_quantiles_index_follows_training_series():
    """
    The forecast index continues directly from the training series index.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(steps=5)
    expected_index = pd.date_range(
        start=y.index[-1] + y.index.freq,
        periods=5,
        freq=y.index.freq,
    )
    pd.testing.assert_index_equal(result.index, expected_index)


def test_predict_quantiles_values_from_fake_pipeline():
    """
    With FakePipeline each quantile column equals its level value.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(steps=3, quantiles=[0.1, 0.5, 0.9])

    np.testing.assert_array_almost_equal(result["q_0.1"].values, [0.1, 0.1, 0.1])
    np.testing.assert_array_almost_equal(result["q_0.5"].values, [0.5, 0.5, 0.5])
    np.testing.assert_array_almost_equal(result["q_0.9"].values, [0.9, 0.9, 0.9])


def test_predict_quantiles_NotFittedError_when_not_fitted_and_no_last_window():
    """
    Raise NotFittedError when forecaster is not fitted and no last_window is given.
    """
    forecaster = make_forecaster()

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `predict_quantiles()`."
    )
    with pytest.raises(NotFittedError, match=err_msg):
        forecaster.predict_quantiles(steps=5)


def test_predict_quantiles_with_last_window_raises_if_not_fitted():
    """
    predict_quantiles() raises NotFittedError even when last_window is provided
    and the forecaster has not been fitted.
    """
    forecaster = make_forecaster()
    with pytest.raises(NotFittedError):
        forecaster.predict_quantiles(steps=3, last_window=y)


def test_predict_quantiles_single_quantile():
    """
    predict_quantiles() works correctly when only one quantile level is given.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(steps=4, quantiles=[0.5])
    assert list(result.columns) == ["q_0.5"]
    assert len(result) == 4


def test_predict_quantiles_with_exog():
    """
    predict_quantiles() with exog passes it to the pipeline correctly.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)
    result = forecaster.predict_quantiles(steps=5, exog=exog_predict)
    assert len(result) == 5
    assert "q_0.5" in result.columns


def test_predict_quantiles_with_last_window_and_last_window_exog():
    """
    predict_quantiles() accepts last_window and last_window_exog.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)
    result = forecaster.predict_quantiles(
        steps=3,
        exog=exog_predict_lw.iloc[:3],
        last_window=y_lw,
        last_window_exog=exog_lw,
    )
    assert len(result) == 3
    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize(
    "quantiles",
    [[0.1, 0.9], [0.25, 0.5, 0.75], [0.05, 0.5, 0.95]],
    ids=lambda v: f"quantiles={v}",
)
def test_predict_quantiles_monotone_columns(quantiles):
    """
    For FakePipeline outputs: quantile columns are monotonically non-decreasing
    (since quantile values equal the quantile levels).
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(steps=5, quantiles=quantiles)

    cols = list(result.columns)
    for i in range(len(cols) - 1):
        assert (result[cols[i]] <= result[cols[i + 1]]).all()


def test_predict_quantiles_exact_index_when_last_window_provided():
    """
    When last_window is provided, the forecast index must start from
    last_window.index[-1] + freq.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(steps=4, last_window=y_lw)
    expected_index = pd.date_range(
        start=y_lw.index[-1] + y_lw.index.freq,
        periods=4,
        freq=y_lw.index.freq,
    )
    pd.testing.assert_index_equal(result.index, expected_index)


def test_predict_quantiles_exact_values_when_last_window_provided():
    """
    With FakePipeline, predict_quantiles using last_window returns the expected
    quantile values regardless of context content.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster.predict_quantiles(
        steps=3, quantiles=[0.1, 0.5, 0.9], last_window=y_lw
    )
    np.testing.assert_array_almost_equal(result["q_0.1"].values, [0.1, 0.1, 0.1])
    np.testing.assert_array_almost_equal(result["q_0.5"].values, [0.5, 0.5, 0.5])
    np.testing.assert_array_almost_equal(result["q_0.9"].values, [0.9, 0.9, 0.9])


# Test exog trimming
# ==============================================================================

def test_predict_quantiles_exog_trimmed_to_steps():
    """
    predict_quantiles() trims exog to the first `steps` rows so users can
    safely pass a longer DataFrame (e.g. the entire test set).
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog)
    long_exog = pd.DataFrame(
        {"feat_a": np.arange(70, 80, dtype=float)},
        index=pd.date_range("2024-03-31", periods=10, freq="ME"),
    )
    # steps=5 means only the first 5 rows of long_exog (10 rows) should be used
    result = forecaster.predict_quantiles(steps=5, exog=long_exog)
    assert len(result) == 5
