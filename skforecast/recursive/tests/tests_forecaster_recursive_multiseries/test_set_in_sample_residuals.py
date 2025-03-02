# Unit test set_in_sample_residuals ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import series, exog


def test_set_in_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `set_in_sample_residuals()`."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_in_sample_residuals(series=series)


@pytest.mark.parametrize("diff_index", 
                         [pd.RangeIndex(start=50, stop=100), 
                          pd.date_range(start='1991-07-01', periods=50, freq='MS')], 
                         ids=lambda idx: f'diff_index: {idx[[0, -1]]}')
def test_set_in_sample_residuals_IndexError_when_series_has_different_index_than_training(diff_index):
    """
    Test IndexError is raised when series has different index than training.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, exog=exog['exog_1'])

    series_diff_index = series.copy()
    series_diff_index.index = diff_index
    series_diff_index_range = series_diff_index.index[[0, -1]]

    exog_diff_index = exog.copy()
    exog_diff_index.index = diff_index

    err_msg = re.escape(
        f"The index range for series '1' does not match the range "
        f"used during training. Please ensure the index is aligned "
        f"with the training data.\n"
        f"    Expected : {forecaster.training_range_['1']}\n"
        f"    Received : {series_diff_index_range}"
    )
    with pytest.raises(IndexError, match = err_msg):
        forecaster.set_in_sample_residuals(series=series_diff_index, exog=exog_diff_index['exog_1'])


def test_set_in_sample_residuals_ValueError_when_X_train_features_names_out_not_the_same():
    """
    Test ValueError is raised when X_train_features_names_out are different from 
    the ones used in training.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    err_msg = re.escape(
        "Feature mismatch detected after matrix creation. The features "
        "generated from the provided data do not match those used during "
        "the training process. To correctly set in-sample residuals, "
        "ensure that the same data and preprocessing steps are applied.\n"
        "    Expected output : ['lag_1', 'lag_2', 'lag_3', 'exog']\n"
        "    Current output  : ['lag_1', 'lag_2', 'lag_3']"
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_in_sample_residuals(y=y)


def test_set_in_sample_residuals_store_same_residuals_as_fit():
    """
    Test that set_in_sample_residuals stores same residuals as fit.
    """
    forecaster_1 = ForecasterRecursive(LinearRegression(), lags=3, binner_kwargs={'n_bins': 3})
    forecaster_1.fit(y=y, exog=exog, store_in_sample_residuals=True)

    forecaster_2 = ForecasterRecursive(LinearRegression(), lags=3, binner_kwargs={'n_bins': 3})
    forecaster_2.fit(y=y, exog=exog, store_in_sample_residuals=False)
    forecaster_2.set_in_sample_residuals(y=y, exog=exog)

    np.testing.assert_almost_equal(forecaster_1.in_sample_residuals_, forecaster_2.in_sample_residuals_)
    for k in forecaster_1.in_sample_residuals_by_bin_.keys():
        np.testing.assert_almost_equal(forecaster_1.in_sample_residuals_by_bin_[k], forecaster_2.in_sample_residuals_by_bin_[k])
    assert forecaster_1.binner_intervals_ == forecaster_2.binner_intervals_
