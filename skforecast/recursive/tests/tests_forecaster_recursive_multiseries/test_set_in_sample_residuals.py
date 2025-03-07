# Unit test set_in_sample_residuals ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
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
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series, exog=exog['exog_1'])

    err_msg = re.escape(
        "Feature mismatch detected after matrix creation. The features "
        "generated from the provided data do not match those used during "
        "the training process. To correctly set in-sample residuals, "
        "ensure that the same data and preprocessing steps are applied.\n"
        "    Expected output : ['lag_1', 'lag_2', 'lag_3', '_level_skforecast', 'exog_1']\n"
        "    Current output  : ['lag_1', 'lag_2', 'lag_3', '_level_skforecast']"
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_in_sample_residuals(series=series)


@pytest.mark.parametrize("encoding", 
                         ["ordinal", "ordinal_category", "onehot", None],
                         ids = lambda value: f'encoding: {value}')
def test_set_in_sample_residuals_store_same_residuals_as_fit(encoding):
    """
    Test that set_in_sample_residuals stores same residuals as fit.
    """
    forecaster_1 = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, transformer_series=StandardScaler(), 
        differentiation=1, binner_kwargs={'n_bins': 3}
    )
    forecaster_1.fit(series=series, exog=exog['exog_1'], store_in_sample_residuals=True)
    results_residuals_1 = forecaster_1.in_sample_residuals_
    results_residuals_bin_1 = forecaster_1.in_sample_residuals_by_bin_
    results_binner_intervals_1 = forecaster_1.binner_intervals_

    forecaster_2 = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, transformer_series=StandardScaler(), 
        differentiation=1, binner_kwargs={'n_bins': 3}
    )
    forecaster_2.fit(series=series, exog=exog['exog_1'], store_in_sample_residuals=False)
    scaler_id_after_fit = {
        level: id(scaler) for level, scaler in forecaster_2.transformer_series_.items()
    }
    differentiator_id_after_fit = {
        level: id(differentiator) for level, differentiator in forecaster_2.differentiator_.items()
    }
    forecaster_2.set_in_sample_residuals(series=series, exog=exog['exog_1'])
    scaler_id_after_set_in_sample_residuals = {
        level: id(scaler) for level, scaler in forecaster_2.transformer_series_.items()
    }
    differentiator_id_after_set_in_sample_residuals = {
        level: id(differentiator) for level, differentiator in forecaster_2.differentiator_.items()
    }
    results_residuals_2 = forecaster_2.in_sample_residuals_
    results_residuals_bin_2 = forecaster_2.in_sample_residuals_by_bin_
    results_binner_intervals_2 = forecaster_2.binner_intervals_

    # Attributes
    assert forecaster_1.series_names_in_ == forecaster_2.series_names_in_
    assert forecaster_1.X_train_series_names_in_ == forecaster_2.X_train_series_names_in_
    assert forecaster_1.X_train_window_features_names_out_ == forecaster_2.X_train_window_features_names_out_
    assert forecaster_1.X_train_features_names_out_ == forecaster_2.X_train_features_names_out_

    # Transformers
    for level in scaler_id_after_fit.keys():
        assert scaler_id_after_fit[level] == scaler_id_after_set_in_sample_residuals[level]

    # Differentiators
    for level in differentiator_id_after_fit.keys():
        assert differentiator_id_after_fit[level] == differentiator_id_after_set_in_sample_residuals[level]

    # In-sample residuals
    assert results_residuals_1.keys() == results_residuals_2.keys()
    for k in results_residuals_1.keys():
        np.testing.assert_array_almost_equal(
            np.sort(results_residuals_1[k]), np.sort(results_residuals_2[k])
        )

    # In-sample residuals by bin
    assert results_residuals_bin_1.keys() == results_residuals_bin_2.keys()
    for level in results_residuals_bin_1.keys():
        assert results_residuals_bin_1[level].keys() == results_residuals_bin_2[level].keys()
        for k in results_residuals_bin_1[level].keys():
            np.testing.assert_array_almost_equal(
                results_residuals_bin_1[level][k], results_residuals_bin_2[level][k]
            )
    
    # Binner intervals
    assert results_binner_intervals_1.keys() == results_binner_intervals_2.keys()
    for level in results_binner_intervals_1.keys():
        assert results_binner_intervals_1[level].keys() == results_binner_intervals_2[level].keys()
        for k in results_binner_intervals_1[level].keys():
            assert results_binner_intervals_1[level][k][0] == approx(results_binner_intervals_2[level][k][0])
            assert results_binner_intervals_1[level][k][1] == approx(results_binner_intervals_2[level][k][1])
