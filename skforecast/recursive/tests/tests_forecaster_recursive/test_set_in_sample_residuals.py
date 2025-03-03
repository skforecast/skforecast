# Unit test set_in_sample_residuals ForecasterRecursive
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursive

# Fixtures
from .fixtures_forecaster_recursive import y, exog


def test_set_in_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `set_in_sample_residuals()`."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_in_sample_residuals(y=y)


@pytest.mark.parametrize("diff_index", 
                         [pd.RangeIndex(start=50, stop=100), 
                          pd.date_range(start='1991-07-01', periods=50, freq='MS')], 
                         ids=lambda idx: f'diff_index: {idx[[0, -1]]}')
def test_set_in_sample_residuals_IndexError_when_y_has_different_index_than_training(diff_index):
    """
    Test IndexError is raised when y has different index than training.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)

    y_diff_index = y.copy()
    y_diff_index.index = diff_index
    y_diff_index_range = y_diff_index.index[[0, -1]]

    err_msg = re.escape(
        f"The index range of `y` does not match the range "
        f"used during training. Please ensure the index is aligned "
        f"with the training data.\n"
        f"    Expected : {forecaster.training_range_}\n"
        f"    Received : {y_diff_index_range}"
    )
    with pytest.raises(IndexError, match = err_msg):
        forecaster.set_in_sample_residuals(y=y_diff_index)


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
    forecaster_1 = ForecasterRecursive(
        LinearRegression(), lags=3, transformer_y=StandardScaler(), 
        differentiation=1, binner_kwargs={'n_bins': 3}
    )
    forecaster_1.fit(y=y, exog=exog, store_in_sample_residuals=True)

    forecaster_2 = ForecasterRecursive(
        LinearRegression(), lags=3, transformer_y=StandardScaler(), 
        differentiation=1, binner_kwargs={'n_bins': 3}
    )
    forecaster_2.fit(y=y, exog=exog, store_in_sample_residuals=False)
    scaler_id_after_fit = id(forecaster_2.transformer_y)
    differentiator_id_after_fit = id(forecaster_2.differentiator)
    forecaster_2.set_in_sample_residuals(y=y, exog=exog)
    scaler_id_after_set_in_sample_residuals = id(forecaster_2.transformer_y)
    differentiator_id_after_set_in_sample_residuals = id(forecaster_2.differentiator)

    # Attributes
    assert forecaster_1.X_train_window_features_names_out_ == forecaster_2.X_train_window_features_names_out_
    assert forecaster_1.X_train_features_names_out_ == forecaster_2.X_train_features_names_out_

    # Transformer
    assert scaler_id_after_fit == scaler_id_after_set_in_sample_residuals

    # Differentiator
    assert differentiator_id_after_fit == differentiator_id_after_set_in_sample_residuals

    # Residuals
    np.testing.assert_almost_equal(forecaster_1.in_sample_residuals_, forecaster_2.in_sample_residuals_)
    for k in forecaster_1.in_sample_residuals_by_bin_.keys():
        np.testing.assert_almost_equal(forecaster_1.in_sample_residuals_by_bin_[k], forecaster_2.in_sample_residuals_by_bin_[k])
    assert forecaster_1.binner_intervals_ == forecaster_2.binner_intervals_
