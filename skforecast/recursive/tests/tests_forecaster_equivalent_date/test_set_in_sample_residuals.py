# Unit test set_in_sample_residuals ForecasterEquivalentDate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.exceptions import NotFittedError
from skforecast.recursive import ForecasterEquivalentDate

# Fixtures
from .fixtures_forecaster_equivalent_date import y


def test_set_in_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                 )

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `set_in_sample_residuals()`."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_in_sample_residuals(y=y)


def test_set_in_sample_residuals_TypeError_when_y_is_not_a_Series():
    """
    Test TypeError is raised when y is not a pandas Series.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                 )
    forecaster.fit(y=y)
    
    not_valid_y = np.arange(10)

    err_msg = re.escape(
        f"`y` must be a pandas Series with a DatetimeIndex or a RangeIndex. "
        f"Found {type(not_valid_y)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        forecaster.set_in_sample_residuals(y=not_valid_y)


@pytest.mark.parametrize("diff_index", 
                         [pd.RangeIndex(start=50, stop=100), 
                          pd.date_range(start='1991-07-01', periods=50, freq='MS')], 
                         ids=lambda idx: f'diff_index: {idx[[0, -1]]}')
def test_set_in_sample_residuals_IndexError_when_y_has_different_index_than_training(diff_index):
    """
    Test IndexError is raised when y has different index than training.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                 )
    forecaster.fit(y=y)

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


def test_set_in_sample_residuals_store_same_residuals_as_fit_offset_int():
    """
    Test that set_in_sample_residuals stores same residuals as fit when offset 
    is an int.
    """
    forecaster_1 = ForecasterEquivalentDate(
                       offset        = 2,
                       n_offsets     = 2,
                       agg_func      = np.mean,
                       binner_kwargs = {'n_bins': 3}
                   )
    forecaster_1.fit(y=y, store_in_sample_residuals=True)

    forecaster_2 = ForecasterEquivalentDate(
                       offset        = 2,
                       n_offsets     = 2,
                       agg_func      = np.mean,
                       binner_kwargs = {'n_bins': 3}
                   )
    forecaster_2.fit(y=y, store_in_sample_residuals=False)
    forecaster_2.set_in_sample_residuals(y=y)

    np.testing.assert_almost_equal(forecaster_1.in_sample_residuals_, forecaster_2.in_sample_residuals_)
    for k in forecaster_1.in_sample_residuals_by_bin_.keys():
        np.testing.assert_almost_equal(forecaster_1.in_sample_residuals_by_bin_[k], forecaster_2.in_sample_residuals_by_bin_[k])
    assert forecaster_1.binner_intervals_ == forecaster_2.binner_intervals_


def test_set_in_sample_residuals_store_same_residuals_as_fit_offset_DateOffset():
    """
    Test that set_in_sample_residuals stores same residuals as fit when offset 
    is an DateOffset.
    """
    forecaster_1 = ForecasterEquivalentDate(
                       offset        = DateOffset(days=2),
                       n_offsets     = 2,
                       agg_func      = np.mean,
                       binner_kwargs = {'n_bins': 3}
                   )
    forecaster_1.fit(y=y, store_in_sample_residuals=True)

    forecaster_2 = ForecasterEquivalentDate(
                       offset        = DateOffset(days=2),
                       n_offsets     = 2,
                       agg_func      = np.mean,
                       binner_kwargs = {'n_bins': 3}
                   )
    forecaster_2.fit(y=y, store_in_sample_residuals=False)
    forecaster_2.set_in_sample_residuals(y=y)

    np.testing.assert_almost_equal(forecaster_1.in_sample_residuals_, forecaster_2.in_sample_residuals_)
    for k in forecaster_1.in_sample_residuals_by_bin_.keys():
        np.testing.assert_almost_equal(forecaster_1.in_sample_residuals_by_bin_[k], forecaster_2.in_sample_residuals_by_bin_[k])
    assert forecaster_1.binner_intervals_ == forecaster_2.binner_intervals_
