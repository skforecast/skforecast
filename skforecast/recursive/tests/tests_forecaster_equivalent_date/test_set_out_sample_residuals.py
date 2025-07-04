# Unit test set_out_sample_residuals ForecasterEquivalentDate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.exceptions import NotFittedError
from skforecast.exceptions import ResidualsUsageWarning
from skforecast.recursive import ForecasterEquivalentDate

# Fixtures
from .fixtures_forecaster_equivalent_date import y


def test_set_out_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                 )
    y_true = {1: np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}
    y_pred = {1: np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `set_out_sample_residuals()`."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_true_is_not_numpy_array_or_pandas_series():
    """
    Test TypeError is raised when y_true argument is not numpy ndarray or pandas Series.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                 )
    forecaster.fit(y)
    y_true = 'invalid'
    y_pred = np.array([1, 2, 3])

    err_msg = re.escape(
        f"`y_true` argument must be `numpy ndarray` or `pandas Series`. "
        f"Got {type(y_true)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_pred_is_not_numpy_array_or_pandas_series():
    """
    Test TypeError is raised when y_pred argument is not numpy ndarray or pandas Series.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                 )
    forecaster.fit(y)
    y_true = np.array([1, 2, 3])
    y_pred = 'invalid'

    err_msg = re.escape(
        f"`y_pred` argument must be `numpy ndarray` or `pandas Series`. "
        f"Got {type(y_pred)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_different_length():
    """
    Test ValueError is raised when y_true and y_pred have different length.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                 )
    forecaster.fit(y)
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3, 4])

    err_msg = re.escape(
        f"`y_true` and `y_pred` must have the same length. "
        f"Got {len(y_true)} and {len(y_pred)}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_different_index():
    """
    Test ValueError is raised when residuals and y_pred have different index.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                 )
    forecaster.fit(y)
    y_true = pd.Series([1, 2, 3], index=[1, 2, 3])
    y_pred = pd.Series([1, 2, 3], index=[1, 2, 4])

    err_msg = re.escape("`y_true` and `y_pred` must have the same index.")
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_and_no_append():
    """
    Test residuals stored when new residuals length is less than 10_000 and 
    append is False.
    """
    rng = np.random.default_rng(12345)
    y_true = pd.Series(rng.normal(loc=10, scale=10, size=1000))
    y_pred = pd.Series(rng.normal(loc=10, scale=10, size=1000))

    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                 )
    forecaster.fit(y_true)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=False)
    results = np.sort(forecaster.out_sample_residuals_)

    expected = np.sort(y_true - y_pred)

    np.testing.assert_array_almost_equal(results, expected)


def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_and_append():
    """
    Test residuals stored when new residuals length is less than 10_000 and 
    append is True.
    """
    rng = np.random.default_rng(12345)
    y_true = pd.Series(rng.normal(loc=10, scale=10, size=1000))
    y_true.index = pd.date_range(start='2000-01-01', periods=1000, freq='D')
    y_pred = pd.Series(rng.normal(loc=10, scale=10, size=1000))
    y_pred.index = pd.date_range(start='2000-01-01', periods=1000, freq='D')

    forecaster = ForecasterEquivalentDate(
                     offset        = DateOffset(days=2),
                     n_offsets     = 2,
                     agg_func      = np.mean,
                 )
    forecaster.fit(y_true)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)
    results = np.sort(forecaster.out_sample_residuals_)
    
    residuals = (y_true - y_pred)
    expected = np.sort(np.concatenate((residuals, residuals)))

    np.testing.assert_array_almost_equal(results, expected)


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_10000():
    """
    Test length residuals stored when its length is greater than 10_000.
    """
    rng = np.random.RandomState(42)
    y_fit = pd.Series(rng.normal(loc=10, scale=10, size=50_000))
    y_fit.index = pd.date_range(start='2000-01-01', periods=50_000, freq='D')

    forecaster = ForecasterEquivalentDate(
                     offset        = DateOffset(days=2),
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     binner_kwargs = {"n_bins": 10}
                 )
    forecaster.fit(y_fit)

    y_pred = [
        y_fit.loc[(y_fit.index - forecaster.offset * n_off)[forecaster.window_size:]]
        for n_off in range(1, forecaster.n_offsets + 1)
    ]

    y_pred = np.apply_along_axis(
                 forecaster.agg_func,
                 axis = 0,
                 arr  = np.vstack(y_pred)
             )

    forecaster.set_out_sample_residuals(
        y_true = y_fit.to_numpy()[forecaster.window_size:],
        y_pred = y_pred
    )

    assert len(forecaster.out_sample_residuals_) == 10_000
    for v in forecaster.out_sample_residuals_by_bin_.values():
        assert len(v) == 1_000


def test_out_sample_residuals_by_bin_and_in_sample_residuals_by_bin_equivalence():
    """
    Test out sample residuals by bin are equivalent to in-sample residuals by bin
    when training data and training predictions are passed.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     binner_kwargs = {"n_bins": 3}
                 )
    forecaster.fit(y, store_in_sample_residuals=True)

    y_pred = [
        y.shift(forecaster.offset * n_off)[forecaster.window_size:]
        for n_off in range(1, forecaster.n_offsets + 1)
    ]

    y_pred = np.apply_along_axis(
                 forecaster.agg_func,
                 axis = 0,
                 arr  = np.vstack(y_pred)
             )

    forecaster.set_out_sample_residuals(
        y_true = y.to_numpy()[forecaster.window_size:],
        y_pred = y_pred
    )

    assert forecaster.in_sample_residuals_by_bin_.keys() == forecaster.out_sample_residuals_by_bin_.keys()
    for k in forecaster.out_sample_residuals_by_bin_.keys():
        np.testing.assert_array_almost_equal(
            forecaster.in_sample_residuals_by_bin_[k],
            forecaster.out_sample_residuals_by_bin_[k]
        )


def test_set_out_sample_residuals_append_new_residuals_per_bin():
    """
    Test that set_out_sample_residuals append residuals per bin until it
    reaches the max allowed size of 10_000 // n_bins
    """
    rng = np.random.default_rng(12345)
    y_fit = pd.Series(
                data=rng.normal(loc=10, scale=1, size=1001),
                index=pd.date_range(start="01-01-2000", periods=1001, freq="h"),
            )

    forecaster = ForecasterEquivalentDate(
                     offset        = 1,
                     n_offsets     = 1,
                     agg_func      = np.mean,
                     binner_kwargs = {"n_bins": 2}
                 )
    forecaster.fit(y_fit)

    y_pred = [
        y_fit.shift(forecaster.offset * n_off)[forecaster.window_size:]
        for n_off in range(1, forecaster.n_offsets + 1)
    ]

    y_pred = np.apply_along_axis(
                 forecaster.agg_func,
                 axis = 0,
                 arr  = np.vstack(y_pred)
             )

    for i in range(1, 20):
        forecaster.set_out_sample_residuals(
            y_true = y_fit.to_numpy()[forecaster.window_size:],
            y_pred = y_pred,
            append = True
        )
        for v in forecaster.out_sample_residuals_by_bin_.values():
            assert len(v) == min(5_000, 500 * i)


def test_set_out_sample_residuals_when_there_are_no_residuals_for_some_bins():
    """
    Test that set_out_sample_residuals works when there are no residuals for some bins.
    """
    rng = np.random.default_rng(12345)
    y = pd.Series(
            data=rng.normal(loc=10, scale=1, size=100),
            index=pd.date_range(start="01-01-2000", periods=100, freq="h"),
        )

    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     binner_kwargs = {"n_bins": 3}
                 )
    forecaster.fit(y)
    y_pred = y.loc[y > 10]
    y_true = y_pred + rng.normal(loc=0, scale=1, size=len(y_pred))

    warn_msg = re.escape(
        f"The following bins have no out of sample residuals: [0]. "
        f"No predicted values fall in the interval "
        f"[{forecaster.binner_intervals_[0]}]. "
        f"Empty bins will be filled with a random sample of residuals."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)

    assert len(forecaster.out_sample_residuals_by_bin_[0]) == len(y_pred)
