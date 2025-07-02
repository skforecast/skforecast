# Unit test fit ForecasterEquivalentDate
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from skforecast.recursive import ForecasterEquivalentDate

# Fixtures
from .fixtures_forecaster_equivalent_date import y


def test_fit_TypeError_when_y_is_not_a_Series():
    """
    Test TypeError is raised when y is not a pandas Series.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                 )
    not_valid_y = np.arange(10)

    err_msg = re.escape(
        f"`y` must be a pandas Series with a DatetimeIndex or a RangeIndex. "
        f"Found {type(not_valid_y)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        forecaster.fit(y=not_valid_y)


@pytest.mark.parametrize("y", 
                         [pd.Series(np.arange(10)), 
                          pd.Series(np.arange(10), index=pd.date_range(start='1/1/2021', periods=10))])
def test_fit_TypeError_offset_DateOffset_y_index_not_DatetimeIndex(y):
    """
    Test TypeError is raised when offset is a DateOffset and y index is 
    not a DatetimeIndex or has no freq.
    """
    forecaster = ForecasterEquivalentDate(
        offset=DateOffset(days=1), n_offsets=2, agg_func=np.mean
    )

    if isinstance(y.index, pd.DatetimeIndex):
        y.index.freq = None

    err_msg = re.escape(
        "If `offset` is a pandas DateOffset, the index of `y` must be a "
        "pandas DatetimeIndex with frequency."
    )
    with pytest.raises(TypeError, match=err_msg):
        forecaster.fit(y=y)


def test_fit_ValueError_length_y_less_than_window_size_offset_int():
    """
    Test ValueError is raised when length of y is less than window_size
    when offset is an int.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 6,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                 )
    y = pd.Series(np.arange(10))

    err_msg = re.escape(
        "Length of `y` must be greater than the maximum window size "
        "needed by the forecaster. This is because  "
        "the offset (6) is larger than the available "
        "data. Try to decrease the size of the offset (6), "
        "the number of `n_offsets` (2) or increase the "
        "size of `y`.\n"
        "    Length `y`: 10.\n"
        "    Max window size: 12.\n"
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.fit(y=y)


@pytest.mark.parametrize("offset, y", 
                         [({'days': 6}, 
                           pd.Series(np.arange(10),
                                     index=pd.date_range(start='01/01/2021', periods=10, freq='D'))), 
                          ({'months': 6}, 
                           pd.Series(np.arange(10), 
                                     index=pd.date_range(start='01/01/2021', periods=10, freq='MS')))])
def test_fit_ValueError_length_y_less_than_window_size_offset_DateOffset(offset, y):
    """
    Test ValueError is raised when length of y is less than window_size
    when offset is a pandas DateOffset.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = DateOffset(**offset),
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     forecaster_id = None
                 )

    err_msg = re.escape(
        f"The length of `y` (10), must be greater than or equal "
        f"to the window size ({forecaster.window_size}). This is because  "
        f"the offset ({forecaster.offset}) is larger than the available "
        f"data. Try to decrease the size of the offset ({forecaster.offset}), "
        f"the number of `n_offsets` (2) or increase the "
        f"size of `y`."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.fit(y=y)


def test_fit_y_index_DatetimeIndex():
    """
    Test index_freq_ is set correctly when y index is a DatetimeIndex.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     forecaster_id = None
                 )
    y = pd.Series(
        np.random.rand(10), index=pd.date_range(start='1/1/2021', periods=10)
    )
    forecaster.fit(y)

    assert forecaster.index_freq_ == y.index.freqstr


def test_fit_y_index_not_DatetimeIndex():
    """
    Test index_freq_ is set correctly when y index is not a DatetimeIndex.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     forecaster_id = None
                 )
    y = pd.Series(np.random.rand(10))
    forecaster.fit(y)

    assert forecaster.index_freq_ == y.index.step


def test_fit_offset_int():
    """
    Test window_size and last_window_ are set correctly when offset is an int.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     forecaster_id = None
                 )
    y = pd.Series(np.random.rand(10), index=pd.RangeIndex(start=0, stop=10))
    forecaster.fit(y)

    assert forecaster.window_size == 4.0
    assert forecaster.last_window_.equals(y)


def test_fit_offset_DateOffset():
    """
    Test window_size and last_window_ are set correctly when offset is a DateOffset.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = DateOffset(days=2),
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     forecaster_id = None
                 )
    y = pd.Series(np.random.rand(10), index=pd.date_range(start='01/01/2021', periods=10))
    forecaster.fit(y)

    assert forecaster.window_size == 4.0
    assert forecaster.last_window_.equals(y)


def test_fit_in_sample_residuals_stored():
    """
    Test that values of in_sample_residuals_ are stored after fitting.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 1,
                     n_offsets     = 1,
                     agg_func      = np.mean,
                     forecaster_id = None
                 )
    forecaster.fit(y=pd.Series(np.arange(5)), store_in_sample_residuals=True)
    results = forecaster.in_sample_residuals_
    expected = np.array([1., 1., 1., 1.])

    assert isinstance(results, np.ndarray)
    np.testing.assert_array_almost_equal(results, expected)


def test_fit_same_residuals_when_residuals_greater_than_10000():
    """
    Test fit return same residuals when residuals len is greater than 10_000.
    Testing with two different forecaster.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 3,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     forecaster_id = None
                 )
    forecaster.fit(y=pd.Series(np.arange(12_000)), store_in_sample_residuals=True)
    results_1 = forecaster.in_sample_residuals_
    forecaster = ForecasterEquivalentDate(
                     offset        = 3,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     forecaster_id = None
                 )
    forecaster.fit(y=pd.Series(np.arange(12_000)), store_in_sample_residuals=True)
    results_2 = forecaster.in_sample_residuals_
    
    assert isinstance(results_1, np.ndarray)
    assert isinstance(results_2, np.ndarray)
    assert len(results_1 == 10_000)
    assert len(results_2 == 10_000)
    np.testing.assert_array_almost_equal(results_1, results_2)


def test_fit_in_sample_residuals_by_bin_stored_offset_int():
    """
    Test that values of in_sample_residuals_by_bin are stored after fitting
    when offset is an int.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 2,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     binner_kwargs = {'n_bins': 3},
                     forecaster_id = None
                 )
    forecaster.fit(y, store_in_sample_residuals=True)

    expected_1 = np.array([
        0.25780865,  0.00437941,  0.50760399,  0.19761913, -0.36918468,
       -0.16185058, -0.38767003,  0.19057608,  0.02651728, -0.50090572,
        0.00716913,  0.34363161, -0.23581652, -0.22338489,  0.24128337,
        0.07510401,  0.27737941,  0.49579211,  0.14147915, -0.07960618,
        0.04301524, -0.40726874, -0.36166069, -0.23872798, -0.24840197,
        0.35536505, -0.23564641,  0.00408149,  0.23795327, -0.03865355,
        0.16434644, -0.15143191, -0.00199522,  0.490416  ,  0.51806922,
       -0.10098851, -0.06130272, -0.58199452, -0.46677101,  0.10609867,
        0.39568995, -0.01476693, -0.10876306,  0.652919  , -0.15518659,
       -0.00511305
    ])

    expected_2 = {
        0: np.array([ 0.02651728,  0.00716913,  0.34363161, -0.22338489,  0.24128337,
            0.27737941,  0.49579211,  0.35536505, -0.23564641,  0.23795327,
            0.16434644,  0.490416  ,  0.10609867, -0.01476693,  0.652919  ]),
        1: np.array([ 0.25780865,  0.00437941,  0.50760399,  0.19761913,  0.19057608,
            -0.23581652,  0.07510401, -0.23872798, -0.24840197,  0.00408149,
            -0.03865355, -0.15143191, -0.00199522,  0.51806922,  0.39568995]),
        2: np.array([-0.36918468, -0.16185058, -0.38767003, -0.50090572,  0.14147915,
            -0.07960618,  0.04301524, -0.40726874, -0.36166069, -0.10098851,
            -0.06130272, -0.58199452, -0.46677101, -0.10876306, -0.15518659,
            -0.00511305])
    }

    expected_3 = {
        0: (0.192909495, 0.41830825),
        1: (0.41830825, 0.5539681),
        2: (0.5539681, 0.850116585),
    }

    np.testing.assert_array_almost_equal(
        forecaster.in_sample_residuals_, expected_1
    )
    for k in expected_2.keys():
        np.testing.assert_array_almost_equal(forecaster.in_sample_residuals_by_bin_[k], expected_2[k])
    for k in expected_3.keys():
        assert forecaster.binner_intervals_[k][0] == approx(expected_3[k][0])
        assert forecaster.binner_intervals_[k][1] == approx(expected_3[k][1])


def test_fit_in_sample_residuals_by_bin_stored_offset_DateOffset():
    """
    Test that values of in_sample_residuals_by_bin are stored after fitting
    when offset is a DateOffset.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = DateOffset(days=2),
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     binner_kwargs = {'n_bins': 3},
                     forecaster_id = None
                 )
    forecaster.fit(y, store_in_sample_residuals=True)

    expected_1 = np.array([
        0.25780865,  0.00437941,  0.50760399,  0.19761913, -0.36918468,
       -0.16185058, -0.38767003,  0.19057608,  0.02651728, -0.50090572,
        0.00716913,  0.34363161, -0.23581652, -0.22338489,  0.24128337,
        0.07510401,  0.27737941,  0.49579211,  0.14147915, -0.07960618,
        0.04301524, -0.40726874, -0.36166069, -0.23872798, -0.24840197,
        0.35536505, -0.23564641,  0.00408149,  0.23795327, -0.03865355,
        0.16434644, -0.15143191, -0.00199522,  0.490416  ,  0.51806922,
       -0.10098851, -0.06130272, -0.58199452, -0.46677101,  0.10609867,
        0.39568995, -0.01476693, -0.10876306,  0.652919  , -0.15518659,
       -0.00511305
    ])

    expected_2 = {
        0: np.array([ 0.02651728,  0.00716913,  0.34363161, -0.22338489,  0.24128337,
            0.27737941,  0.49579211,  0.35536505, -0.23564641,  0.23795327,
            0.16434644,  0.490416  ,  0.10609867, -0.01476693,  0.652919  ]),
        1: np.array([ 0.25780865,  0.00437941,  0.50760399,  0.19761913,  0.19057608,
            -0.23581652,  0.07510401, -0.23872798, -0.24840197,  0.00408149,
            -0.03865355, -0.15143191, -0.00199522,  0.51806922,  0.39568995]),
        2: np.array([-0.36918468, -0.16185058, -0.38767003, -0.50090572,  0.14147915,
            -0.07960618,  0.04301524, -0.40726874, -0.36166069, -0.10098851,
            -0.06130272, -0.58199452, -0.46677101, -0.10876306, -0.15518659,
            -0.00511305])
    }

    expected_3 = {
        0: (0.192909495, 0.41830825),
        1: (0.41830825, 0.5539681),
        2: (0.5539681, 0.850116585),
    }

    np.testing.assert_array_almost_equal(
        forecaster.in_sample_residuals_, expected_1
    )
    for k in expected_2.keys():
        np.testing.assert_array_almost_equal(forecaster.in_sample_residuals_by_bin_[k], expected_2[k])
    for k in expected_3.keys():
        assert forecaster.binner_intervals_[k][0] == approx(expected_3[k][0])
        assert forecaster.binner_intervals_[k][1] == approx(expected_3[k][1])


def test_fit_in_sample_residuals_not_stored_probabilistic_mode_binned():
    """
    Test that values of in_sample_residuals_ are not stored after fitting
    when `store_in_sample_residuals=False`. Binner intervals are stored.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = DateOffset(days=2),
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     binner_kwargs = {'n_bins': 3},
                     forecaster_id = None
                 )
    forecaster.fit(y, store_in_sample_residuals=False)

    expected_binner_intervals_ = {
        0: (0.192909495, 0.41830825),
        1: (0.41830825, 0.5539681),
        2: (0.5539681, 0.850116585),
    }

    assert forecaster.in_sample_residuals_ is None
    assert forecaster.in_sample_residuals_by_bin_ is None
    assert forecaster.binner_intervals_.keys() == expected_binner_intervals_.keys()
    for k in expected_binner_intervals_.keys():
        assert forecaster.binner_intervals_[k][0] == approx(expected_binner_intervals_[k][0])
        assert forecaster.binner_intervals_[k][1] == approx(expected_binner_intervals_[k][1])


def test_fit_in_sample_residuals_not_stored_probabilistic_mode_False():
    """
    Test that values of in_sample_residuals_ are not stored after fitting
    when `store_in_sample_residuals=False` and _probabilistic_mode=False.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = DateOffset(days=2),
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     binner_kwargs = {'n_bins': 3},
                     forecaster_id = None
                 )
    forecaster._probabilistic_mode = False
    forecaster.fit(y=y, store_in_sample_residuals=False)

    assert forecaster.in_sample_residuals_ is None
    assert forecaster.in_sample_residuals_by_bin_ is None
    assert forecaster.binner_intervals_ is None
