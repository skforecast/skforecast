# Unit test fit ForecasterEquivalentDate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from skforecast.recursive import ForecasterEquivalentDate


@pytest.mark.parametrize("y", 
                         [pd.Series(np.arange(10)), 
                          pd.Series(np.arange(10), 
                                    index=pd.date_range(start='1/1/2021', periods=10))])
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
        ("If `offset` is a pandas DateOffset, the index of `y` must be a "
         "pandas DatetimeIndex with frequency.")
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
        (f"The length of `y` (10), must be greater than or equal "
         f"to the window size (12). This is because  "
         f"the offset (6) is larger than the available "
         f"data. Try to decrease the size of the offset (6), "
         f"the number of n_offsets (2) or increase the "
         f"size of `y`.")
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.fit(y=y)


@pytest.mark.parametrize("offset, y", 
                         [({'days': 6}  , 
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
        (f"The length of `y` (10), must be greater than or equal "
         f"to the window size ({forecaster.window_size}). This is because  "
         f"the offset ({forecaster.offset}) is larger than the available "
         f"data. Try to decrease the size of the offset ({forecaster.offset}), "
         f"the number of n_offsets (2) or increase the "
         f"size of `y`.")
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.fit(y=y)


def test_fit_y_index_DatetimeIndex():
    """
    Test index_freq_ is set correctly when y index is a DatetimeIndex.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 5,
                     n_offsets     = 2,
                     agg_func      = np.mean,
                     forecaster_id = None
                 )
    y = pd.Series(np.random.rand(10), index=pd.date_range(start='1/1/2021', periods=10))
    forecaster.fit(y)

    assert forecaster.index_freq_ == y.index.freqstr


def test_fit_y_index_not_DatetimeIndex():
    """
    Test index_freq_ is set correctly when y index is not a DatetimeIndex.
    """
    forecaster = ForecasterEquivalentDate(
                     offset        = 5,
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
    y = pd.Series(np.random.rand(10), 
                  index=pd.date_range(start='01/01/2021', periods=10))
    forecaster.fit(y)

    assert forecaster.window_size == 4.0
    assert forecaster.last_window_.equals(y)