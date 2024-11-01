# Unit test preprocess_last_window
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import preprocess_last_window


def test_output_preprocess_last_window_when_last_window_index_is_DatetimeIndex_and_has_frequency():
    """
    Test values returned by when last_window is a pandas Series with DatetimeIndex
    and freq is not None.
    """
    last_window = pd.Series(
            data = np.arange(3),
            index = pd.date_range("1990-01-01", periods=3, freq='D')
        )
    results = preprocess_last_window(last_window)
    expected = (np.arange(3),
                pd.DatetimeIndex(['1990-01-01', '1990-01-02', '1990-01-03'],
                                 dtype='datetime64[ns]', freq='D')
               )
    
    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()


def test_output_preprocess_last_window_when_last_window_index_is_RangeIndex():
    """
    Test values returned by when last_window is a pandas Series with RangeIndex
    """
    last_window = pd.Series(
            data = np.arange(3),
            index = pd.RangeIndex(start=0, stop=3, step=1)
        )
    results = preprocess_last_window(last_window)
    expected = (np.arange(3),
                pd.RangeIndex(start=0, stop=3, step=1)
               )
    
    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()


def test_output_preprocess_last_window_when_last_window_index_is_DatetimeIndex_but_has_not_frequency():
    """
    Test values returned by when last_window is a pandas Series with DatetimeIndex
    and freq is None.
    """
    last_window = pd.Series(
            data = np.arange(3),
            index = pd.to_datetime(["1990-01-01", "1990-01-02", "1990-01-03"])
        )
    results = preprocess_last_window(last_window)
    expected = (np.arange(3),
                pd.RangeIndex(start=0, stop=3, step=1)
               )
    
    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()
    
    
def test_output_preprocess_last_window_when_last_window_index_is_not_DatetimeIndex_or_RangeIndex():
    """
    Test values returned by when last_window is a pandas Series without DatetimeIndex or RangeIndex.
    """
    last_window = pd.Series(data=np.arange(3), index=['0', '1', '2'])
    results = preprocess_last_window(last_window)
    expected = (np.arange(3),
                pd.RangeIndex(start=0, stop=3, step=1)
               )
    
    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()