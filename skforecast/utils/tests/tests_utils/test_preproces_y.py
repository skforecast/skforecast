# Unit test preprocess_y
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import preprocess_y


def test_output_preprocess_y_when_y_index_is_DatetimeIndex_and_has_frequency():
    """
    Test values returned by when y is a pandas Series DatetimeIndex and freq is
    not None.
    """
    y = pd.Series(
            data = np.arange(3),
            index = pd.date_range("1990-01-01", periods=3, freq='D')
        )
    results = preprocess_y(y)
    expected = (np.arange(3),
                pd.DatetimeIndex(['1990-01-01', '1990-01-02', '1990-01-03'],
                                 dtype='datetime64[ns]', freq='D')
               )
    
    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()


def test_output_preprocess_y_when_y_index_is_RangeIndex():
    """
    Test values returned by when y is a pandas Series with RangeIndex
    """
    y = pd.Series(
            data = np.arange(3),
            index = pd.RangeIndex(start=0, stop=3, step=1)
        )
    results = preprocess_y(y)
    expected = (np.arange(3),
                pd.RangeIndex(start=0, stop=3, step=1)
               )
    
    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()


def test_output_preprocess_y_when_y_index_is_DatetimeIndex_but_has_not_frequency():
    """
    Test values returned by when y is a pandas Series with DatetimeIndex but freq 
    is None.
    """
    y = pd.Series(
            data = np.arange(3),
            index = pd.to_datetime(["1990-01-01", "1990-01-02", "1990-01-03"])
        )
    results = preprocess_y(y)
    expected = (np.arange(3),
                pd.RangeIndex(start=0, stop=3, step=1)
               )
    
    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()
    
    
def test_output_preprocess_y_when_y_index_is_not_DatetimeIndex_or_RangeIndex():
    """
    Test values returned by when y is a pandas Series without DatetimeIndex or RangeIndex.
    """
    y = pd.Series(data=np.arange(3), index=['0', '1', '2'])
    results = preprocess_y(y)
    expected = (np.arange(3),
                pd.RangeIndex(start=0, stop=3, step=1)
               )
    
    assert (results[0] == expected[0]).all()
    assert (results[1] == expected[1]).all()