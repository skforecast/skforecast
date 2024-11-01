# Unit test expand_index
# ==============================================================================
import re
import pytest
import pandas as pd
from skforecast.utils import expand_index


def test_TypeError_expand_index_when_steps_is_not_int():
    """
    Test TypeError is raised when `steps` is not a int.
    """
    index = pd.date_range(start='1990-01-01', periods=3, freq='D')
    steps = 2.5
    
    err_msg = re.escape(f"`steps` must be an integer. Got {type(steps)}.")
    with pytest.raises(TypeError, match=err_msg):
        expand_index(index, steps=steps)


def test_TypeError_expand_index_when_index_is_no_pandas_DatetimeIndex_or_RangeIndex():
    """
    Test TypeError is raised when input is not a pandas DatetimeIndex or RangeIndex.
    """
    index = pd.Index([0, 1, 2])

    err_msg = "Argument `index` must be a pandas DatetimeIndex or RangeIndex."
    with pytest.raises(TypeError, match = err_msg):
        expand_index(index, steps=3)


def test_output_expand_index_when_index_is_DatetimeIndex():
    """
    Test values returned by expand_index when input is DatetimeIndex.
    """
    index = pd.DatetimeIndex(['1990-01-01', '1990-01-02', '1990-01-03'],
                             dtype='datetime64[ns]', freq='D')
    expected = pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06'],
                             dtype='datetime64[ns]', freq='D')
    results = expand_index(index, steps=3)
    
    pd.testing.assert_index_equal(results, expected)


def test_output_expand_index_when_index_is_RangeIndex():
    """
    Test values returned by expand_index when input is RangeIndex.
    """
    index = pd.RangeIndex(start=0, stop=3, step=1)
    expected = pd.RangeIndex(start=3, stop=6, step=1)
    results  = expand_index(index, steps=3)
    
    pd.testing.assert_index_equal(results, expected)


def test_output_expand_index_when_index_is_not_pandas_index():
    """
    Test values returned by expand_index when input is not a pandas index.
    """
    index = ['0', '1', '2']
    expected = pd.RangeIndex(start=0, stop=3, step=1)
    results = expand_index(index, steps=3)
    
    pd.testing.assert_index_equal(results, expected)
