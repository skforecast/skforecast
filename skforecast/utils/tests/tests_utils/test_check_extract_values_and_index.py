# Unit test check_extract_values_and_index
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_extract_values_and_index
from skforecast.utils._pandas_compat import PANDAS_GE_3


def test_check_extract_values_and_index_ValueError_when_DatetimeIndex_without_frequency():
    """
    Check ValueError is raised when the index of data is a pandas DatetimeIndex
    without a frequency.
    """

    data = pd.Series([1, 2, 3], name='exog') 
    data.index = pd.date_range("2020-01-01", periods=3, freq=None)
    data.index.freq = None  # Remove frequency to simulate the error

    err_msg = re.escape(
        "`exog` has a pandas DatetimeIndex without a frequency. "
        "To avoid this error, set the frequency of the DatetimeIndex."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_extract_values_and_index(data=data, data_label='`exog`')


@pytest.mark.parametrize("tz", [None, "Europe/Madrid"], ids=lambda tz: f"tz: {tz}")
def test_check_extract_values_and_index_ValueError_message_when_DatetimeIndex_without_frequency_is_tz_aware(tz):
    """
    Check the error message explains how to keep the frequency of a
    timezone-aware index with pandas 3 (its `tz_convert` drops 'D'), and
    only then.
    """

    if tz is None:
        index = pd.date_range("2021-03-20", periods=15, freq="D")
    else:
        index = pd.date_range("2021-03-20", periods=15, freq="D", tz="UTC").tz_convert(tz)
    data = pd.Series(np.arange(15.0), index=pd.DatetimeIndex(index.to_list()), name='y')

    with pytest.raises(ValueError, match="without a frequency") as exc_info:
        check_extract_values_and_index(data=data, data_label='`y`')
    assert ("freq='24h'" in str(exc_info.value)) == (tz is not None and PANDAS_GE_3)


def test_check_extract_values_and_index_TypeError_when_no_DatetimeIndex_or_RangeIndex():
    """
    Check TypeError is raised when the index of data is neither a pandas
    DatetimeIndex nor a RangeIndex.
    """

    data = pd.Series([1, 2, 3], name='last_window') 
    data.index = pd.Index(['a', 'b', 'c'])  # Non-DatetimeIndex or RangeIndex

    err_msg = re.escape(
        f"`last_window` has an unsupported index type. The index must be a "
        f"pandas DatetimeIndex or a RangeIndex. Got {type(data.index)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_extract_values_and_index(data=data, data_label='`last_window`')


@pytest.mark.parametrize("idx", 
                         [pd.date_range("2020-01-01", periods=3, freq='D'),
                          pd.RangeIndex(start=0, stop=3)], 
                         ids = lambda idx: f'index type: {type(idx)}')
def test_check_extract_values_and_index_output(idx):
    """
    Test check_extract_values_and_index returns the correct values and index
    when the index is a pandas DatetimeIndex or a RangeIndex.
    """
    data = pd.Series([1, 2, 3], name='exog', index=idx)
    data_values, data_index = check_extract_values_and_index(data=data, data_label='`exog`')
    
    np.testing.assert_array_almost_equal(data_values, np.array([1, 2, 3]))
    pd.testing.assert_index_equal(data_index, idx)
