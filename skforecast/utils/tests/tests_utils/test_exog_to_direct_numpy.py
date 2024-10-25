# Unit test exog_to_direct_numpy
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import exog_to_direct_numpy


def test_exog_to_direct_TypeError_when_exog_not_numpy_ndarray_pandas_series_or_dataframe():
    """
    Test TypeError is raised when exog is not a numpy ndarray or pandas
    Series or DataFrame.
    """
    exog = 'not_valid_type'

    err_msg = re.escape(
        f"`exog` must be a numpy ndarray, pandas Series or DataFrame. "
        f"Got {type(exog)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        exog_to_direct_numpy(exog=exog, steps=2)


def test_exog_to_direct_numpy_when_steps_1_exog_numpy_array_1d():
    """
    Test exog_to_direct_numpy results when using steps 1 and exog is a  
    1d numpy array.
    """
    exog = np.arange(10)
    results = exog_to_direct_numpy(exog=exog, steps=1)
    expected = (
        np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]),
        None
    )

    np.testing.assert_array_equal(results[0], expected[0])
    assert results[1] == expected[1]


def test_exog_to_direct_numpy_when_steps_2_exog_numpy_array_1d():
    """
    Test exog_to_direct_numpy results when using steps 2 and exog is a  
    1d numpy array.
    """
    exog = np.arange(10)
    results = exog_to_direct_numpy(exog=exog, steps=2)
    expected = (
        np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9]]),
        None
    )

    np.testing.assert_array_equal(results[0], expected[0])
    assert results[1] == expected[1]


def test_exog_to_direct_numpy_when_steps_len_exog_numpy_array_1d():
    """
    Test exog_to_direct_numpy results when using steps len_exog (5) and exog 
    is a 1d numpy array.
    """
    exog = np.arange(5)
    results = exog_to_direct_numpy(exog=exog, steps=len(exog))
    expected = (
        np.array([[0, 1, 2, 3, 4]]),
        None
    )

    np.testing.assert_array_equal(results[0], expected[0])
    assert results[1] == expected[1]


def test_exog_to_direct_numpy_when_steps_1_exog_numpy_array_2d():
    """
    Test exog_to_direct_numpy results when using steps 1 and exog is a  
    2d numpy array.
    """
    exog = np.column_stack([np.arange(100, 110), np.arange(1000, 1010)])
    results = exog_to_direct_numpy(exog=exog, steps=1)
    expected = (
        np.array([[100, 1000],
                  [101, 1001],
                  [102, 1002],
                  [103, 1003],
                  [104, 1004],
                  [105, 1005],
                  [106, 1006],
                  [107, 1007],
                  [108, 1008],
                  [109, 1009]]),
        None
    )

    np.testing.assert_array_equal(results[0], expected[0])
    assert results[1] == expected[1]


def test_exog_to_direct_numpy_when_steps_2_exog_numpy_array_2d():
    """
    Test exog_to_direct_numpy results when using steps 2 and exog is a  
    2d numpy array.
    """
    exog = np.column_stack([np.arange(100, 110), np.arange(1000, 1010)])
    results = exog_to_direct_numpy(exog=exog, steps=2)
    expected = (
        np.array([[100, 1000, 101, 1001],
                  [101, 1001, 102, 1002],
                  [102, 1002, 103, 1003],
                  [103, 1003, 104, 1004],
                  [104, 1004, 105, 1005],
                  [105, 1005, 106, 1006],
                  [106, 1006, 107, 1007],
                  [107, 1007, 108, 1008],
                  [108, 1008, 109, 1009]]),
        None
    )

    np.testing.assert_array_equal(results[0], expected[0])
    assert results[1] == expected[1]


def test_exog_to_direct_numpy_when_steps_3_exog_numpy_array_2d():
    """
    Test exog_to_direct_numpy results when using steps 3 and exog is a  
    2d numpy array.
    """
    exog = np.column_stack([np.arange(100, 110), np.arange(1000, 1010)])
    results = exog_to_direct_numpy(exog=exog, steps=3)
    expected = (
        np.array([[100, 1000, 101, 1001, 102, 1002],
                  [101, 1001, 102, 1002, 103, 1003],
                  [102, 1002, 103, 1003, 104, 1004],
                  [103, 1003, 104, 1004, 105, 1005],
                  [104, 1004, 105, 1005, 106, 1006],
                  [105, 1005, 106, 1006, 107, 1007],
                  [106, 1006, 107, 1007, 108, 1008],
                  [107, 1007, 108, 1008, 109, 1009]]),
        None
    )

    np.testing.assert_array_equal(results[0], expected[0])
    assert results[1] == expected[1]


def test_exog_to_direct_numpy_when_steps_len_exog_numpy_array_2d():
    """
    Test exog_to_direct_numpy results when using steps len exog (5) and 
    exog is a 2d numpy array.
    """
    exog = np.column_stack([np.arange(100, 105), np.arange(1000, 1005)])
    results = exog_to_direct_numpy(exog=exog, steps=len(exog))
    expected = (
        np.array([[100, 1000, 101, 1001, 102, 1002, 
                   103, 1003, 104, 1004]]),
        None
    )

    np.testing.assert_array_equal(results[0], expected[0])
    assert results[1] == expected[1]


@pytest.mark.parametrize("exog", 
                         [pd.Series(np.arange(5), name='exog', dtype=float),
                          pd.DataFrame(pd.Series(np.arange(5), dtype=float), columns=['exog'])], 
                         ids = lambda exog: f'exog: {exog}')
def test_exog_to_direct_numpy_when_steps_series_len_exog_1d_float_int(exog):
    """
    Test exog_to_direct_numpy results when using steps equal to series len (5) and 
    exog is a 1d pandas Series or 1 column DataFrame of ints or floats.
    """
    results = exog_to_direct_numpy(exog=exog, steps=len(exog))
    expected = (
        np.array([[0., 1., 2., 3., 4.]]),
        ['exog_step_1', 'exog_step_2', 'exog_step_3', 'exog_step_4', 'exog_step_5']
    )

    np.testing.assert_array_equal(results[0], expected[0])
    assert results[1] == expected[1]


@pytest.mark.parametrize("exog", 
                         [pd.Series(np.arange(10), name='exog', dtype=float),
                          pd.DataFrame(pd.Series(np.arange(10), dtype=float), columns=['exog'])], 
                         ids = lambda exog: f'exog: {exog}')
def test_exog_to_direct_numpy_when_steps_2_exog_1d_pandas(exog):
    """
    Test exog_to_direct_numpy results when using steps 2 and exog is a
    pandas Series or 1 column DataFrame.
    """
    results = exog_to_direct_numpy(exog=exog, steps=2)
    expected = (
        np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9]]),
        ['exog_step_1', 'exog_step_2']
    )

    np.testing.assert_array_equal(results[0], expected[0])
    assert results[1] == expected[1]


def test_exog_to_direct_numpy_when_steps_2_exog_pandas_DataFrame():
    """
    Test exog_to_direct_numpy results when using steps 2 and exog is a
    pandas DataFrame.
    """
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=float),
                         'exog_2': np.arange(1000, 1010, dtype=float)})
    results = exog_to_direct_numpy(exog=exog, steps=2)
    expected = (
        np.array([[100., 1000., 101., 1001.],
                  [101., 1001., 102., 1002.],
                  [102., 1002., 103., 1003.],
                  [103., 1003., 104., 1004.],
                  [104., 1004., 105., 1005.],
                  [105., 1005., 106., 1006.],
                  [106., 1006., 107., 1007.],
                  [107., 1007., 108., 1008.],
                  [108., 1008., 109., 1009.]]),
        ['exog_1_step_1', 'exog_2_step_1', 'exog_1_step_2', 'exog_2_step_2']
    )

    np.testing.assert_array_equal(results[0], expected[0])
    assert results[1] == expected[1]


def test_exog_to_direct_numpy_when_steps_2_exog_pandas_DataFrame_3_columns():
    """
    Test exog_to_direct_numpy results when using steps 2 and exog is a
    pandas DataFrame with 3 columns.
    """
    exog = pd.DataFrame(
        {'exog_1': np.arange(100, 110, dtype=float),
         'exog_2': np.arange(1000, 1010, dtype=float),
         'exog_3': np.arange(10000, 10010, dtype=float)}
    )
    results = exog_to_direct_numpy(exog=exog, steps=2)
    expected = (
        np.array([[100., 1000., 10000., 101., 1001., 10001.],
                  [101., 1001., 10001., 102., 1002., 10002.],
                  [102., 1002., 10002., 103., 1003., 10003.],
                  [103., 1003., 10003., 104., 1004., 10004.],
                  [104., 1004., 10004., 105., 1005., 10005.],
                  [105., 1005., 10005., 106., 1006., 10006.],
                  [106., 1006., 10006., 107., 1007., 10007.],
                  [107., 1007., 10007., 108., 1008., 10008.],
                  [108., 1008., 10008., 109., 1009., 10009.]]),
        ['exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 
         'exog_1_step_2', 'exog_2_step_2', 'exog_3_step_2']
    )

    np.testing.assert_array_equal(results[0], expected[0])
    assert results[1] == expected[1]
