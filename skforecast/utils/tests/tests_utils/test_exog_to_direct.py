# Unit test exog_to_direct
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import exog_to_direct
from pandas.api.types import CategoricalDtype


def test_exog_to_direct_TypeError_when_exog_not_pandas_series_or_dataframe():
    """
    Test TypeError is raised when exog is not a pandas Series or DataFrame.
    """
    exog = np.arange(7)

    err_msg = re.escape(f"`exog` must be a pandas Series or DataFrame. Got {type(exog)}.")
    with pytest.raises(TypeError, match = err_msg):
        exog_to_direct(exog=exog, steps=2)


@pytest.mark.parametrize("exog                                               , dtype", 
                         [(pd.Series(np.arange(10), name='exog', dtype=int)  , int),
                          (pd.Series(np.arange(10), name='exog', dtype=float), float),
                          (pd.DataFrame(pd.Series(np.arange(10), dtype=int)  , columns=['exog']), int),
                          (pd.DataFrame(pd.Series(np.arange(10), dtype=float), columns=['exog']), float)], 
                         ids = lambda exog: f'exog: {exog}')
def test_exog_to_direct_when_steps_2_exog_1d_float_int(exog, dtype):
    """
    Test exog_to_direct results when using steps 2 and exog is a  
    1d pandas Series or 1 column DataFrame of ints or floats.
    """
    results = exog_to_direct(exog=exog, steps=2)
    expected = (
        pd.DataFrame(
            data = np.array([[0, 1],
                            [1, 2],
                            [2, 3],
                            [3, 4],
                            [4, 5],
                            [5, 6],
                            [6, 7],
                            [7, 8],
                            [8, 9]]),
            index   = pd.RangeIndex(start=1, stop=10, step=1),
            columns = ['exog_step_1', 'exog_step_2']
        ).astype({'exog_step_1': dtype, 'exog_step_2': dtype}
        ),
        ['exog_step_1', 'exog_step_2']
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]


@pytest.mark.parametrize("exog                                              , dtype", 
                         [(pd.Series(np.arange(5), name='exog', dtype=int)  , int),
                          (pd.Series(np.arange(5), name='exog', dtype=float), float),
                          (pd.DataFrame(pd.Series(np.arange(5), dtype=int)  , columns=['exog']), int),
                          (pd.DataFrame(pd.Series(np.arange(5), dtype=float), columns=['exog']), float)], 
                         ids = lambda exog: f'exog: {exog}')
def test_exog_to_direct_when_steps_series_len_exog_1d_float_int(exog, dtype):
    """
    Test exog_to_direct results when using steps equal to series len (5) and 
    exog is a 1d pandas Series or 1 column DataFrame of ints or floats.
    """
    results = exog_to_direct(exog=exog, steps=len(exog))
    expected = (
        pd.DataFrame(
            data    = np.array([[0, 1, 2, 3, 4]]),
            index   = pd.RangeIndex(start=4, stop=5, step=1),
            columns = ['exog_step_1', 'exog_step_2', 'exog_step_3',
                       'exog_step_4', 'exog_step_5']
        ).astype({'exog_step_1': dtype, 'exog_step_2': dtype,
                  'exog_step_3': dtype, 'exog_step_4': dtype,
                  'exog_step_5': dtype}
        ),
        ['exog_step_1', 'exog_step_2', 'exog_step_3', 'exog_step_4', 'exog_step_5']
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_exog_to_direct_when_steps_2_exog_DataFrame_2d_float_int(dtype):
    """
    Test exog_to_direct results when using steps 2 and exog is a  
    pandas DataFrame with 2 columns of floats or ints.
    """
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})
    results = exog_to_direct(exog=exog, steps=2)
    expected = (
        pd.DataFrame(
            data = np.array([[100, 1000, 101, 1001],
                            [101, 1001, 102, 1002],
                            [102, 1002, 103, 1003],
                            [103, 1003, 104, 1004],
                            [104, 1004, 105, 1005],
                            [105, 1005, 106, 1006],
                            [106, 1006, 107, 1007],
                            [107, 1007, 108, 1008],
                            [108, 1008, 109, 1009]]),
            index   = pd.RangeIndex(start=1, stop=10, step=1),
            columns = ['exog_1_step_1', 'exog_2_step_1', 
                       'exog_1_step_2', 'exog_2_step_2']
        ).astype({'exog_1_step_1': dtype, 'exog_2_step_1': dtype,
                  'exog_1_step_2': dtype, 'exog_2_step_2': dtype}
        ),
        ['exog_1_step_1', 'exog_2_step_1', 'exog_1_step_2', 'exog_2_step_2']
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]


def test_exog_to_direct_when_steps_2_exog_1d_category():
    """
    Test exog_to_direct results when using steps 2 and exog is a  
    1d pandas Series or 1 column DataFrame of category.
    """
    exog = pd.Series(range(10), name='exog', dtype='category')
    results = exog_to_direct(exog=exog, steps=2)
    expected = (
        pd.DataFrame(
            {'exog_step_1': pd.Categorical(range(9), categories=range(10)),
             'exog_step_2': pd.Categorical(range(1, 10), categories=range(10))},
            index = pd.RangeIndex(start=1, stop=10, step=1)
        ),
        ['exog_step_1', 'exog_step_2']
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]


def test_exog_to_direct_when_steps_2_exog_DataFrame_2d_category():
    """
    Test exog_to_direct results when using steps 2 and exog is a  
    pandas DataFrame with 2 columns of category.
    """
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})
    results = exog_to_direct(exog=exog, steps=2)
    expected = (
        pd.DataFrame({
            'exog_1_step_1': pd.Categorical(range(9), categories=range(10)),
            'exog_2_step_1': pd.Categorical(range(100, 109), categories=range(100, 110)),
            'exog_1_step_2': pd.Categorical(range(1, 10), categories=range(10)),
            'exog_2_step_2': pd.Categorical(range(101, 110), categories=range(100, 110))},
            index = pd.RangeIndex(start=1, stop=10, step=1)
        ),
        ['exog_1_step_1', 'exog_2_step_1', 
         'exog_1_step_2', 'exog_2_step_2']
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]


def test_exog_to_direct_when_steps_2_exog_DataFrame_3d_float_int_category():
    """
    Test exog_to_direct results when using steps 2 and exog is a  
    pandas DataFrame with 3 columns of float, int, category.
    """
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype='float64'),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype='int64'),
                         'exog_3': pd.Categorical(range(100, 110))})
    results = exog_to_direct(exog=exog, steps=2)
    expected = (
        pd.DataFrame(
            data = np.array([[100., 1000., 100., 101., 1001., 101.],
                            [101., 1001., 101., 102., 1002., 102.],
                            [102., 1002., 102., 103., 1003., 103.],
                            [103., 1003., 103., 104., 1004., 104.],
                            [104., 1004., 104., 105., 1005., 105.],
                            [105., 1005., 105., 106., 1006., 106.],
                            [106., 1006., 106., 107., 1007., 107.],
                            [107., 1007., 107., 108., 1008., 108.],
                            [108., 1008., 108., 109., 1009., 109.]]),
            index   = pd.RangeIndex(start=1, stop=10, step=1),
            columns = ['exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 
                       'exog_1_step_2', 'exog_2_step_2', 'exog_3_step_2']
        ).astype(
            {'exog_1_step_1': 'float64',
             'exog_2_step_1': 'int64',
             'exog_3_step_1': 'int64',
             'exog_1_step_2': 'float64',
             'exog_2_step_2': 'int64',
             'exog_3_step_2': 'int64'}
        ).astype(
            {'exog_3_step_1': CategoricalDtype(categories=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109], ordered=False),
             'exog_3_step_2': CategoricalDtype(categories=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109], ordered=False)}
        ),
        ['exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 
         'exog_1_step_2', 'exog_2_step_2', 'exog_3_step_2']
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]


def test_exog_to_direct_when_steps_len_exog_exog_DataFrame_3d_float_int_category():
    """
    Test exog_to_direct results when using steps len exog (5) and exog is a  
    pandas DataFrame with 3 columns of float, int, category.
    """
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 103), dtype='float64'),
                         'exog_2': pd.Series(np.arange(1000, 1003), dtype='int64'),
                         'exog_3': pd.Categorical(range(100, 103))})
    results = exog_to_direct(exog=exog, steps=len(exog))
    expected = (
        pd.DataFrame(
            data = np.array([[100., 1000., 100., 101., 1001., 101., 102., 1002., 102.]]),
            index   = pd.RangeIndex(start=2, stop=3, step=1),
            columns = ['exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 
                       'exog_1_step_2', 'exog_2_step_2', 'exog_3_step_2', 
                       'exog_1_step_3', 'exog_2_step_3', 'exog_3_step_3']
        ).astype(
            {'exog_1_step_1': 'float64',
             'exog_2_step_1': 'int64',
             'exog_3_step_1': 'int64',
             'exog_1_step_2': 'float64',
             'exog_2_step_2': 'int64',
             'exog_3_step_2': 'int64',
             'exog_1_step_3': 'float64',
             'exog_2_step_3': 'int64',
             'exog_3_step_3': 'int64'}
        ).astype(
            {'exog_3_step_1': CategoricalDtype(categories=[100, 101, 102], ordered=False),
             'exog_3_step_2': CategoricalDtype(categories=[100, 101, 102], ordered=False),
             'exog_3_step_3': CategoricalDtype(categories=[100, 101, 102], ordered=False)}
        ),
        ['exog_1_step_1', 'exog_2_step_1', 'exog_3_step_1', 
         'exog_1_step_2', 'exog_2_step_2', 'exog_3_step_2', 
         'exog_1_step_3', 'exog_2_step_3', 'exog_3_step_3']
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt: f'dtype: {dt}')
def test_exog_to_direct_when_steps_3_exog_DataFrame_2d_float_int(dtype):
    """
    Test exog_to_direct results when using steps 3 and exog is a  
    pandas DataFrame with 2 columns of floats or ints.
    """
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})
    results = exog_to_direct(exog=exog, steps=3)
    expected = (
        pd.DataFrame(
            data = np.array([[100, 1000, 101, 1001, 102, 1002],
                            [101, 1001, 102, 1002, 103, 1003],
                            [102, 1002, 103, 1003, 104, 1004],
                            [103, 1003, 104, 1004, 105, 1005],
                            [104, 1004, 105, 1005, 106, 1006],
                            [105, 1005, 106, 1006, 107, 1007],
                            [106, 1006, 107, 1007, 108, 1008],
                            [107, 1007, 108, 1008, 109, 1009]]),
            index   = pd.RangeIndex(start=2, stop=10, step=1),
            columns = ['exog_1_step_1', 'exog_2_step_1', 
                       'exog_1_step_2', 'exog_2_step_2', 
                       'exog_1_step_3', 'exog_2_step_3']
        ).astype({'exog_1_step_1': dtype, 'exog_2_step_1': dtype, 
                  'exog_1_step_2': dtype, 'exog_2_step_2': dtype, 
                  'exog_1_step_3': dtype, 'exog_2_step_3': dtype}
        ),
        ['exog_1_step_1', 'exog_2_step_1', 
         'exog_1_step_2', 'exog_2_step_2', 
         'exog_1_step_3', 'exog_2_step_3']
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    assert results[1] == expected[1]
