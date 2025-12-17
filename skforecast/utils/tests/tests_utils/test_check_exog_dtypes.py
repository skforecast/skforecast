# Unit test check_exog_dtypes
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_exog_dtypes
from skforecast.exceptions import DataTypeWarning


@pytest.mark.parametrize("exog", 
                         [pd.Series(['A', 'B', 'C']),
                          pd.Series(['A', 'B', 'C']).to_frame()], 
                         ids = lambda exog: f'exog type: {type(exog)}')
def test_check_exog_dtypes_DataTypeWarning_when_exog_has_str_values(exog):
    """
    Check DataTypeWarning is issued when exog is pandas Series or DataFrame 
    with missing values.
    """
    warn_msg = re.escape(
        "`exog` may contain only `int`, `float` or `category` dtypes. Most "
        "machine learning models do not allow other types of values. "
        "Fitting the forecaster may fail."
    )
    with pytest.warns(DataTypeWarning, match = warn_msg):
        check_exog_dtypes(exog, call_check_exog=False)


@pytest.mark.parametrize("exog", 
                         [pd.Series([True, True, True], name='exog'),
                          pd.Series([True, True, True], name='exog').to_frame()], 
                         ids = lambda exog: f'exog type: {type(exog)}')
def test_check_exog_dtypes_DataTypeWarning_when_exog_has_bool_values(exog):
    """
    Check DataTypeWarning is issued when exog is pandas Series or DataFrame 
    with missing values.
    """
    warn_msg = re.escape(
        "`exog` may contain only `int`, `float` or `category` dtypes. Most "
        "machine learning models do not allow other types of values. "
        "Fitting the forecaster may fail."
    )
    with pytest.warns(DataTypeWarning, match = warn_msg):
        check_exog_dtypes(exog)


def test_check_exog_dtypes_TypeError_when_exog_is_Series_with_no_int_categories():
    """
    Check TypeError is raised when exog is pandas Series with no integer
    categories.
    """
    err_msg = re.escape(
        ("Categorical dtypes in exog must contain only integer values. "
         "See skforecast docs for more info about how to include "
         "categorical features https://skforecast.org/"
         "latest/user_guides/categorical-features.html")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_exog_dtypes(pd.Series(['A', 'B', 'C'], dtype='category', name='exog'))


def test_check_exog_dtypes_TypeError_when_exog_is_DataFrame_with_no_int_categories():
    """
    Check TypeError is raised when exog is pandas DataFrame with no integer
    categories.
    """
    err_msg = re.escape(
        ("Categorical dtypes in exog must contain only integer values. "
         "See skforecast docs for more info about how to include "
         "categorical features https://skforecast.org/"
         "latest/user_guides/categorical-features.html")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_exog_dtypes(pd.Series(['A', 'B', 'C'], dtype='category', name='exog').to_frame())


@pytest.mark.parametrize("dtype", 
                         ['int8', 'int16', 'int32', 'int64',
                          'float16', 'float32', 'float64',
                          'category'], 
                         ids = lambda exog: f'exog type: {type(exog)}')
def test_check_exog_dtypes_pandas_Series(dtype):
    """
    Test check_exog_dtypes accepts all dtypes in a pandas Series.
    """
    series = pd.Series(np.array([1, 2, 3]), name='exog', dtype=dtype) 
    _ = check_exog_dtypes(series)

    assert _ is None


def test_check_exog_dtypes_pandas_DataFrame():
    """
    Test check_exog_dtypes accepts all dtypes in a pandas DataFrame.
    """
    df = pd.DataFrame({'col1': [1., 2., 3.], 'col2': [4, 5, 6]}) 
    df['col2'] = df['col2'].astype('category')
    _ = check_exog_dtypes(df)

    assert _ is None


def test_check_exog_dtypes_DataFrame_multiple_categorical_columns():
    """
    Test check_exog_dtypes accepts DataFrame with multiple categorical columns
    containing integer values.
    """
    df = pd.DataFrame({
        'cat1': pd.Categorical([1, 2, 3]),
        'cat2': pd.Categorical([4, 5, 6]),
        'float_col': [1.0, 2.0, 3.0]
    })
    result = check_exog_dtypes(df, call_check_exog=False)
    
    assert result is None


def test_check_exog_dtypes_DataFrame_invalid_dtype_with_valid_categorical():
    """
    Test check_exog_dtypes raises warning when DataFrame has invalid dtype
    even if it also has valid categorical columns.
    """
    df = pd.DataFrame({
        'cat_col': pd.Categorical([1, 2, 3]),
        'str_col': ['a', 'b', 'c']
    })
    warn_msg = re.escape(
        "`exog` may contain only `int`, `float` or `category` dtypes. Most "
        "machine learning models do not allow other types of values. "
        "Fitting the forecaster may fail."
    )
    with pytest.warns(DataTypeWarning, match=warn_msg):
        check_exog_dtypes(df, call_check_exog=False)


@pytest.mark.parametrize("dtype", 
                         ['uint8', 'uint16', 'uint32', 'uint64'], 
                         ids=lambda d: f'dtype: {d}')
def test_check_exog_dtypes_pandas_Series_uint_dtypes(dtype):
    """
    Test check_exog_dtypes accepts unsigned integer dtypes in a pandas Series.
    """
    series = pd.Series(np.array([1, 2, 3]), name='exog', dtype=dtype)
    result = check_exog_dtypes(series, call_check_exog=False)

    assert result is None


@pytest.mark.parametrize("dtype", 
                         ['Int8', 'Int16', 'Int32', 'Int64',
                          'Float32', 'Float64'], 
                         ids=lambda d: f'dtype: {d}')
def test_check_exog_dtypes_pandas_Series_nullable_dtypes(dtype):
    """
    Test check_exog_dtypes accepts nullable integer and float dtypes (Int, Float)
    in a pandas Series.
    """
    series = pd.Series([1, 2, 3], name='exog', dtype=dtype)
    result = check_exog_dtypes(series, call_check_exog=False)

    assert result is None


def test_check_exog_dtypes_custom_series_id_in_warning():
    """
    Test check_exog_dtypes uses custom series_id in warning message.
    """
    exog = pd.Series(['A', 'B', 'C'], name='my_exog')
    custom_id = "`custom_exog_name`"
    warn_msg = re.escape(
        f"{custom_id} may contain only `int`, `float` or `category` dtypes. Most "
        "machine learning models do not allow other types of values. "
        "Fitting the forecaster may fail."
    )
    with pytest.warns(DataTypeWarning, match=warn_msg):
        check_exog_dtypes(exog, call_check_exog=False, series_id=custom_id)


def test_check_exog_dtypes_categorical_with_negative_integers():
    """
    Test check_exog_dtypes accepts categorical Series with negative integer values.
    """
    series = pd.Series([-1, -2, 3], name='exog', dtype='category')
    result = check_exog_dtypes(series, call_check_exog=False)

    assert result is None


def test_check_exog_dtypes_DataFrame_uint_and_nullable_dtypes():
    """
    Test check_exog_dtypes accepts DataFrame with mixed uint and nullable dtypes.
    """
    df = pd.DataFrame({
        'uint_col': pd.array([1, 2, 3], dtype='uint32'),
        'nullable_int': pd.array([4, 5, 6], dtype='Int64'),
        'nullable_float': pd.array([1.0, 2.0, 3.0], dtype='Float64')
    })
    result = check_exog_dtypes(df, call_check_exog=False)

    assert result is None
