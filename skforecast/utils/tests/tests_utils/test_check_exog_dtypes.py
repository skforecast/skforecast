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
                         ids = lambda exog : f'exog type: {type(exog)}')
def test_check_exog_dtypes_DataTypeWarning_when_exog_has_str_values(exog):
    """
    Check DataTypeWarning is issued when exog is pandas Series or DataFrame 
    with missing values.
    """
    warn_msg = re.escape(
        ("`exog` may contain only `int`, `float` or `category` dtypes. Most "
         "machine learning models do not allow other types of values. "
         "Fitting the forecaster may fail.")
    )
    with pytest.warns(DataTypeWarning, match = warn_msg):
        check_exog_dtypes(exog, call_check_exog=False)


@pytest.mark.parametrize("exog", 
                         [pd.Series([True, True, True], name='exog'),
                          pd.Series([True, True, True], name='exog').to_frame()], 
                         ids = lambda exog : f'exog type: {type(exog)}')
def test_check_exog_dtypes_DataTypeWarning_when_exog_has_bool_values(exog):
    """
    Check DataTypeWarning is issued when exog is pandas Series or DataFrame 
    with missing values.
    """
    warn_msg = re.escape(
        ("`exog` may contain only `int`, `float` or `category` dtypes. Most "
         "machine learning models do not allow other types of values. "
         "Fitting the forecaster may fail.")
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
                         ['int8','int16','int32','int64',
                          'float16','float32', 'float64',
                          'category'], 
                         ids = lambda exog : f'exog type: {type(exog)}')
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