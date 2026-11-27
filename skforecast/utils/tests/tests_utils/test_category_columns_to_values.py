# Unit test _category_columns_to_values
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import DataTypeWarning
from skforecast.utils import _category_columns_to_values


def test_category_columns_to_values_returns_same_object_when_nothing_to_replace():
    """
    Test that the same DataFrame is returned, without warnings, when there are
    no columns of dtype category other than the series identifier
    `_level_skforecast`.
    """
    X = pd.DataFrame({
            'lag_1': [1.5, 2.5, 3.5],
            '_level_skforecast': pd.Categorical([0, 1, 1]),
            'exog_1': [10, 20, 30]
        })

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        results = _category_columns_to_values(X, warn=True)

    assert results is X
    assert not record


@pytest.mark.parametrize(
    'warn',
    [True, False],
    ids=lambda warn: f'warn: {warn}'
)
def test_category_columns_to_values_output_when_integer_and_float_categories(warn):
    """
    Test that the columns of dtype category are replaced by their values (not
    their codes) as float, with NaN in the missing values, that the series
    identifier `_level_skforecast` and the other columns are kept, that the
    input is not modified and that a DataTypeWarning names the replaced columns
    only if `warn=True`.
    """
    X = pd.DataFrame({
            'lag_1': [1.5, 2.5, 3.5, 4.5],
            '_level_skforecast': pd.Categorical([0, 0, 1, 1]),
            'exog_int': pd.Categorical(
                [30, 10, np.nan, 20], categories=[30, 10, 20]
            ),
            'exog_float': pd.Categorical([0.5, 2.5, 1.5, 0.5]),
            'exog_num': [1, 2, 3, 4]
        })
    X_copy = X.copy()

    warn_msg = re.escape(
        "The exogenous variables ['exog_int', 'exog_float'] have dtype "
        "`category` but are not included in `categorical_features`, so the "
        "forecaster does not encode them: the estimator receives their "
        "values, not their category codes. To let the forecaster handle them "
        "as categorical features, include them in `categorical_features` or "
        "use `categorical_features='auto'`."
    )
    if warn:
        with pytest.warns(DataTypeWarning, match=warn_msg):
            results = _category_columns_to_values(X, warn=True)
    else:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            results = _category_columns_to_values(X, warn=False)
        assert not record

    expected = pd.DataFrame({
                   'lag_1': [1.5, 2.5, 3.5, 4.5],
                   '_level_skforecast': pd.Categorical([0, 0, 1, 1]),
                   'exog_int': [30., 10., np.nan, 20.],
                   'exog_float': [0.5, 2.5, 1.5, 0.5],
                   'exog_num': [1, 2, 3, 4]
               })

    pd.testing.assert_frame_equal(results, expected)
    pd.testing.assert_frame_equal(X, X_copy)


@pytest.mark.parametrize(
    'column',
    [pd.Categorical(['a', 'b', 'c']),
     pd.Categorical(['1', '2', '3']),
     pd.Categorical([True, False, True]),
     pd.Categorical(pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']))],
    ids=['text', 'numeric strings', 'bool', 'datetime']
)
def test_category_columns_to_values_TypeError_when_categories_are_not_numbers(column):
    """
    Test TypeError is raised, with the names of the columns, when the
    categories of a column of dtype category are not integers or floats. The
    DataTypeWarning of the replaced columns is not issued before the error.
    """
    X = pd.DataFrame({
            'lag_1': [1.5, 2.5, 3.5],
            'exog_int': pd.Categorical([10, 20, 30]),
            'exog_cat': column
        })

    err_msg = re.escape(
        "The exogenous variables ['exog_cat'] have dtype `category` and their "
        "categories are not numbers, so they cannot be passed to the "
        "estimator with their values. Include them in `categorical_features`, "
        "or use `categorical_features='auto'`, so that the forecaster encodes "
        "them."
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        with pytest.raises(TypeError, match=err_msg):
            _category_columns_to_values(X, warn=True)

    assert not [w for w in record if issubclass(w.category, DataTypeWarning)]
