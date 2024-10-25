# Unit test prepare_residuals_multiseries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from skforecast.exceptions import UnknownLevelWarning
from skforecast.utils import prepare_residuals_multiseries
from skforecast.recursive import ForecasterRecursiveMultiSeries

# Fixtures
from skforecast.recursive.tests.tests_forecaster_recursive_multiseries.fixtures_forecaster_recursive_multiseries import series


def test_prepare_residuals_multiseries_ValueError_when_not_in_sample_residuals_for_any_level():
    """
    Test ValueError is raised when there is no in_sample_residuals_ for any level.
    """
    levels = ['1', '2']
    use_in_sample_residuals = True

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series['1'].to_frame())

    warn_msg = re.escape(
        ("`levels` {'2'} are not present in `forecaster.in_sample_residuals_`, "
         "most likely because they were not present in the training data. "
         "A random sample of the residuals from other levels will be used. "
         "This can lead to inaccurate intervals for the unknown levels.")
    )
    with pytest.warns(UnknownLevelWarning, match = warn_msg):
        residuals = prepare_residuals_multiseries(
                        levels                  = levels,
                        use_in_sample_residuals = use_in_sample_residuals, 
                        encoding                = forecaster.encoding,
                        in_sample_residuals_    = forecaster.in_sample_residuals_,
                        out_sample_residuals_   = forecaster.out_sample_residuals_
                    )
        
    expected = {
        '1': forecaster.in_sample_residuals_['1'],
        '2': forecaster.in_sample_residuals_['_unknown_level'],
        '_unknown_level': forecaster.in_sample_residuals_['_unknown_level']
    }
    
    for k in residuals.keys():
        np.testing.assert_array_almost_equal(residuals[k], expected[k])


def test_prepare_residuals_multiseries_ValueError_when_out_sample_residuals_is_None():
    """
    Test ValueError is raised when `use_in_sample_residuals` is False and 
    `out_sample_residuals_` is None.
    """
    levels = ['1', '2']
    use_in_sample_residuals = False

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)

    err_msg = re.escape(
        ("`forecaster.out_sample_residuals_` is `None`. Use "
         "`use_in_sample_residuals=True` or the `set_out_sample_residuals()` "
         "method before predicting.")
    )
    with pytest.raises(ValueError, match = err_msg):
        prepare_residuals_multiseries(
            levels                  = levels,
            use_in_sample_residuals = use_in_sample_residuals, 
            encoding                = forecaster.encoding,
            in_sample_residuals_    = forecaster.in_sample_residuals_,
            out_sample_residuals_   = forecaster.out_sample_residuals_
        )


def test_prepare_residuals_multiseries_ValueError_when_not_out_sample_residuals_for_all_levels():
    """
    Test ValueError is raised when out_sample_residuals_ is not 
    available for all levels.
    """
    levels = ['1', '2']
    use_in_sample_residuals = False
    new_residuals = {'1': np.array([1, 2, 3, 4, 5])}

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series['1'].to_frame())
    forecaster.set_out_sample_residuals(new_residuals)

    warn_msg = re.escape(
        ("`levels` {'2'} are not present in `forecaster.out_sample_residuals_`. "
         "A random sample of the residuals from other levels will be used. "
         "This can lead to inaccurate intervals for the unknown levels. "
         "Otherwise, Use the `set_out_sample_residuals()` method before "
         "predicting to set the residuals for these levels."),
    )
    with pytest.warns(UnknownLevelWarning, match = warn_msg):
        residuals = prepare_residuals_multiseries(
                        levels                  = levels,
                        use_in_sample_residuals = use_in_sample_residuals, 
                        encoding                = forecaster.encoding,
                        in_sample_residuals_    = forecaster.in_sample_residuals_,
                        out_sample_residuals_   = forecaster.out_sample_residuals_
                    )
        
    expected = {
        '1': np.array([1, 2, 3, 4, 5]),
        '2': np.array([1, 2, 3, 4, 5]),
        '_unknown_level': np.array([1, 2, 3, 4, 5])
    }
    
    for k in residuals.keys():
        np.testing.assert_array_almost_equal(residuals[k], expected[k])


@pytest.mark.parametrize("transformer_series", 
                         [None, StandardScaler()],
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_prepare_residuals_multiseries_ValueError_when_level_out_sample_residuals_value_is_None(transformer_series):
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_ has a level with a None.
    """
    levels = ['1', '2']
    use_in_sample_residuals = False
    out_sample_residuals_ = {'1': np.array([1, 2, 3, 4, 5])}

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=transformer_series)
    forecaster.fit(series=series)
    forecaster.set_out_sample_residuals(residuals=out_sample_residuals_)

    err_msg = re.escape(
        ("Not available residuals for level '2'. "
         "Check `forecaster.out_sample_residuals_`.")
    )
    with pytest.raises(ValueError, match = err_msg):
        prepare_residuals_multiseries(
            levels                  = levels,
            use_in_sample_residuals = use_in_sample_residuals, 
            encoding                = forecaster.encoding,
            in_sample_residuals_    = forecaster.in_sample_residuals_,
            out_sample_residuals_   = forecaster.out_sample_residuals_
        )


@pytest.mark.parametrize("transformer_series", 
                         [None, StandardScaler()],
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_prepare_residuals_multiseries_ValueError_when_level_out_sample_residuals_value_contains_None_or_NaNs(transformer_series):
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_ has a level with a None or NaN value.
    """
    levels = ['1', '2']
    use_in_sample_residuals = False
    out_sample_residuals_ = {
        '1': np.array([1, 2, 3, 4, 5]),
        '2': np.array([1, 2, 3, 4, None])  # StandardScaler() transforms None to NaN
    }
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=transformer_series)
    forecaster.fit(series=series)
    forecaster.set_out_sample_residuals(residuals = out_sample_residuals_)

    err_msg = re.escape(
        ("forecaster residuals for level '2' contains `None` "
         "or `NaNs` values. Check `forecaster.out_sample_residuals_`.")
    )
    with pytest.raises(ValueError, match = err_msg):
        prepare_residuals_multiseries(
            levels                  = levels,
            use_in_sample_residuals = use_in_sample_residuals, 
            encoding                = forecaster.encoding,
            in_sample_residuals_    = forecaster.in_sample_residuals_,
            out_sample_residuals_   = forecaster.out_sample_residuals_
        )


@pytest.mark.parametrize("use_in_sample_residuals", 
                         [True, False],
                         ids = lambda use_in_sample_residuals: f'use_in_sample_residuals: {use_in_sample_residuals}')
def test_output_prepare_residuals_multiseries(use_in_sample_residuals):
    """
    Test output of prepare_residuals_multiseries when use_in_sample_residuals=True and False.
    """
    levels = ['1', '2']
    residuals = {
        '1': np.array([1, 2, 3, 4, 5]),
        '2': np.array([1, 2, 3, 4, 5]),
        '_unknown_level': np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    }
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series)
    forecaster.in_sample_residuals_ = residuals
    forecaster.set_out_sample_residuals(residuals=residuals)

    residuals_levels = prepare_residuals_multiseries(
                            levels                  = levels,
                            use_in_sample_residuals = use_in_sample_residuals, 
                            encoding                = forecaster.encoding,
                            in_sample_residuals_    = forecaster.in_sample_residuals_,
                            out_sample_residuals_   = forecaster.out_sample_residuals_
                       )
    
    for level in residuals.keys():
        np.testing.assert_array_equal(residuals_levels[level], residuals[level])
