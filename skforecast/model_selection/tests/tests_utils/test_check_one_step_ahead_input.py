# Unit test check_one_step_ahead_input
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.exceptions import OneStepAheadValidationWarning
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection._split import OneStepAheadFold
from skforecast.model_selection._utils import check_one_step_ahead_input

# Fixtures
from skforecast.model_selection.tests.fixtures_model_selection import y
from skforecast.model_selection.tests.fixtures_model_selection_multiseries import (
    series_dict_dt,
    series_dict_dt_different_periods
)


def test_check_one_step_ahead_input_TypeError_when_cv_not_OneStepAheadFold():
    """
    Test TypeError is raised in check_one_step_ahead_input if `cv` is not a
    OneStepAheadFold object.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=2)
    y = pd.Series(np.arange(50))
    y.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')
    
    class BadCv():
        pass

    err_msg = re.escape("`cv` must be a 'OneStepAheadFold' object. Got 'BadCv'.")
    with pytest.raises(TypeError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = BadCv(),
            metric            = 'mean_absolute_error',
            y                 = y,
            show_progress     = False,
            suppress_warnings = False
        )


def test_check_one_step_ahead_input_TypeError_when_forecaster_not_allowed():
    """
    Test TypeError is raised in check_one_step_ahead_input if `forecaster` is not
    allowed for OneStepAheadFold.
    """
    y = pd.Series(np.arange(50))
    y.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')

    forecasters_one_step_ahead = [
        "ForecasterRecursive",
        "ForecasterDirect",
        'ForecasterRecursiveClassifier',
        'ForecasterRecursiveMultiSeries',
        'ForecasterDirectMultiVariate'
    ]
    
    class BadForecaster():
        pass

    forecaster = BadForecaster()
    cv = OneStepAheadFold(initial_train_size=12)

    err_msg = re.escape(
        f"Only forecasters of type {forecasters_one_step_ahead} are allowed "
        f"when using `cv` of type `OneStepAheadFold`. Got {type(forecaster).__name__}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            y                 = y,
            show_progress     = False,
            suppress_warnings = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterRecursive(estimator=Ridge(), lags=2),
                          ForecasterDirect(estimator=Ridge(), lags=2, steps=3)], 
                         ids = lambda fr: f'{type(fr).__name__}')
def test_check_one_step_ahead_input_TypeError_when_y_is_not_pandas_Series(forecaster):
    """
    Test TypeError is raised in check_one_step_ahead_input if `y` is not a 
    pandas Series in forecasters uni-series.
    """
    bad_y = np.arange(50)
    cv = OneStepAheadFold(initial_train_size=len(bad_y) - 12)

    err_msg = re.escape(f"`y` must be a pandas Series. Got {type(bad_y)}")
    with pytest.raises(TypeError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            y                 = bad_y,
            show_progress     = False,
            suppress_warnings = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterDirectMultiVariate(estimator=Ridge(), lags=2, steps=3, level='l1')], 
                         ids = lambda fr: f'{type(fr).__name__}')
def test_check_one_step_ahead_input_TypeError_when_series_is_not_pandas_DataFrame(forecaster):
    """
    Test TypeError is raised in check_one_step_ahead_input if `series` is not a 
    pandas DataFrame in forecasters multiseries.
    """
    bad_series = pd.Series(np.arange(50))
    cv = OneStepAheadFold(initial_train_size=len(bad_series) - 12)

    err_msg = re.escape(f"`series` must be a pandas DataFrame. Got {type(bad_series)}")
    with pytest.raises(TypeError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            series            = bad_series,
            show_progress     = False,
            suppress_warnings = False
        )


def test_check_one_step_ahead_input_TypeError_when_not_valid_exog_type():
    """
    Test TypeError is raised in check_one_step_ahead_input if `exog` is not a
    pandas Series, DataFrame or None.
    """
    y = pd.Series(np.arange(50))
    y.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')

    forecaster = ForecasterRecursive(estimator=Ridge(), lags=2)

    bad_exog = np.arange(50)
    
    cv = OneStepAheadFold(initial_train_size=len(y) - 12)

    err_msg = re.escape(
        f"`exog` must be a pandas Series, DataFrame or None. Got {type(bad_exog)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            y                 = y,
            exog              = bad_exog,
            show_progress     = False,
            suppress_warnings = False
        )


def test_check_one_step_ahead_input_TypeError_when_metric_not_correct_type():
    """
    Test TypeError is raised in check_one_step_ahead_input if `metric` is not string, 
    a callable function, or a list containing multiple strings and/or callables.
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )
    
    cv = OneStepAheadFold(initial_train_size=len(y) - 12)
    
    metric = 5
    
    err_msg = re.escape(
        f"`metric` must be a string, a callable function, or a list containing "
        f"multiple strings and/or callables. Got {type(metric)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = metric,
            y                 = y,
            show_progress     = False,
            suppress_warnings = False
        )


@pytest.mark.parametrize("initial_train_size", 
                         ['greater', 'smaller', 'date'], 
                         ids = lambda initial: f'initial_train_size: {initial}')
@pytest.mark.parametrize("forecaster", 
                         [ForecasterRecursive(estimator=Ridge(), lags=3),
                          ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=3)], 
                         ids = lambda fr: f'{type(fr).__name__}')
def test_check_one_step_ahead_input_ValueError_when_initial_train_size_not_correct_value(initial_train_size, forecaster):
    """
    Test ValueError is raised in check_one_step_ahead_input when 
    initial_train_size >= length `y` or span of `series` or initial_train_size
    < window_size.
    """
    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start='2020-01-01', periods=len(y), freq='D')

    if type(forecaster).__name__ == 'ForecasterRecursive':
        data_length = len(y_datetime)
        data_length_name = 'the length of `y`'
    else:
        data_length = len(series_dict_dt['l1'])
        data_length_name = 'the length of the span of `series`'

    if initial_train_size == 'greater':
        initial_train_size = data_length
    elif initial_train_size == 'smaller':
        initial_train_size = forecaster.window_size - 1
    else:
        initial_train_size = '2020-01-02'  # Smaller than window_size
    
    cv = OneStepAheadFold(initial_train_size=initial_train_size)
    
    err_msg = re.escape(
        f"If `initial_train_size` is an integer, it must be greater than "
        f"the `window_size` of the forecaster ({forecaster.window_size}) "
        f"and smaller than {data_length_name} ({data_length}). If "
        f"it is a date, it must be within this range of the index."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            y                 = y_datetime,
            series            = series_dict_dt,
            show_progress     = False,
            suppress_warnings = False
        )


@pytest.mark.parametrize("initial_train_size",
                         [50, '2020-02-19'],
                         ids = lambda initial: f'initial_train_size: {initial}')
def test_check_one_step_ahead_input_ValueError_when_initial_train_size_not_smaller_than_span_of_series_dict(initial_train_size):
    """
    Test ValueError is raised in check_one_step_ahead_input when `series` is a
    dict of series that do not cover the same period and `initial_train_size`
    is not smaller than the length of the span of the series (50), which is
    longer than any of them (30).
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=3)
    cv = OneStepAheadFold(initial_train_size=initial_train_size)

    err_msg = re.escape(
        "If `initial_train_size` is an integer, it must be greater than "
        "the `window_size` of the forecaster (3) and smaller than the length "
        "of the span of `series` (50). If it is a date, it must be within this "
        "range of the index."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            series            = series_dict_dt_different_periods,
            show_progress     = False,
            suppress_warnings = False
        )


@pytest.mark.parametrize("initial_train_size",
                         [40, '2020-02-09'],
                         ids = lambda initial: f'initial_train_size: {initial}')
def test_check_one_step_ahead_input_no_error_when_initial_train_size_greater_than_longest_series_but_smaller_than_span(initial_train_size):
    """
    Test no error is raised in check_one_step_ahead_input when `series` is a
    dict of series that do not cover the same period and `initial_train_size`
    is greater than the length of the longest series (30) but smaller than
    the span of the series (50), over which the train and test sets are split.
    """
    forecaster = ForecasterRecursiveMultiSeries(estimator=Ridge(), lags=3)
    cv = OneStepAheadFold(initial_train_size=initial_train_size)

    check_one_step_ahead_input(
        forecaster        = forecaster,
        cv                = cv,
        metric            = 'mean_absolute_error',
        series            = series_dict_dt_different_periods,
        show_progress     = False,
        suppress_warnings = True
    )


@pytest.mark.parametrize("boolean_argument",
                         ['show_progress', 'suppress_warnings'],
                         ids = lambda argument: f'{argument}')
def test_check_one_step_ahead_input_TypeError_when_boolean_arguments_not_bool(boolean_argument):
    """
    Test TypeError is raised in check_one_step_ahead_input when boolean arguments 
    are not boolean.
    """
    forecaster = ForecasterRecursive(
                    estimator = Ridge(random_state=123),
                    lags      = 2
                 )
    
    cv = OneStepAheadFold(initial_train_size= len(y) - 12)
    
    boolean_arguments = {
        'show_progress': False,
        'suppress_warnings': False
    }
    boolean_arguments[boolean_argument] = 'not_bool'
    
    err_msg = re.escape(f"`{boolean_argument}` must be a boolean: `True`, `False`.")
    with pytest.raises(TypeError, match = err_msg):
        check_one_step_ahead_input(
            forecaster = forecaster,
            cv         = cv,
            metric     = 'mean_absolute_error',
            y          = y,
            **boolean_arguments
        )


def test_check_one_step_ahead_input_OneStepAheadValidationWarning():
    """
    Test OneStepAheadValidationWarning is warned in check_one_step_ahead_input
    when all checks are passed and suppress_warnings is False.
    """
    forecaster = ForecasterRecursive(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )
    
    cv = OneStepAheadFold(initial_train_size=len(y) - 12)
    
    warn_msg = re.escape(
        "One-step-ahead predictions are used for faster model comparison, but they "
        "may not fully represent multi-step prediction performance. It is recommended "
        "to backtest the final model for a more accurate multi-step performance "
        "estimate."
    )
    with pytest.warns(OneStepAheadValidationWarning, match = warn_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            y                 = y,
            show_progress     = False,
            suppress_warnings = False
        )
