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
from skforecast.model_selection.tests.fixtures_model_selection_multiseries import series


def test_check_one_step_ahead_input_TypeError_when_cv_not_OneStepAheadFold():
    """
    Test TypeError is raised in check_one_step_ahead_input if `cv` is not a
    OneStepAheadFold object.
    """
    forecaster = ForecasterRecursive(regressor=Ridge(), lags=2)
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
                         [ForecasterRecursive(regressor=Ridge(), lags=2),
                          ForecasterDirect(regressor=Ridge(), lags=2, steps=3)], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_one_step_ahead_input_TypeError_when_y_is_not_pandas_Series_uniseries(forecaster):
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
                         [ForecasterDirectMultiVariate(regressor=Ridge(), lags=2, steps=3, level='l1')], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_one_step_ahead_input_TypeError_when_series_is_not_pandas_DataFrame_multiseries(forecaster):
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


def test_check_one_step_ahead_input_TypeError_when_series_is_not_pandas_DataFrame_multiseries_dict():
    """
    Test TypeError is raised in check_one_step_ahead_input if `series` is not a 
    pandas DataFrame in forecasters multiseries with dict.
    """
    forecaster = ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)
    bad_series = pd.Series(np.arange(50))
    cv = OneStepAheadFold(initial_train_size=len(bad_series) - 12)

    err_msg = re.escape(
        f"`series` must be a pandas DataFrame or a dict of DataFrames or Series. "
        f"Got {type(bad_series)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            series            = bad_series,
            show_progress     = False,
            suppress_warnings = False
        )


def test_check_one_step_ahead_input_TypeError_when_series_is_dict_of_pandas_Series_multiseries_dict():
    """
    Test TypeError is raised in check_one_step_ahead_input if `series` is not a 
    dict of pandas Series in forecasters multiseries with dict.
    """
    forecaster = ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)
    bad_series = {'l1': np.arange(50)}
    cv = OneStepAheadFold(initial_train_size=len(bad_series['l1']) - 12)

    err_msg = re.escape(
        "If `series` is a dictionary, all series must be a named "
        "pandas Series or a pandas DataFrame with a single column. "
        "Review series: ['l1']"
    )
    with pytest.raises(TypeError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            series            = bad_series,
            show_progress     = False,
            suppress_warnings = False
        )


def test_check_one_step_ahead_input_ValueError_when_series_is_dict_no_DatetimeIndex_multiseries_dict():
    """
    Test ValueError is raised in check_one_step_ahead_input if `series` is a 
    dict with pandas Series with no DatetimeIndex in forecasters 
    multiseries with dict.
    """
    forecaster = ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)
    series_dict = {
        'l1': pd.Series(np.arange(50)),
        'l2': pd.Series(np.arange(50))
    }
    cv = OneStepAheadFold(initial_train_size=len(series_dict['l1']) - 12)

    err_msg = re.escape(
        "If `series` is a dictionary, all series must have a Pandas DatetimeIndex "
        "as index with the same frequency. Review series: ['l1', 'l2']"
    )
    with pytest.raises(ValueError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            series            = series_dict,
            show_progress     = False,
            suppress_warnings = False
        )


def test_check_one_step_ahead_input_ValueError_when_series_is_dict_diff_freq_multiseries_dict():
    """
    Test ValueError is raised in check_one_step_ahead_input if `series` is a 
    dict with pandas Series of difference frequency in forecasters 
    multiseries with dict.
    """
    forecaster = ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)
    series_dict = {
        'l1': pd.Series(np.arange(50)),
        'l2': pd.Series(np.arange(50))
    }
    series_dict['l1'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l1']), freq='D'
    )
    series_dict['l2'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l2']), freq='MS'
    )
    
    cv = OneStepAheadFold(initial_train_size=len(series_dict['l1']) - 12)

    err_msg = re.escape(
        "If `series` is a dictionary, all series must have a Pandas DatetimeIndex "
        "as index with the same frequency. Found frequencies: ['<Day>', '<MonthBegin>']"
    )
    with pytest.raises(ValueError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            series            = series_dict,
            show_progress     = False,
            suppress_warnings = False
        )


def test_check_one_step_ahead_input_TypeError_when_not_valid_exog_type_multiseries_dict():
    """
    Test TypeError is raised in check_one_step_ahead_input if `exog` is not a
    pandas Series, DataFrame, dictionary of pandas Series/DataFrames or None.
    """
    forecaster = ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)
    series_dict = {
        'l1': pd.Series(np.arange(50)),
        'l2': pd.Series(np.arange(50))
    }
    series_dict['l1'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l1']), freq='D'
    )
    series_dict['l2'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l2']), freq='D'
    )

    bad_exog = np.arange(50)
    
    cv = OneStepAheadFold(initial_train_size=len(series_dict['l1']) - 12)

    err_msg = re.escape(
        f"`exog` must be a pandas Series, DataFrame, dictionary of pandas "
        f"Series/DataFrames or None. Got {type(bad_exog)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            series            = series_dict,
            exog              = bad_exog,
            show_progress     = False,
            suppress_warnings = False
        )


def test_check_one_step_ahead_input_TypeError_when_not_valid_exog_dict_type_multiseries_dict():
    """
    Test TypeError is raised in check_one_step_ahead_input if `exog` is not a
    dictionary of pandas Series/DataFrames.
    """
    forecaster = ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)
    series_dict = {
        'l1': pd.Series(np.arange(50)),
        'l2': pd.Series(np.arange(50))
    }
    series_dict['l1'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l1']), freq='D'
    )
    series_dict['l2'].index = pd.date_range(
        start='2000-01-01', periods=len(series_dict['l2']), freq='D'
    )

    bad_exog = {'l1': np.arange(50)}
    
    cv = OneStepAheadFold(initial_train_size=len(series_dict['l1']) - 12)

    err_msg = re.escape(
        "If `exog` is a dictionary, All exog must be a named pandas "
        "Series, a pandas DataFrame or None. Review exog: ['l1']"
    )
    with pytest.raises(TypeError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            series            = series_dict,
            exog              = bad_exog,
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

    forecaster = ForecasterRecursive(regressor=Ridge(), lags=2)

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


@pytest.mark.parametrize("differentiation", 
    [{'l1': 1, 'l2': 2, '_unknown_level': 1}, {'l1': 2, 'l2': None, '_unknown_level': 1}], 
     ids = lambda diff: f'differentiation: {diff}')
def test_check_one_step_ahead_input_ValueError_when_ForecasterRecursiveMultiSeries_diff_dict_not_cv_diff(differentiation):
    """
    Test ValueError is raised in check_one_step_ahead_input if `differentiation`
    of the ForecasterRecursiveMultiSeries as dict is different from 
    `differentiation` of the cv.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=Ridge(), lags=2, differentiation=differentiation
    )
    cv = OneStepAheadFold(initial_train_size=len(series) - 12, differentiation=1)
    
    err_msg = re.escape(
        "When using a dict as `differentiation` in ForecasterRecursiveMultiSeries, "
        "the `differentiation` included in the cv (1) must be "
        "the same as the maximum `differentiation` included in the forecaster "
        "(2). Set the same value "
        "for both using the `differentiation` argument."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            series            = series,
            show_progress     = False,
            suppress_warnings = False
        )


@pytest.mark.parametrize("forecaster", 
    [ForecasterRecursive(regressor=Ridge(), lags=2, differentiation=2),
     ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2, differentiation=2)], 
     ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_one_step_ahead_input_ValueError_when_forecaster_diff_not_cv_diff(forecaster):
    """
    Test ValueError is raised in check_one_step_ahead_input if `differentiation`
    of the forecaster is different from `differentiation` of the cv.
    """
    if type(forecaster).__name__ == 'ForecasterRecursive':
        data_length = len(y)
    else:
        data_length = len(series)
    
    cv = OneStepAheadFold(initial_train_size=data_length - 12, differentiation=1)
    
    err_msg = re.escape(
        "The differentiation included in the forecaster "
        "(2) differs from the differentiation "
        "included in the cv (1). Set the same value "
        "for both using the `differentiation` argument."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            y                 = y,
            series            = series,
            show_progress     = False,
            suppress_warnings = False
        )


def test_check_one_step_ahead_input_TypeError_when_metric_not_correct_type():
    """
    Test TypeError is raised in check_one_step_ahead_input if `metric` is not string, 
    a callable function, or a list containing multiple strings and/or callables.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
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
                         [ForecasterRecursive(regressor=Ridge(), lags=3),
                          ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=3)], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_one_step_ahead_input_ValueError_when_initial_train_size_not_correct_value(initial_train_size, forecaster):
    """
    Test ValueError is raised in check_one_step_ahead_input when 
    initial_train_size >= length `y` or `series` or initial_train_size < window_size.
    """
    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')

    series_datetime = series.copy()
    series_datetime.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')

    if type(forecaster).__name__ == 'ForecasterRecursive':
        data_length = len(y_datetime)
        data_name = 'y'
    else:
        data_length = len(series_datetime)
        data_name = 'series'

    if initial_train_size == 'greater':
        initial_train_size = data_length
    elif initial_train_size == 'smaller':
        initial_train_size = forecaster.window_size - 1
    else:
        initial_train_size = '2000-01-02'  # Smaller than window_size
    
    cv = OneStepAheadFold(initial_train_size=initial_train_size)
    
    err_msg = re.escape(
        f"If `initial_train_size` is an integer, it must be greater than "
        f"the `window_size` of the forecaster ({forecaster.window_size}) "
        f"and smaller than the length of `{data_name}` ({data_length}). If "
        f"it is a date, it must be within this range of the index."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_one_step_ahead_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            y                 = y_datetime,
            series            = series_datetime,
            show_progress     = False,
            suppress_warnings = False
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
                    regressor = Ridge(random_state=123),
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
                     regressor = Ridge(random_state=123),
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
