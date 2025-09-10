# Unit test check_backtesting_input
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.recursive import ForecasterSarimax
from skforecast.recursive import ForecasterEquivalentDate
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection._utils import check_backtesting_input

# Fixtures
from skforecast.model_selection.tests.fixtures_model_selection import y
from skforecast.model_selection.tests.fixtures_model_selection_multiseries import (
    series_wide_range,
    series_dict_range,
    series_dict_dt
)


def test_check_backtesting_input_TypeError_when_cv_not_TimeSeriesFold():
    """
    Test TypeError is raised in check_backtesting_input if `cv` is not a
    TimeSeriesFold object.
    """
    forecaster = ForecasterRecursive(regressor=Ridge(), lags=2)
    y = pd.Series(np.arange(50))
    y.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')
    
    class BadCv():
        pass

    err_msg = re.escape("`cv` must be a 'TimeSeriesFold' object. Got 'BadCv'.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = BadCv(),
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = None,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterRecursive(regressor=Ridge(), lags=2),
                          ForecasterDirect(regressor=Ridge(), lags=2, steps=3),
                          ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1)))], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_backtesting_input_TypeError_when_y_is_not_pandas_Series(forecaster):
    """
    Test TypeError is raised in check_backtesting_input if `y` is not a 
    pandas Series in forecasters uni-series.
    """
    bad_y = np.arange(50)
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(bad_y) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             verbose               = False
         )

    err_msg = re.escape("`y` must be a pandas Series.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = bad_y,
            series                  = None,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterDirectMultiVariate(regressor=Ridge(), lags=2, 
                                                        steps=3, level='l1')], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_backtesting_input_TypeError_when_series_not_pandas_DataFrame(forecaster):
    """
    Test TypeError is raised in check_backtesting_input if `series` is not a 
    pandas DataFrame in forecasters multiseries.
    """
    bad_series = pd.Series(np.arange(50))
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(bad_series) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             verbose               = False
         )

    err_msg = re.escape("`series` must be a pandas DataFrame.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            series                  = bad_series,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_TypeError_when_not_valid_exog_type():
    """
    Test TypeError is raised in check_backtesting_input if `exog` is not a
    pandas Series, DataFrame or None.
    """
    y = pd.Series(np.arange(50))
    y.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')

    forecaster = ForecasterRecursive(regressor=Ridge(), lags=2)

    bad_exog = np.arange(50)
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y[:-12]),
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             verbose               = False
         )

    err_msg = re.escape(
        f"`exog` must be a pandas Series, DataFrame or None. Got {type(bad_exog)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            exog                    = bad_exog,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("differentiation", 
    [{'l1': 1, 'l2': 2, '_unknown_level': 1}, {'l1': 2, 'l2': None, '_unknown_level': 1}], 
     ids = lambda diff: f'differentiation: {diff}')
def test_check_backtesting_input_ValueError_when_ForecasterRecursiveMultiSeries_diff_dict_not_cv_diff(differentiation):
    """
    Test ValueError is raised in check_backtesting_input if `differentiation`
    of the ForecasterRecursiveMultiSeries as dict is different from 
    `differentiation` of the cv.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=Ridge(), lags=2, differentiation=differentiation
    )

    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(series_dict_range['l1']) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             differentiation       = 1,
             verbose               = False
         )
    
    err_msg = re.escape(
        "When using a dict as `differentiation` in ForecasterRecursiveMultiSeries, "
        "the `differentiation` included in the cv (1) must be "
        "the same as the maximum `differentiation` included in the forecaster "
        "(2). Set the same value "
        "for both using the `differentiation` argument."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            series                  = series_dict_range,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("forecaster", 
    [ForecasterRecursive(regressor=Ridge(), lags=2, differentiation=2),
     ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2, differentiation=2)], 
     ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_backtesting_input_ValueError_when_forecaster_diff_not_cv_diff(forecaster):
    """
    Test ValueError is raised in check_backtesting_input if `differentiation`
    of the forecaster is different from `differentiation` of the cv.
    """
    if type(forecaster).__name__ == 'ForecasterRecursive':
        data_length = len(y)
    else:
        data_length = len(series_dict_range['l1'])
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = data_length - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             differentiation       = 1,
             verbose               = False
         )
    
    err_msg = re.escape(
        "The differentiation included in the forecaster "
        "(2) differs from the differentiation "
        "included in the cv (1). Set the same value "
        "for both using the `differentiation` argument."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series_dict_range,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_TypeError_when_metric_not_correct_type():
    """
    Test TypeError is raised in check_backtesting_input if `metric` is not string, 
    a callable function, or a list containing multiple strings and/or callables.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    cv = TimeSeriesFold(
             steps                 = 5,
             initial_train_size    = len(y[:-12]),
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric = 5
    
    err_msg = re.escape(
        f"`metric` must be a string, a callable function, or a list containing "
        f"multiple strings and/or callables. Got {type(metric)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = metric,
            y                       = y,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_ValueError_when_initial_train_size_is_None_ForecasterEquivalentDate():
    """
    Test ValueError is raised in check_backtesting_input when initial_train_size
    is None with a ForecasterEquivalentDate with a offset of type pd.DateOffset.
    """

    data_length = len(y)
    data_name = 'y'

    forecaster = ForecasterEquivalentDate(
                     offset    = pd.DateOffset(days=10),
                     n_offsets = 2 
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = None,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    err_msg = re.escape(
        f"`initial_train_size` must be an integer greater than "
        f"the `window_size` of the forecaster ({forecaster.window_size}) "
        f"and smaller than the length of `{data_name}` ({data_length}) or "
        f"a date within this range of the index."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("initial_train_size", 
                         ['greater', 'smaller', 'date'], 
                         ids = lambda initial: f'initial_train_size: {initial}')
@pytest.mark.parametrize("forecaster", 
                         [ForecasterRecursive(regressor=Ridge(), lags=3),
                          ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=3)], 
                         ids = lambda fr: f'{type(fr).__name__}')
def test_check_backtesting_input_ValueError_when_initial_train_size_not_correct_value(initial_train_size, forecaster):
    """
    Test ValueError is raised in check_backtesting_input when 
    initial_train_size >= length `y` or `series` or initial_train_size < window_size.
    """
    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start='2020-01-01', periods=len(y), freq='D')

    if type(forecaster).__name__ == 'ForecasterRecursive':
        data_length = len(y_datetime)
        data_name = 'y'
    else:
        data_length = len(series_dict_dt['l1'])
        data_name = 'series'

    if initial_train_size == 'greater':
        initial_train_size = data_length
    elif initial_train_size == 'smaller':
        initial_train_size = forecaster.window_size - 1
    else:
        initial_train_size = '2020-01-02'  # Smaller than window_size
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = initial_train_size,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    err_msg = re.escape(
        f"If `initial_train_size` is an integer, it must be greater than "
        f"the `window_size` of the forecaster ({forecaster.window_size}) "
        f"and smaller than the length of `{data_name}` ({data_length}). If "
        f"it is a date, it must be within this range of the index."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y_datetime,
            series                  = series_dict_dt,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("initial_train_size", 
                         ['int', 'date'], 
                         ids = lambda initial: f'initial_train_size: {initial}')
@pytest.mark.parametrize("forecaster", 
                         [ForecasterRecursive(regressor=Ridge(), lags=2),
                          ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)], 
                         ids = lambda fr: f'{type(fr).__name__}')
def test_check_backtesting_input_ValueError_when_not_enough_data_to_create_a_fold_allow_incomplete_fold(initial_train_size, forecaster):
    """
    Test ValueError is raised in check_backtesting_input when there is not enough 
    data to evaluate even single fold because `allow_incomplete_fold` = `True`.
    """
    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')

    if type(forecaster).__name__ == 'ForecasterRecursive':
        data_length = len(y_datetime)
        data_name = 'y'
    else:
        data_length = len(series_dict_dt['l1'])
        data_name = 'series'

    if initial_train_size == 'int':
        initial_train_size = data_length - 1
    else:
        initial_train_size = '2000-02-19'
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = data_length - 1,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 2,
             allow_incomplete_fold = True
         )
    
    err_msg = re.escape(
        f"`{data_name}` must have more than `initial_train_size + gap` "
        f"observations to create at least one fold.\n"
        f"    Time series length: {data_length}\n"
        f"    Required > {cv.initial_train_size + cv.gap}\n"
        f"    initial_train_size: {cv.initial_train_size}\n"
        f"    gap: {cv.gap}\n"
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series_dict_range,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("initial_train_size", 
                         ['int', 'date'], 
                         ids = lambda initial: f'initial_train_size: {initial}')
@pytest.mark.parametrize("forecaster", 
                         [ForecasterRecursive(regressor=Ridge(), lags=2),
                          ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)], 
                         ids = lambda fr: f'{type(fr).__name__}')
def test_check_backtesting_input_ValueError_when_not_enough_data_to_create_a_fold_allow_incomplete_fold_False(initial_train_size, forecaster):
    """
    Test ValueError is raised in check_backtesting_input when there is not enough 
    data to evaluate even single fold because `allow_incomplete_fold` = `False`.
    """
    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')

    if type(forecaster).__name__ == 'ForecasterRecursive':
        data_length = len(y_datetime)
        data_name = 'y'
    else:
        data_length = len(series_dict_dt['l1'])
        data_name = 'series'

    if initial_train_size == 'int':
        initial_train_size = data_length - 1
    else:
        initial_train_size = '2000-02-19'
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = data_length - 1,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 2,
             allow_incomplete_fold = False
         )
    
    err_msg = re.escape(
        f"`{data_name}` must have at least `initial_train_size + gap + steps` "
        f"observations to create a minimum of one complete fold "
        f"(allow_incomplete_fold=False).\n"
        f"    Time series length: {data_length}\n"
        f"    Required >= {cv.initial_train_size + cv.gap + cv.steps}\n"
        f"    initial_train_size: {cv.initial_train_size}\n"
        f"    gap: {cv.gap}\n"
        f"    steps: {cv.steps}\n"
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series_dict_range,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterSarimax(regressor=Sarimax(order=(1, 1, 1))),
                          ForecasterEquivalentDate(offset=1, n_offsets=1)], 
                         ids = lambda fr: f'{type(fr).__name__}')
def test_check_backtesting_input_ValueError_Sarimax_Equivalent_when_initial_train_size_is_None(forecaster):
    """
    Test ValueError is raised in check_backtesting_input when initial_train_size 
    is None with a ForecasterSarimax or ForecasterEquivalentDate.
    """
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = None,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    err_msg = re.escape(
        f"`initial_train_size` must be an integer smaller than the "
        f"length of `y` ({len(y)})."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_NotFittedError_when_initial_train_size_None_and_forecaster_not_fitted():
    """
    Test NotFittedError is raised in check_backtesting_input when 
    initial_train_size is None and forecaster is not fitted.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = None,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    err_msg = re.escape(
        "`forecaster` must be already trained if no `initial_train_size` "
        "is provided."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_ValueError_when_initial_train_size_None_and_refit_True():
    """
    Test ValueError is raised in check_backtesting_input when initial_train_size 
    is None and refit is True.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    forecaster.is_fitted = True
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = None,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    err_msg = re.escape(
        "`refit` is only allowed when `initial_train_size` is not `None`."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_ValueError_when_skip_folds_in_ForecasterSarimax():
    """
    Test ValueError is raised in check_backtesting_input if `skip_folds` is
    used in ForecasterSarimax.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             skip_folds            = 2,
         )
    
    err_msg = re.escape(
        "`skip_folds` is not allowed for ForecasterSarimax. Set it to `None`."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


@pytest.mark.parametrize("boolean_argument", 
                         ['add_aggregated_metric', 'use_in_sample_residuals', 
                          'use_binned_residuals', 'return_predictors', 'show_progress', 
                          'suppress_warnings', 'suppress_warnings_fit'], 
                         ids = lambda argument: f'{argument}')
def test_check_backtesting_input_TypeError_when_boolean_arguments_not_bool(boolean_argument):
    """
    Test TypeError is raised in check_backtesting_input when boolean arguments 
    are not boolean.
    """
    forecaster = ForecasterRecursive(
                    regressor = Ridge(random_state=123),
                    lags      = 2
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    boolean_arguments = {
        'add_aggregated_metric': False,
        'use_in_sample_residuals': False,
        'use_binned_residuals': False,
        'return_predictors': False,
        'show_progress': False,
        'suppress_warnings': False,
        'suppress_warnings_fit': False
    }
    boolean_arguments[boolean_argument] = 'not_bool'
    
    err_msg = re.escape(f"`{boolean_argument}` must be a boolean: `True`, `False`.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster   = forecaster,
            cv           = cv,
            metric       = 'mean_absolute_error',
            y            = y,
            interval     = None,
            alpha        = None,
            n_boot       = 500,
            random_state = 123,
            **boolean_arguments
        )


@pytest.mark.parametrize("int_argument, value",
                         [('n_boot', 2.2), 
                          ('n_boot', -2),
                          ('random_state', 'not_int'),  
                          ('random_state', -3)], 
                         ids = lambda argument: f'{argument}')
def test_check_backtesting_input_TypeError_when_integer_args_not_int_or_greater_than_0(int_argument, value):
    """
    Test TypeError is raised in check_backtesting_input when integer arguments 
    are not int or are greater than 0.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    integer_arguments = {'n_boot': 500, 'random_state': 123}
    integer_arguments[int_argument] = value
    
    err_msg = re.escape(f"`{int_argument}` must be an integer greater than 0. Got {value}.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            interval                = None,
            alpha                   = None,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False,
            **integer_arguments
        )


@pytest.mark.parametrize("n_jobs", 
                         [1.0, 'not_int_auto'], 
                         ids = lambda value: f'n_jobs: {value}')
def test_check_backtesting_input_TypeError_when_n_jobs_not_int_or_auto(n_jobs):
    """
    Test TypeError is raised in check_backtesting_input when n_jobs  
    is not an integer or 'auto'.
    """
    forecaster = ForecasterRecursive(
                     regressor = Ridge(random_state=123),
                     lags      = 2
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    err_msg = re.escape(f"`n_jobs` must be an integer or `'auto'`. Got {n_jobs}.")
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            interval                = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            n_jobs                  = n_jobs,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_raises_when_interval_not_None_and_interval_method():
    """
    Test raises errors in check_backtesting_input when interval is not None
    and the forecaster uses bootstrapping or conformal.
    """
    forecaster = ForecasterRecursive(regressor=Ridge(), lags=2)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12)
    
    kwargs = {
        'forecaster': forecaster,
        'cv': cv,
        'metric': 'mean_absolute_error',
        'y': y,
        'interval': [10, 90],
        'interval_method': 'bootstrapping',
        'n_boot': 500,
        'use_in_sample_residuals': True,
        'use_binned_residuals': False,
        'random_state': 123,
        'show_progress': False,
        'suppress_warnings': False
    }

    kwargs['interval'] = {'10': 10, '90': 90}
    kwargs['interval_method'] = 'conformal'
    err_msg = re.escape(
        f"When `interval_method` is 'conformal', `interval` must "
        f"be a float or a list/tuple defining a symmetric interval. "
        f"Got {type(kwargs['interval'])}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(**kwargs)

    kwargs['interval'] = {'10': 10, '90': 90}
    kwargs['interval_method'] = 'bootstrapping'
    err_msg = re.escape(
        f"When `interval_method` is 'bootstrapping', `interval` "
        f"must be a float, a list or tuple of floats, a "
        f"scipy.stats distribution object (with methods `_pdf` and "
        f"`fit`) or the string 'bootstrapping'. Got {type(kwargs['interval'])}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(**kwargs)

    class CustomObject:  # pragma: no cover
        pass
    
    kwargs['interval'] = CustomObject()
    err_msg = re.escape(
        f"When `interval_method` is 'bootstrapping', `interval` "
        f"must be a float, a list or tuple of floats, a "
        f"scipy.stats distribution object (with methods `_pdf` and "
        f"`fit`) or the string 'bootstrapping'. Got {type(kwargs['interval'])}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(**kwargs)

    kwargs['interval'] = ['10', '90']
    err_msg = re.escape(
        f"`interval` must be a list or tuple of floats. "
        f"Got {type('10')} in {kwargs['interval']}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(**kwargs)

    kwargs['interval'] = [0, 100, 101]
    err_msg = re.escape(
        "When `interval` is a list or tuple, all values must be "
        "between 0 and 100 inclusive."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(**kwargs)

    kwargs['interval'] = 'not_bootstrapping'
    err_msg = re.escape(
        f"When `interval` is a string, it must be 'bootstrapping'."
        f"Got {kwargs['interval']}."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(**kwargs)

    kwargs['interval_method'] = 'not_bootstrapping_or_conformal'
    err_msg = re.escape(
        f"`interval_method` must be 'bootstrapping' or 'conformal'. "
        f"Got {kwargs['interval_method']}."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(**kwargs)


def test_check_backtesting_input_ValueError_when_return_predictors_and_forecaster_not_return_predictors():
    """
    Test ValueError is raised in check_backtesting_input when `return_predictors` is True
    and the forecaster is not an allowed type.
    """
    forecaster = ForecasterSarimax(
        regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
    )
    
    cv = TimeSeriesFold(
             steps              = 3,
             initial_train_size = len(y) - 12,
         )
    
    forecaster_name = type(forecaster).__name__
    forecasters_return_predictors = [
        "ForecasterRecursive",
        "ForecasterDirect",
        "ForecasterRecursiveMultiSeries",
        "ForecasterDirectMultiVariate"
    ]
    
    err_msg = re.escape(
        f"`return_predictors` is only allowed for forecasters of type "
        f"{forecasters_return_predictors}. Got {forecaster_name}."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            y                 = y,
            return_predictors = True,
            show_progress     = False,
            suppress_warnings = False
        )


@pytest.mark.parametrize("forecaster", 
                         [ForecasterDirect(regressor=Ridge(), lags=5, steps=5),
                          ForecasterDirectMultiVariate(regressor=Ridge(), level='l1', lags=5, steps=5)], 
                         ids = lambda fr: f'{type(fr).__name__}')
def test_check_backtesting_input_ValueError_when_Direct_forecaster_not_enough_steps(forecaster):
    """
    Test ValueError is raised in check_backtesting_input when there is not enough 
    steps to predict steps + gap in a Direct forecaster.
    """
    
    cv = TimeSeriesFold(
             steps                 = 1,
             initial_train_size    = len(y[:-12]),
             window_size           = None,
             differentiation       = None,
             refit                 = False,
             fixed_train_size      = True,
             gap                   = 5,
             skip_folds            = None,
             allow_incomplete_fold = True,
             return_all_indexes    = False,
         )
    
    err_msg = re.escape(
        f"When using a {type(forecaster).__name__}, the combination of steps "
        f"+ gap ({cv.steps + cv.gap}) cannot be greater than the `steps` parameter "
        f"declared when the forecaster is initialized ({forecaster.max_step})."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster        = forecaster,
            cv                = cv,
            metric            = 'mean_absolute_error',
            y                 = y,
            series            = series_wide_range,
            show_progress     = False,
            suppress_warnings = False
        )
