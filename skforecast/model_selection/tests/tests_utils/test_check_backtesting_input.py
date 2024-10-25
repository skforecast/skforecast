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
from skforecast.model_selection.tests.fixtures_model_selection_multiseries import series


def test_check_backtesting_input_TypeError_when_cv_not_TimeSeries_Fold():
    """
    Test TypeError is raised in check_backtesting_input if `cv` is not a
    TimeSeriesFold object.
    """
    forecaster = ForecasterRecursive(regressor=Ridge(), lags=2)
    y = pd.Series(np.arange(50))
    y.index = pd.date_range(start='2000-01-01', periods=len(y), freq='D')
    
    class BadCv():
        pass

    err_msg = re.escape("`cv` must be a TimeSeriesFold object. Got BadCv.")
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
def test_check_backtesting_input_TypeError_when_y_is_not_pandas_Series_uniseries(forecaster):
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
def test_check_backtesting_input_TypeError_when_series_is_not_pandas_DataFrame_multiseries(forecaster):
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


def test_check_backtesting_input_TypeError_when_series_is_not_pandas_DataFrame_multiseries_dict():
    """
    Test TypeError is raised in check_backtesting_input if `series` is not a 
    pandas DataFrame in forecasters multiseries with dict.
    """
    forecaster = ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)
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

    err_msg = re.escape(
        (f"`series` must be a pandas DataFrame or a dict of DataFrames or Series. "
         f"Got {type(bad_series)}.")
    )
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


def test_check_backtesting_input_TypeError_when_series_is_dict_of_pandas_Series_multiseries_dict():
    """
    Test TypeError is raised in check_backtesting_input if `series` is not a 
    dict of pandas Series in forecasters multiseries with dict.
    """
    forecaster = ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)
    bad_series = {'l1': np.arange(50)}
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(bad_series['l1'][:-12]),
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             verbose               = False
         )

    err_msg = re.escape(
        ("If `series` is a dictionary, all series must be a named "
         "pandas Series or a pandas DataFrame with a single column. "
         "Review series: ['l1']")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = None,
            series                  = bad_series,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_ValueError_when_series_is_dict_no_DatetimeIndex_multiseries_dict():
    """
    Test ValueError is raised in check_backtesting_input if `series` is a 
    dict with pandas Series with no DatetimeIndex in forecasters 
    multiseries with dict.
    """
    forecaster = ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)
    series_dict = {
        'l1': pd.Series(np.arange(50)),
        'l2': pd.Series(np.arange(50))
    }
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(series_dict['l1'][:-12]),
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             verbose               = False
         )

    err_msg = re.escape(
        ("If `series` is a dictionary, all series must have a Pandas DatetimeIndex "
         "as index with the same frequency. Review series: ['l1', 'l2']")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            series                  = series_dict,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_ValueError_when_series_is_dict_diff_freq_multiseries_dict():
    """
    Test ValueError is raised in check_backtesting_input if `series` is a 
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
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(series_dict['l1'][:-12]),
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             verbose               = False
         )

    err_msg = re.escape(
        ("If `series` is a dictionary, all series must have a Pandas DatetimeIndex "
         "as index with the same frequency. Found frequencies: ['<Day>', '<MonthBegin>']")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            series                  = series_dict,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_TypeError_when_not_valid_exog_type_multiseries_dict():
    """
    Test TypeError is raised in check_backtesting_input if `exog` is not a
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
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(series_dict['l1'][:-12]),
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             verbose               = False
         )

    err_msg = re.escape(
        (f"`exog` must be a pandas Series, DataFrame, dictionary of pandas "
         f"Series/DataFrames or None. Got {type(bad_exog)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            series                  = series_dict,
            exog                    = bad_exog,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )


def test_check_backtesting_input_TypeError_when_not_valid_exog_dict_type_multiseries_dict():
    """
    Test TypeError is raised in check_backtesting_input if `exog` is not a
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
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(series_dict['l1'][:-12]),
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             verbose               = False
         )

    err_msg = re.escape(
        ("If `exog` is a dictionary, All exog must be a named pandas "
         "Series, a pandas DataFrame or None. Review exog: ['l1']")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            series                  = series_dict,
            exog                    = bad_exog,
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
        (f"`exog` must be a pandas Series, DataFrame or None. Got {type(bad_exog)}.")
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


def test_check_backtesting_input_ValueError_when_forecaster_diff_not_cv_diff():
    """
    Test ValueError is raised in check_backtesting_input if `differentiation`
    of the forecaster is different from `differentiation` of the cv.
    """
    forecaster = ForecasterRecursive(
                     regressor       = Ridge(random_state=123),
                     lags            = 2,
                     differentiation = 2
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y[:-12]),
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True,
             differentiation       = 1,
             verbose               = False
         )
    
    err_msg = re.escape(
        ("The differentiation included in the forecaster "
         "(2) differs from the differentiation "
         "included in the cv (1). Set the same value "
         "for both.")
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
        (f"`metric` must be a string, a callable function, or a list containing "
         f"multiple strings and/or callables. Got {type(metric)}.")
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
        (f"`initial_train_size` must be an integer greater than "
         f"the `window_size` of the forecaster ({forecaster.window_size}) "
         f"and smaller than the length of `{data_name}` ({data_length}).")
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
                         ['greater', 'smaller'], 
                         ids = lambda initial: f'initial_train_size: {initial}')
@pytest.mark.parametrize("forecaster", 
                         [ForecasterRecursive(regressor=Ridge(), lags=2),
                          ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_backtesting_input_ValueError_when_initial_train_size_not_correct_value(initial_train_size, forecaster):
    """
    Test ValueError is raised in check_backtesting_input when 
    initial_train_size >= length `y` or `series` or initial_train_size < window_size.
    """
    if type(forecaster).__name__ == 'ForecasterRecursive':
        data_length = len(y)
        data_name = 'y'
    else:
        data_length = len(series)
        data_name = 'series'

    if initial_train_size == 'greater':
        initial_train_size = data_length
    else:
        initial_train_size = forecaster.window_size - 1
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = initial_train_size,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    err_msg = re.escape(
        (f"If used, `initial_train_size` must be an integer greater than "
         f"the `window_size` of the forecaster ({forecaster.window_size}) "
         f"and smaller than the length of `{data_name}` ({data_length}).")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series,
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
                          ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_backtesting_input_ValueError_when_initial_train_size_plus_gap_less_than_data_length(forecaster):
    """
    Test ValueError is raised in check_backtesting_input when 
    initial_train_size + gap >= length `y` or `series` depending on the forecaster.
    """
    if type(forecaster).__name__ == 'ForecasterRecursive':
        data_length = len(y)
        data_name = 'y'
    else:
        data_length = len(series)
        data_name = 'series'
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = data_length - 1,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 2,
             allow_incomplete_fold = True
         )
    
    err_msg = re.escape(
        (f"The combination of initial_train_size {cv.initial_train_size} and "
         f"gap {cv.gap} cannot be greater than the length of `{data_name}` "
         f"({data_length}).")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series,
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
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
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
        (f"`initial_train_size` must be an integer smaller than the "
         f"length of `y` ({len(y)}).")
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
        ("`forecaster` must be already trained if no `initial_train_size` "
         "is provided.")
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
                          'use_binned_residuals', 'show_progress', 
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
def test_check_backtesting_input_TypeError_when_integer_arguments_not_int_or_greater_than_0(int_argument, value):
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
    
    integer_arguments = {'n_boot': 500,
                         'random_state': 123}
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
    
    err_msg = re.escape(
        (f"`n_jobs` must be an integer or `'auto'`. Got {n_jobs}.")
    )
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


@pytest.mark.parametrize("forecaster", 
                         [ForecasterRecursive(regressor=Ridge(), lags=2),
                          ForecasterRecursiveMultiSeries(regressor=Ridge(), lags=2)], 
                         ids = lambda fr: f'forecaster: {type(fr).__name__}')
def test_check_backtesting_input_ValueError_when_not_enough_data_to_create_a_fold(forecaster):
    """
    Test ValueError is raised in check_backtesting_input when there is not enough 
    data to evaluate even single fold because `allow_incomplete_fold` = `False`.
    """
    if type(forecaster).__name__ == 'ForecasterRecursive':
        data_length = len(y)
    else:
        data_length = len(series)
    
    cv = TimeSeriesFold(
             steps                 = 5,
             initial_train_size    = data_length - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 10,
             allow_incomplete_fold = False
         )
    
    err_msg = re.escape(
        (f"There is not enough data to evaluate {cv.steps} steps in a single "
         f"fold. Set `allow_incomplete_fold` to `True` to allow incomplete folds.\n"
         f"    Data available for test : {data_length - (cv.initial_train_size + cv.gap)}\n"
         f"    Steps                   : {cv.steps}")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_backtesting_input(
            forecaster              = forecaster,
            cv                      = cv,
            metric                  = 'mean_absolute_error',
            y                       = y,
            series                  = series,
            interval                = None,
            alpha                   = None,
            n_boot                  = 500,
            random_state            = 123,
            use_in_sample_residuals = True,
            show_progress           = False,
            suppress_warnings       = False
        )
