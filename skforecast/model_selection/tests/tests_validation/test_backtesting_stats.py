# Unit test backtesting_stats
# ==============================================================================
import re
import platform
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.stats import Sarimax, Arar, Ets, Arima
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterStats
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection import backtesting_stats
from skforecast.exceptions import IgnoredArgumentWarning

# Fixtures
from ....stats.tests.tests_arima.fixtures_arima import air_passengers
from ....recursive.tests.tests_forecaster_stats.fixtures_forecaster_stats import y_datetime
from ....recursive.tests.tests_forecaster_stats.fixtures_forecaster_stats import exog_datetime


def test_backtesting_forecaster_TypeError_when_forecaster_not_supported_types():
    """
    Test TypeError is raised in backtesting_forecaster if Forecaster is not one 
    of the types supported by the function.
    """
    forecaster = ForecasterRecursive(
                    estimator = Ridge(random_state=123),
                    lags      = 2
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )

    err_msg = re.escape(
        "`forecaster` must be of type `ForecasterStats`. For all other "
        "types of forecasters use the other functions available in the "
        "`model_selection` module."
    )
    with pytest.raises(TypeError, match = err_msg):
        backtesting_stats(
            forecaster    = forecaster,
            y             = y_datetime,
            cv            = cv,
            metric        = 'mean_absolute_error',
            exog          = None,
            alpha         = None,
            interval      = None,
            verbose       = False,
            show_progress = False
        )


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_stats_no_refit_no_exog_no_remainder_with_mocked(n_jobs):
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, no exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                       forecaster = forecaster,
                                       y          = y_datetime,
                                       cv         = cv,
                                       metric     = 'mean_squared_error',
                                       alpha      = None,
                                       interval   = None,
                                       n_jobs     = n_jobs,
                                       verbose    = True
                                   )
    
    expected_metric = pd.DataFrame({"mean_squared_error": [0.03683793335495359]})
    expected_preds = pd.DataFrame(
        data    = np.array([0.51853756, 0.5165776 , 0.51790214, 0.80140303, 0.84979734,
                            0.91918321, 0.84363512, 0.8804787 , 0.91651026, 0.42747836,
                            0.39041178, 0.23407875]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


@pytest.mark.parametrize("initial_train_size", 
                         [len(y_datetime) - 12, "2037-12-31"],
                         ids=lambda init: f'initial_train_size: {init}')
def test_output_backtesting_stats_no_refit_no_exog_remainder_with_mocked(initial_train_size):
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=5 (remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 5,
             initial_train_size    = initial_train_size,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                       forecaster = forecaster,
                                       y          = y_datetime,
                                       cv         = cv,
                                       metric     = 'mean_squared_error',
                                       alpha      = None,
                                       interval   = None,
                                       verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_squared_error": [0.07396344749165738]})
    expected_preds = pd.DataFrame(
        data    = np.array([0.51853756, 0.5165776 , 0.51790214, 0.51193703, 0.4991191 ,
                            0.89343704, 0.95023804, 1.00278782, 1.07322123, 1.13932909,
                            0.5673885 , 0.43713008]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_stats_yes_refit_no_exog_no_remainder_with_mocked(n_jobs):
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, 
    no exog, yes refit, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_squared_error'. (Mocked done with skforecast 0.7.0.)
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                       forecaster = forecaster,
                                       y          = y_datetime,
                                       cv         = cv,
                                       metric     = 'mean_squared_error',
                                       alpha      = None,
                                       interval   = None,
                                       n_jobs     = n_jobs,
                                       verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_squared_error": [0.038704200731126036]})
    expected_preds = pd.DataFrame(
        data    = np.array([0.51853756, 0.5165776 , 0.51790214, 0.80295192, 0.85238217,
                            0.9244119 , 0.84173367, 0.8793909 , 0.91329115, 0.42336972,
                            0.38434305, 0.2093133 ]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


@pytest.mark.parametrize("n_jobs", [1, -1, "auto"],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_stats_yes_refit_no_exog_remainder_with_mocked(n_jobs):
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, 
    no exog, yes refit, 12 observations to backtest, steps=5 (remainder), 
    metric='mean_squared_error'. (Mocked done with skforecast 0.7.0.)
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 5,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        metric     = 'mean_squared_error',
                                        alpha      = None,
                                        interval   = None,
                                        n_jobs     = n_jobs,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_squared_error": [0.0754085450012623]})
    expected_preds = pd.DataFrame(
        data    = np.array([0.51853756, 0.5165776 , 0.51790214, 0.51193703, 0.4991191 ,
                            0.89513678, 0.94913026, 1.00437767, 1.07534674, 1.14049886,
                            0.56289528, 0.40343592]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_stats_yes_refit_fixed_train_size_no_exog_no_remainder_with_mocked():
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, no exog, 
    yes refit, fixed_train_size yes, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_squared_error'. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        metric     = 'mean_squared_error',
                                        alpha      = None,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_squared_error": [0.04116499283290456]})
    expected_preds = pd.DataFrame(
        data    = np.array([0.51853756, 0.5165776 , 0.51790214, 0.80320348, 0.85236718,
                            0.92421562, 0.85060945, 0.88539784, 0.92172861, 0.41776604,
                            0.37024487, 0.1878739 ]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_stats_yes_refit_fixed_train_size_no_exog_remainder_with_mocked():
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, no exog, 
    yes refit, fixed_train_size yes, 12 observations to backtest, steps=5 (remainder), 
    metric='mean_squared_error'. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 5,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        metric     = 'mean_squared_error',
                                        alpha      = None,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_squared_error": [0.07571810495568278]})
    expected_preds = pd.DataFrame(
        data    = np.array([0.51853756, 0.5165776 , 0.51790214, 0.51193703, 0.4991191 ,
                            0.8959923 , 0.95147449, 1.00612185, 1.07723486, 1.14356597,
                            0.56321268, 0.40920121]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)
    

def test_output_backtesting_stats_no_refit_yes_exog_with_mocked():
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        exog       = exog_datetime,
                                        metric     = 'mean_squared_error',
                                        alpha      = None,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_squared_error": [0.18551856781581755]})
    expected_preds = pd.DataFrame(
        data    = np.array([ 0.59409098,  0.78323365,  0.99609033,  0.87882152,  1.02722143,
                             1.16176993,  0.85860472,  0.86636317,  0.68987477,  0.17788782,
                            -0.13577   , -0.50570715]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_stats_yes_refit_yes_exog_with_mocked():
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        exog       = exog_datetime,
                                        metric     = 'mean_squared_error',
                                        alpha      = None,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_squared_error": [0.198652574804823]})
    expected_preds = pd.DataFrame(
        data    = np.array([ 0.59409098,  0.78323365,  0.99609033,  0.8786089 ,  1.02218448,
                             1.15283534,  0.8597644 ,  0.87093769,  0.71221024,  0.16839089,
                            -0.16421948, -0.55386343]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_stats_yes_refit_fixed_train_size_yes_exog_with_mocked():
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=5 (remainder), 
    metric='mean_squared_error'. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 5,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        exog       = exog_datetime,
                                        metric     = 'mean_squared_error',
                                        alpha      = None,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_squared_error": [0.0917642049564646]})
    expected_preds = pd.DataFrame(
        data    = np.array([0.59409098, 0.78323365, 0.99609033, 1.21449931, 1.4574755 ,
                            0.89448353, 0.99712901, 1.05090061, 0.92362208, 0.76795064,
                            0.45382855, 0.26823527]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def my_metric(y_true, y_pred):  # pragma: no cover
    """
    Callable metric
    """
    metric = ((y_true - y_pred) / len(y_true)).mean()
    
    return metric


def test_output_backtesting_stats_no_refit_yes_exog_callable_metric_with_mocked():
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), callable metric. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        exog       = exog_datetime,
                                        metric     = my_metric,
                                        alpha      = None,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"my_metric": [0.007364452865679387]})
    expected_preds = pd.DataFrame(
        data    = np.array([ 0.59409098,  0.78323365,  0.99609033,  0.87882152,  1.02722143,
                             1.16176993,  0.85860472,  0.86636317,  0.68987477,  0.17788782,
                            -0.13577   , -0.50570715]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_stats_no_refit_no_exog_list_of_metrics_with_mocked():
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), list of metrics. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        metric     = [my_metric, 'mean_absolute_error'],
                                        alpha      = None,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"my_metric": [0.004423392707787538], 
                                    "mean_absolute_error": [0.1535720350789038]})
    expected_preds = pd.DataFrame(
        data    = np.array([0.51853756, 0.5165776 , 0.51790214, 0.80140303, 0.84979734,
                            0.91918321, 0.84363512, 0.8804787 , 0.91651026, 0.42747836,
                            0.39041178, 0.23407875]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_stats_yes_refit_no_exog_callable_metric_with_mocked():
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), callable metric. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        metric     = my_metric,
                                        alpha      = None,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"my_metric": [0.004644148042633733]})
    expected_preds = pd.DataFrame(
        data    = np.array([0.51853756, 0.5165776 , 0.51790214, 0.80295192, 0.85238217,
                            0.9244119 , 0.84173367, 0.8793909 , 0.91329115, 0.42336972,
                            0.38434305, 0.2093133 ]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_stats_yes_refit_fixed_train_size_yes_exog_list_of_metrics_with_mocked():
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=3 (no remainder), 
    list of metrics. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        exog       = exog_datetime,
                                        metric     = [my_metric, 'mean_absolute_scaled_error'],
                                        alpha      = None,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"my_metric": [0.007877420102652216], 
                                    "mean_absolute_scaled_error": [3.899618814139531]})
    expected_preds = pd.DataFrame(
        data    = np.array([ 0.59409098,  0.78323365,  0.99609033,  0.88202026,  1.03241114,
                             1.16808941,  0.86535534,  0.87277596,  0.69357041,  0.16628876,
                            -0.17189178, -0.56342057]),
        columns = ['pred'],
        index   = pd.date_range(start='2038', periods=12, freq='YE')
    )
    expected_preds.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)
    

@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values: f'alpha, interval: {values}')
def test_output_backtesting_stats_no_refit_yes_exog_interval_with_mocked(alpha, interval):
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_absolute_error',
    interval. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        exog       = exog_datetime,
                                        metric     = ['mean_absolute_error'],
                                        alpha      = alpha,
                                        interval   = interval,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_absolute_error": [0.3040748056175932]})
    expected_values = np.array([[ 0.59409098, -1.08941968,  2.27760163],
                                [ 0.78323365, -2.51570095,  4.08216824],
                                [ 0.99609033, -4.02240666,  6.01458732],
                                [ 0.87882152, -0.80468913,  2.56233218],
                                [ 1.02722143, -2.27171317,  4.32615603],
                                [ 1.16176993, -3.85672705,  6.18026692],
                                [ 0.85860472, -0.82490594,  2.54211537],
                                [ 0.86636317, -2.43257143,  4.16529777],
                                [ 0.68987477, -4.32862221,  5.70837176],
                                [ 0.17788782, -1.50562283,  1.86139848],
                                [-0.13577   , -3.4347046 ,  3.16316459],
                                [-0.50570715, -5.52420413,  4.51278984]])

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred', 'lower_bound', 'upper_bound'],
                                        index   = pd.date_range(start='2038', periods=12, freq='YE')
                                    )  
    expected_backtest_predictions.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])                                                   

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values: f'alpha, interval: {values}')
def test_output_backtesting_stats_yes_refit_yes_exog_interval_with_mocked(alpha, interval):
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_absolute_error', interval. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        exog       = exog_datetime,
                                        metric     = 'mean_absolute_error',
                                        alpha      = alpha,
                                        interval   = interval,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_absolute_error": [0.31145145397674484]})
    expected_values = np.array([[ 0.59409098, -1.08941968,  2.27760163],
                                [ 0.78323365, -2.51570095,  4.08216824],
                                [ 0.99609033, -4.02240666,  6.01458732],
                                [ 0.8786089 , -0.8166216 ,  2.57383939],
                                [ 1.02218448, -2.25140783,  4.29577679],
                                [ 1.15283534, -3.7591572 ,  6.06482787],
                                [ 0.8597644 , -0.84109595,  2.56062475],
                                [ 0.87093769, -2.39317559,  4.13505098],
                                [ 0.71221024, -4.21947832,  5.64389879],
                                [ 0.16839089, -1.52752939,  1.86431116],
                                [-0.16421948, -3.50117442,  3.17273546],
                                [-0.55386343, -5.6847384 ,  4.57701154]])

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred', 'lower_bound', 'upper_bound'],
                                        index   = pd.date_range(start='2038', periods=12, freq='YE')
                                    )  
    expected_backtest_predictions.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])                                                     

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values: f'alpha, interval: {values}')
def test_output_backtesting_stats_yes_refit_fixed_train_size_yes_exog_interval_with_mocked(alpha, interval):
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_absolute_error', interval. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                       forecaster = forecaster,
                                       y          = y_datetime,
                                       cv         = cv,
                                       exog       = exog_datetime,
                                       metric     = ['mean_absolute_error'],
                                       alpha      = alpha,
                                       interval   = interval,
                                       verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_absolute_error": [0.31329767191681507]})
    expected_values = np.array([[ 0.59409098, -1.08941968,  2.27760163],
                                [ 0.78323365, -2.51570095,  4.08216824],
                                [ 0.99609033, -4.02240666,  6.01458732],
                                [ 0.88202026, -0.79712278,  2.56116331],
                                [ 1.03241114, -2.26295601,  4.32777829],
                                [ 1.16808941, -3.82752621,  6.16370504],
                                [ 0.86535534, -0.82088676,  2.55159744],
                                [ 0.87277596, -2.39786807,  4.14342   ],
                                [ 0.69357041, -4.32350585,  5.71064667],
                                [ 0.16628876, -1.53370994,  1.86628747],
                                [-0.17189178, -3.5242321 ,  3.18044855],
                                [-0.56342057, -5.74734656,  4.62050543]])

    expected_backtest_predictions = pd.DataFrame(
                                        data    = expected_values,
                                        columns = ['pred', 'lower_bound', 'upper_bound'],
                                        index   = pd.date_range(start='2038', periods=12, freq='YE')
                                    )
    expected_backtest_predictions.insert(0, 'fold', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])                                                    

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)


def test_output_backtesting_stats_fold_stride_with_mocked():
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=3 (no remainder), 
    fold stride. Mocked done with skforecast 0.18.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 6,
             initial_train_size    = len(y_datetime) - 12,
             fold_stride           = 3,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        exog       = exog_datetime,
                                        metric     = 'mean_absolute_error',
                                        alpha      = None,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_absolute_error": [0.313297672064354]})
    expected_preds = pd.DataFrame(
        data    = np.array([
                      [0.        ,  0.59409098],
                      [0.        ,  0.78323365],
                      [0.        ,  0.99609033],
                      [0.        ,  1.21449931],
                      [0.        ,  1.4574755 ],
                      [0.        ,  1.68696099],
                      [1.        ,  0.88202026],
                      [1.        ,  1.03241114],
                      [1.        ,  1.16808941],
                      [1.        ,  1.34390467],
                      [1.        ,  1.4696968 ],
                      [1.        ,  1.41286359],
                      [2.        ,  0.86535534],
                      [2.        ,  0.87277596],
                      [2.        ,  0.69357041],
                      [2.        ,  0.48557182],
                      [2.        ,  0.28762654],
                      [2.        ,  0.05433193],
                      [3.        ,  0.16628876],
                      [3.        , -0.17189178],
                      [3.        , -0.56342057]]),
        columns = ['fold', 'pred'],
        index   = pd.DatetimeIndex([
                    '2038-12-31', '2039-12-31', '2040-12-31', '2041-12-31',
                    '2042-12-31', '2043-12-31', '2041-12-31', '2042-12-31',
                    '2043-12-31', '2044-12-31', '2045-12-31', '2046-12-31',
                    '2044-12-31', '2045-12-31', '2046-12-31', '2047-12-31',
                    '2048-12-31', '2049-12-31', '2047-12-31', '2048-12-31',
                    '2049-12-31'
                ])
    ).astype({'fold': int})

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_stats_fold_stride_greater_than_steps_with_mocked():
    """
    Test output of backtesting_stats with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=3 (no remainder), 
    fold stride greater than steps. Mocked done with skforecast 0.18.0.
    """
    forecaster = ForecasterStats(
                     estimator = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 2,
             initial_train_size    = len(y_datetime) - 12,
             fold_stride           = 4,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        exog       = exog_datetime,
                                        metric     = 'mean_absolute_error',
                                        alpha      = None,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({"mean_absolute_error": [0.080745243332592]})
    expected_preds = pd.DataFrame(
        data    = np.array([
                      [0.        , 0.59409098],
                      [0.        , 0.78323365],
                      [1.        , 0.76895405],
                      [1.        , 0.81593985],
                      [2.        , 0.77460313],
                      [2.        , 0.5839772 ]]),
        columns = ['fold', 'pred'],
        index   = pd.DatetimeIndex([
                    '2038-12-31', '2039-12-31', '2042-12-31', '2043-12-31',
                    '2046-12-31', '2047-12-31'
                ])
    ).astype({'fold': int})

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


@pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason="Ets optimizer converges to different local minima on macOS"
)
def test_output_backtesting_stats_multiple_estimators_refit_False_with_mocked():
    """
    Test output of backtesting_stats with multiple estimators, refit=False,
    12 observations to backtest, steps=3, metric='mean_squared_error'.
    Note: Estimators different from Sarimax require refitting, so refit is
    internally set to True regardless of the value provided.
    """
    estimators = [
        Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False),
        Arar(),
        Ets(),
        Arima(order=(1, 1, 1))
    ]
    forecaster = ForecasterStats(estimator=estimators)
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    warn_msg = re.escape(
        "Estimators different from `skforecast.stats.Sarimax` require refitting "
        "since predictions must start from the end of the training set. `refit` "
        "is set to `True`, regardless of the value provided."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        metric, backtest_predictions = backtesting_stats(
                                            forecaster = forecaster,
                                            y          = y_datetime,
                                            cv         = cv,
                                            metric     = 'mean_squared_error',
                                            alpha      = None,
                                            interval   = None,
                                            verbose    = False
                                       )
    
    expected_metric = pd.DataFrame({
        'estimator_id': ['skforecast.Sarimax', 'skforecast.Arar', 
                         'skforecast.Ets', 'skforecast.Arima'],
        'mean_squared_error': [0.03870420, 0.00075339, 0.02130093, 0.01233176]
    })
    
    expected_preds = pd.DataFrame(
        data = {
            'fold': [0]*4 + [0]*4 + [0]*4 + [1]*4 + [1]*4 + [1]*4 + 
                    [2]*4 + [2]*4 + [2]*4 + [3]*4 + [3]*4 + [3]*4,
            'estimator_id': ['skforecast.Sarimax', 'skforecast.Arar', 
                             'skforecast.Ets', 'skforecast.Arima'] * 12,
            'pred': np.array([
                0.51853756, 0.59008805, 0.55841927, 0.56981577,
                0.5165776 , 0.6215785 , 0.558423  , 0.57511248,
                0.51790214, 0.72728926, 0.55842598, 0.57838768,
                0.80295192, 0.74408697, 0.74574166, 0.74604025,
                0.85238217, 0.75550972, 0.75722036, 0.74315278,
                0.9244119 , 0.82159729, 0.76846949, 0.74475521,
                0.84173367, 0.95293211, 0.82573018, 0.80033296,
                0.8793909 , 0.93913951, 0.83668506, 0.80841525,
                0.91329115, 0.5480628 , 0.84742083, 0.80363893,
                0.42336972, 0.6195116 , 0.75333365, 0.56392292,
                0.38434305, 0.58648398, 0.75541092, 0.58633461,
                0.2093133 , 0.62192848, 0.75738421, 0.60061975
            ])
        },
        index = pd.DatetimeIndex(
            ['2038-12-31']*4 + ['2039-12-31']*4 + ['2040-12-31']*4 +
            ['2041-12-31']*4 + ['2042-12-31']*4 + ['2043-12-31']*4 +
            ['2044-12-31']*4 + ['2045-12-31']*4 + ['2046-12-31']*4 +
            ['2047-12-31']*4 + ['2048-12-31']*4 + ['2049-12-31']*4
        )
    )

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


@pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason="Ets optimizer converges to different local minima on macOS"
)
def test_output_backtesting_stats_multiple_estimators_refit_True_with_mocked():
    """
    Test output of backtesting_stats with multiple estimators, refit=True,
    12 observations to backtest, steps=3, metric='mean_squared_error'.
    """
    estimators = [
        Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False),
        Arar(),
        Ets(),
        Arima(order=(1, 1, 1))
    ]
    forecaster = ForecasterStats(estimator=estimators)
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        metric     = 'mean_squared_error',
                                        alpha      = None,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    expected_metric = pd.DataFrame({
        'estimator_id': ['skforecast.Sarimax', 'skforecast.Arar', 
                         'skforecast.Ets', 'skforecast.Arima'],
        'mean_squared_error': [0.03870420, 0.00075339, 0.02130093, 0.01233176]
    })
    
    expected_preds = pd.DataFrame(
        data = {
            'fold': [0]*4 + [0]*4 + [0]*4 + [1]*4 + [1]*4 + [1]*4 + 
                    [2]*4 + [2]*4 + [2]*4 + [3]*4 + [3]*4 + [3]*4,
            'estimator_id': ['skforecast.Sarimax', 'skforecast.Arar', 
                             'skforecast.Ets', 'skforecast.Arima'] * 12,
            'pred': np.array([
                0.51853756, 0.59008805, 0.55841927, 0.56981577,
                0.5165776 , 0.6215785 , 0.558423  , 0.57511248,
                0.51790214, 0.72728926, 0.55842598, 0.57838768,
                0.80295192, 0.74408697, 0.74574166, 0.74604025,
                0.85238217, 0.75550972, 0.75722036, 0.74315278,
                0.9244119 , 0.82159729, 0.76846949, 0.74475521,
                0.84173367, 0.95293211, 0.82573018, 0.80033296,
                0.8793909 , 0.93913951, 0.83668506, 0.80841525,
                0.91329115, 0.5480628 , 0.84742083, 0.80363893,
                0.42336972, 0.6195116 , 0.75333365, 0.56392292,
                0.38434305, 0.58648398, 0.75541092, 0.58633461,
                0.2093133 , 0.62192848, 0.75738421, 0.60061975
            ])
        },
        index = pd.DatetimeIndex(
            ['2038-12-31'] * 4 + ['2039-12-31'] * 4 + ['2040-12-31'] * 4 +
            ['2041-12-31'] * 4 + ['2042-12-31'] * 4 + ['2043-12-31'] * 4 +
            ['2044-12-31'] * 4 + ['2045-12-31'] * 4 + ['2046-12-31'] * 4 +
            ['2047-12-31'] * 4 + ['2048-12-31'] * 4 + ['2049-12-31'] * 4
        )
    )

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_stats_multiple_estimators_refit_False_interval_with_mocked():
    """
    Test output of backtesting_stats with multiple estimators, refit=False,
    12 observations to backtest, steps=3, metric='mean_squared_error', interval.
    Note: Estimators different from Sarimax require refitting, so refit is
    internally set to True regardless of the value provided.
    This test verifies the structure and columns of the output, but not exact
    interval values due to numerical variability in statistical models.
    """
    estimators = [
        Sarimax(order=(2, 1, 0)),
        Arar(),
        Ets(model='ANN'),
        Arima(order=(1, 1, 0))
    ]
    forecaster = ForecasterStats(estimator=estimators)
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    warn_msg = re.escape(
        "Estimators different from `skforecast.stats.Sarimax` require refitting "
        "since predictions must start from the end of the training set. `refit` "
        "is set to `True`, regardless of the value provided."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        metric, backtest_predictions = backtesting_stats(
                                            forecaster = forecaster,
                                            y          = y_datetime,
                                            cv         = cv,
                                            metric     = 'mean_squared_error',
                                            alpha      = 0.05,
                                            interval   = None,
                                            verbose    = False
                                       )
    
    # Check metric structure
    assert list(metric.columns) == ['estimator_id', 'mean_squared_error']
    assert len(metric) == 4
    assert list(metric['estimator_id']) == [
        'skforecast.Sarimax', 'skforecast.Arar', 
        'skforecast.Ets', 'skforecast.Arima'
    ]
    
    # Check predictions structure
    assert list(backtest_predictions.columns) == [
        'fold', 'estimator_id', 'pred', 'lower_bound', 'upper_bound'
    ]
    assert len(backtest_predictions) == 48  # 12 timesteps x 4 estimators
    assert set(backtest_predictions['estimator_id']) == {
        'skforecast.Sarimax', 'skforecast.Arar', 
        'skforecast.Ets', 'skforecast.Arima'
    }
    
    # Check that lower_bound < pred < upper_bound
    assert (backtest_predictions['lower_bound'] < backtest_predictions['pred']).all()
    assert (backtest_predictions['pred'] < backtest_predictions['upper_bound']).all()


def test_output_backtesting_stats_multiple_estimators_refit_True_interval_with_mocked():
    """
    Test output of backtesting_stats with multiple estimators, refit=True,
    12 observations to backtest, steps=3, metric='mean_squared_error', interval.
    This test verifies the structure and columns of the output, but not exact
    interval values due to numerical variability in statistical models.
    """
    estimators = [
        Sarimax(order=(2, 1, 0)),
        Arar(),
        Ets(model='ANN'),
        Arima(order=(1, 1, 0))
    ]
    forecaster = ForecasterStats(estimator=estimators)
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster = forecaster,
                                        y          = y_datetime,
                                        cv         = cv,
                                        metric     = 'mean_squared_error',
                                        alpha      = 0.05,
                                        interval   = None,
                                        verbose    = False
                                   )
    
    # Check metric structure
    assert list(metric.columns) == ['estimator_id', 'mean_squared_error']
    assert len(metric) == 4
    assert list(metric['estimator_id']) == [
        'skforecast.Sarimax', 'skforecast.Arar', 
        'skforecast.Ets', 'skforecast.Arima'
    ]
    
    # Check predictions structure
    assert list(backtest_predictions.columns) == [
        'fold', 'estimator_id', 'pred', 'lower_bound', 'upper_bound'
    ]
    assert len(backtest_predictions) == 48  # 12 timesteps x 4 estimators
    assert set(backtest_predictions['estimator_id']) == {
        'skforecast.Sarimax', 'skforecast.Arar', 
        'skforecast.Ets', 'skforecast.Arima'
    }
    
    # Check that lower_bound < pred < upper_bound
    assert (backtest_predictions['lower_bound'] < backtest_predictions['pred']).all()
    assert (backtest_predictions['pred'] < backtest_predictions['upper_bound']).all()


@pytest.mark.skipif(
    platform.system() == 'Darwin',
    reason="ARIMA optimizer converges to different values on macOS"
)
def test_output_backtesting_stats_auto_arima_arar_freeze_params_False_gap_air_passengers():
    """
    Test output of backtesting_stats with Arima in auto mode (order=None, 
    seasonal_order=None) and Arar estimators, freeze_params=False, gap=2.
    """

    arima_model = Arima(
        order=None,
        seasonal_order=None,
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        max_P=1,
        max_Q=1,
        max_order=5,
        max_d=1,
        max_D=1,
        ic="aic",
        seasonal=True,
        test="kpss",
        nmodels=94,
        optim_method="BFGS",
        m=12,
        trace=False,
        stepwise=True,
    )

    forecaster = ForecasterStats(estimator=[arima_model, Arar()])
    
    cv = TimeSeriesFold(
             steps                 = 4,
             initial_train_size    = len(air_passengers) - 25,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 2,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_stats(
                                        forecaster        = forecaster,
                                        y                 = air_passengers,
                                        cv                = cv,
                                        metric            = 'mean_squared_error',
                                        freeze_params     = False,
                                        verbose           = False,
                                        suppress_warnings = True
                                   )

    if platform.system() == 'Windows':
        pred = np.array([
            324.39056041, 340.10229308, 368.36594786, 395.8690001 ,
            354.37194124, 383.12152568, 369.3704818 , 400.36513771,
            470.68571171, 489.47008406, 525.13830204, 550.55008119,
            537.96708878, 564.55998644, 436.08061344, 462.02373038,
            399.01574871, 407.75589214, 348.60372625, 353.68722631,
            372.54122203, 386.24074229, 394.33853139, 410.67276845,
            381.54091948, 385.7882863 , 443.89508682, 453.40807421,
            431.97608402, 446.68136059, 454.94222745, 472.61938719,
            495.10233702, 518.4513859 , 573.05240108, 600.48436578,
            585.55331424, 616.97132163, 490.70852774, 511.4534994 ,
            478.39345005, 456.12460312, 433.48247753, 405.13228127,
            476.45577472, 444.53909803
        ])
        mse_expected = np.array([880.057541, 180.798112])
    else:
        # Linux
        pred = np.array([
            324.39056041, 340.10229308, 368.36594786, 395.8690001 ,
            354.37194124, 383.12152568, 369.3704818 , 400.36513771,
            470.68571171, 489.47008406, 525.13830204, 550.55008119,
            537.96708878, 564.55998644, 436.08061344, 462.02373038,
            399.01574871, 407.75589214, 348.60372625, 353.68722631,
            372.54122203, 386.24074229, 394.33853139, 410.67276845,
            381.54091948, 385.7882863 , 443.89508682, 453.40807421,
            431.97608402, 446.68136059, 454.94222745, 472.61938719,
            495.10233702, 518.4513859 , 573.05240108, 600.48436578,
            585.55331424, 616.97132163, 490.70852774, 511.4534994 ,
            478.39345005, 456.12460312, 433.48247753, 405.13228127,
            476.45577472, 444.53909803
        ])
        mse_expected = np.array([880.0986434976576, 180.79811204373485])
    
    expected_metric = pd.DataFrame({
        'estimator_id': ['skforecast.Arima', 'skforecast.Arar'],
        'mean_squared_error': mse_expected
    })
    
    expected_preds = pd.DataFrame(
        data = {
            'fold': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                     2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
                     5, 5],
            'estimator_id': ['skforecast.Arima', 'skforecast.Arar', 'skforecast.Arima',
                             'skforecast.Arar', 'skforecast.Arima', 'skforecast.Arar',
                             'skforecast.Arima', 'skforecast.Arar', 'skforecast.Arima',
                             'skforecast.Arar', 'skforecast.Arima', 'skforecast.Arar',
                             'skforecast.Arima', 'skforecast.Arar', 'skforecast.Arima',
                             'skforecast.Arar', 'skforecast.Arima', 'skforecast.Arar',
                             'skforecast.Arima', 'skforecast.Arar', 'skforecast.Arima',
                             'skforecast.Arar', 'skforecast.Arima', 'skforecast.Arar',
                             'skforecast.Arima', 'skforecast.Arar', 'skforecast.Arima',
                             'skforecast.Arar', 'skforecast.Arima', 'skforecast.Arar',
                             'skforecast.Arima', 'skforecast.Arar', 'skforecast.Arima',
                             'skforecast.Arar', 'skforecast.Arima', 'skforecast.Arar',
                             'skforecast.Arima', 'skforecast.Arar', 'skforecast.Arima',
                             'skforecast.Arar', 'skforecast.Arima', 'skforecast.Arar',
                             'skforecast.Arima', 'skforecast.Arar', 'skforecast.Arima',
                             'skforecast.Arar'],
            'pred': pred,
            'estimator_params': [
                'AutoArima(1,1,0)(0,1,0)[12]', 'AutoArima(1,1,0)(0,1,0)[12]',
                'AutoArima(1,1,0)(0,1,0)[12]', 'AutoArima(1,1,0)(0,1,0)[12]',
                'Arar(lags=(1, 2, 12, 13))', 'Arar(lags=(1, 2, 12, 13))',
                'Arar(lags=(1, 2, 12, 13))', 'Arar(lags=(1, 2, 12, 13))',
                'AutoArima(1,0,0)(0,1,0)[12]', 'AutoArima(1,0,0)(0,1,0)[12]',
                'AutoArima(1,0,0)(0,1,0)[12]', 'AutoArima(1,0,0)(0,1,0)[12]',
                'Arar(lags=(1, 2, 12, 13))', 'Arar(lags=(1, 2, 12, 13))',
                'Arar(lags=(1, 2, 12, 13))', 'Arar(lags=(1, 2, 12, 13))',
                'AutoArima(3,0,0)(0,1,0)[12]', 'AutoArima(3,0,0)(0,1,0)[12]',
                'AutoArima(3,0,0)(0,1,0)[12]', 'AutoArima(3,0,0)(0,1,0)[12]',
                'Arar(lags=(1, 2, 9, 10))', 'Arar(lags=(1, 2, 9, 10))',
                'Arar(lags=(1, 2, 9, 10))', 'Arar(lags=(1, 2, 9, 10))',
                'AutoArima(3,0,0)(0,1,0)[12]', 'AutoArima(3,0,0)(0,1,0)[12]',
                'AutoArima(3,0,0)(0,1,0)[12]', 'AutoArima(3,0,0)(0,1,0)[12]',
                'Arar(lags=(1, 2, 10, 13))', 'Arar(lags=(1, 2, 10, 13))',
                'Arar(lags=(1, 2, 10, 13))', 'Arar(lags=(1, 2, 10, 13))',
                'AutoArima(1,0,0)(0,1,0)[12]', 'AutoArima(1,0,0)(0,1,0)[12]',
                'AutoArima(1,0,0)(0,1,0)[12]', 'AutoArima(1,0,0)(0,1,0)[12]',
                'Arar(lags=(1, 2, 9, 10))', 'Arar(lags=(1, 2, 9, 10))',
                'Arar(lags=(1, 2, 9, 10))', 'Arar(lags=(1, 2, 9, 10))',
                'AutoArima(1,1,0)(0,1,0)[12]', 'AutoArima(1,1,0)(0,1,0)[12]',
                'AutoArima(1,1,0)(0,1,0)[12]', 'Arar(lags=(1, 2, 9, 10))',
                'Arar(lags=(1, 2, 9, 10))', 'Arar(lags=(1, 2, 9, 10))'
            ]
        },
        index = pd.Index([121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 126, 126, 127, 127,
                          128, 128, 129, 129, 130, 130, 131, 131, 132, 132, 133, 133, 134, 134,
                          135, 135, 136, 136, 137, 137, 138, 138, 139, 139, 140, 140, 141, 141,
                          142, 142, 143, 143])
    )

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)
