# Unit test backtesting_sarimax
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterSarimax
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection import backtesting_sarimax

# Fixtures
from ....recursive.tests.tests_forecaster_sarimax.fixtures_forecaster_sarimax import y_datetime
from ....recursive.tests.tests_forecaster_sarimax.fixtures_forecaster_sarimax import exog_datetime


def test_backtesting_forecaster_TypeError_when_forecaster_not_supported_types():
    """
    Test TypeError is raised in backtesting_forecaster if Forecaster is not one 
    of the types supported by the function.
    """
    forecaster = ForecasterRecursive(
                    regressor = Ridge(random_state=123),
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
        ("`forecaster` must be of type `ForecasterSarimax`, for all other "
         "types of forecasters use the functions available in the other "
         "`model_selection` modules.")
    )
    with pytest.raises(TypeError, match = err_msg):
        backtesting_sarimax(
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
def test_output_backtesting_sarimax_no_refit_no_exog_no_remainder_with_mocked(n_jobs):
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_no_refit_no_exog_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=5 (remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 5,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_sarimax_yes_refit_no_exog_no_remainder_with_mocked(n_jobs):
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, 
    no exog, yes refit, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_squared_error'. (Mocked done with skforecast 0.7.0.)
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


@pytest.mark.parametrize("n_jobs", [1, -1, "auto"],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_sarimax_yes_refit_no_exog_remainder_with_mocked(n_jobs):
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, 
    no exog, yes refit, 12 observations to backtest, steps=5 (remainder), 
    metric='mean_squared_error'. (Mocked done with skforecast 0.7.0.)
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 5,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_no_exog_no_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, fixed_train_size yes, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_squared_error'. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_no_exog_remainder_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, fixed_train_size yes, 12 observations to backtest, steps=5 (remainder), 
    metric='mean_squared_error'. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 5,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)
    

def test_output_backtesting_sarimax_no_refit_yes_exog_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_yes_exog_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_squared_error'. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_yes_exog_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=5 (remainder), 
    metric='mean_squared_error'. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 5,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def my_metric(y_true, y_pred):  # pragma: no cover
    """
    Callable metric
    """
    metric = ((y_true - y_pred) / len(y_true)).mean()
    
    return metric


def test_output_backtesting_sarimax_no_refit_yes_exog_callable_metric_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), callable metric. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_no_refit_no_exog_list_of_metrics_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), list of metrics. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_no_exog_callable_metric_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, no exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), callable metric. 
    Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)


def test_output_backtesting_sarimax_yes_refit_fixed_train_size_yes_exog_list_of_metrics_with_mocked():
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=3 (no remainder), 
    list of metrics. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_preds, backtest_predictions, atol=0.0001)
    

@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values: f'alpha, interval: {values}')
def test_output_backtesting_sarimax_no_refit_yes_exog_interval_with_mocked(alpha, interval):
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    no refit, 12 observations to backtest, steps=3 (no remainder), metric='mean_absolute_error',
    interval. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = False,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values: f'alpha, interval: {values}')
def test_output_backtesting_sarimax_yes_refit_yes_exog_interval_with_mocked(alpha, interval):
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_absolute_error', interval. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = False,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)


@pytest.mark.parametrize("alpha, interval", 
                         [(0.05, [1, 99]), 
                          (None, [2.5, 97.5])], 
                         ids = lambda values: f'alpha, interval: {values}')
def test_output_backtesting_sarimax_yes_refit_fixed_train_size_yes_exog_interval_with_mocked(alpha, interval):
    """
    Test output of backtesting_sarimax with backtesting mocked, Series y is mocked, yes exog, 
    yes refit, fixed_train_size, 12 observations to backtest, steps=3 (no remainder), 
    metric='mean_absolute_error', interval. Mocked done with skforecast 0.7.0.
    """
    forecaster = ForecasterSarimax(
                     regressor = Sarimax(order=(3, 2, 0), maxiter=1000, method='cg', disp=False)
                 )
    
    cv = TimeSeriesFold(
             steps                 = 3,
             initial_train_size    = len(y_datetime) - 12,
             refit                 = True,
             fixed_train_size      = True,
             gap                   = 0,
             allow_incomplete_fold = True
         )
    
    metric, backtest_predictions = backtesting_sarimax(
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

    pd.testing.assert_frame_equal(expected_metric, metric, atol=0.0001)
    pd.testing.assert_frame_equal(expected_backtest_predictions, backtest_predictions, atol=0.0001)
