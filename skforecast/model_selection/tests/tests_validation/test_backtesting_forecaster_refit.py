# Unit test _backtesting_forecaster Refit
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.model_selection._split import TimeSeriesFold
from skforecast.model_selection._validation import _backtesting_forecaster

# Fixtures
from skforecast.exceptions import IgnoredArgumentWarning
from ..fixtures_model_selection import y
from ..fixtures_model_selection import exog
from ..fixtures_model_selection import out_sample_residuals


# ******************************************************************************
# * Test _backtesting_forecaster No Interval                                   *
# ******************************************************************************

@pytest.mark.parametrize("n_jobs", [-1, 1, 'auto'],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_forecaster_no_exog_no_remainder_ForecasterAutoreg_with_mocked(n_jobs):
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    ForecasterRecursive.
    """
    expected_metric = pd.DataFrame({'mean_squared_error': [0.06598802629306816]})
    expected_predictions = pd.DataFrame({
        'pred': np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 
                         0.38969292, 0.52778339, 0.49152015, 0.4841678, 
                         0.4076433, 0.50904672, 0.50249462, 0.49232817])}, 
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)
    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 4,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster = forecaster,
                                        y          = y,
                                        exog       = None,
                                        cv         = cv,
                                        metric     = 'mean_squared_error',
                                        n_jobs     = n_jobs,
                                        verbose    = False
                                   )
                                   
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_no_exog_no_remainder_ForecasterAutoregDirect_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    ForecasterDirect.
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.07076203468824617]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.5468482,
                    0.44670961,
                    0.57651222,
                    0.52511275,
                    0.3686309,
                    0.56234835,
                    0.44276032,
                    0.52260065,
                    0.37665741,
                    0.5382938,
                    0.48755548,
                    0.44534071,
                ]
            )
        },
        index=pd.RangeIndex(start=38, stop=50, step=1),
    )

    forecaster = ForecasterDirect(
                     regressor = LinearRegression(), 
                     lags      = 3,
                     steps     = 4
                 )
    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 4,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster = forecaster,
                                        y          = y,
                                        cv         = cv,
                                        exog       = None,
                                        metric     = 'mean_squared_error',
                                        verbose    = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_no_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    """

    expected_metric = pd.DataFrame({"mean_squared_error": [0.06916732087926723]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.55717779,
                    0.43355138,
                    0.54969767,
                    0.52945466,
                    0.48308861,
                    0.5096801,
                    0.49519677,
                    0.47997916,
                    0.49177914,
                    0.495797,
                    0.57738724,
                    0.44370472,
                ]
            )
        },
        index=pd.RangeIndex(start=38, stop=50, step=1),
    )
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 5,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster = forecaster,
                                        y          = y,
                                        exog       = None,
                                        cv         = cv,
                                        metric     = 'mean_squared_error',
                                        verbose    = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_yes_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """

    expected_metric = pd.DataFrame({"mean_squared_error": [0.05663345135204598]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.59059622,
                    0.47257504,
                    0.53024098,
                    0.46163343,
                    0.42295275,
                    0.46286083,
                    0.43618422,
                    0.43552906,
                    0.48687517,
                    0.55455072,
                    0.55577332,
                    0.53943402,
                ]
            )
        },
        index=pd.RangeIndex(start=38, stop=50, step=1),
    )
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 4,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster = forecaster,
                                        y          = y,
                                        exog       = exog,
                                        cv         = cv,
                                        metric     = 'mean_squared_error',
                                        verbose    = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_yes_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.061723961096013524]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.59059622,
                    0.47257504,
                    0.53024098,
                    0.46163343,
                    0.50035119,
                    0.43595809,
                    0.4349167,
                    0.42381237,
                    0.55165332,
                    0.53442833,
                    0.65361802,
                    0.51297419,
                ]
            )
        },
        index=pd.RangeIndex(start=38, stop=50, step=1),
    )
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 5,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = exog,
                                        cv                 = cv,
                                        metric             = 'mean_squared_error',
                                        verbose            = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_yes_exog_yes_remainder_skip_folds_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked,
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    skip_folds=2
    """
    expected_metric = pd.DataFrame({'mean_squared_error': [0.04512295747656866]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.59059622,
                    0.47257504,
                    0.53024098,
                    0.46163343,
                    0.50035119,
                    0.65361802,
                    0.51297419,
                ]
            )
        },
        index=[38, 39, 40, 41, 42, 48, 49],
    )
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 5,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = 2,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster = forecaster,
                                        y          = y,
                                        exog       = exog,
                                        cv         = cv,
                                        metric     = 'mean_squared_error',
                                        verbose    = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_yes_exog_yes_remainder_skip_folds_intermittent_refit_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked,
    24 observations to backtest, steps=3 (0 remainder), metric='mean_squared_error',
    skip_folds=2 and intermittent refit.
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.055092292428684]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array([
                0.43818148, 0.67328031, 0.61839117, 0.61725831, 0.49333332,
                0.39198592, 0.6550626, 0.48837405, 0.54686219, 0.42804696,
                0.4227273, 0.5499255])
        },
        index=[26, 27, 28, 32, 33, 34, 38, 39, 40, 44, 45, 46],
    )
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)

    n_backtest = 24
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 3,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = 3,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = 2,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        cv                 = cv,
                                        exog               = exog,
                                        metric             = 'mean_squared_error',
                                        verbose            = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Test _backtesting_forecaster Interval                                      *
# ******************************************************************************

def test_output_backtesting_forecaster_interval_no_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'use_in_sample_residuals = True'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.06598802629306816]})
    expected_predictions = pd.DataFrame(
        data=np.array([
            [0.55717779, 0.24709348, 0.95368172],
            [0.43355138, 0.08322668, 0.78788097],
            [0.54969767, 0.18838016, 0.9154328 ],
            [0.52945466, 0.15832868, 0.88376994],
            [0.38969292, 0.00353475, 0.78486946],
            [0.52778339, 0.11005189, 0.90737404],
            [0.49152015, 0.1002448 , 0.90143298],
            [0.4841678 , 0.08437119, 0.88977692],
            [0.4076433 , 0.0113795 , 0.78058831],
            [0.50904672, 0.10067467, 0.90908821],
            [0.50249462, 0.11675238, 0.90516282],
            [0.49232817, 0.09435161, 0.90728506]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 4,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y,
                                        exog                    = None,
                                        cv                      = cv,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False
                                   )
                             
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_interval_no_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    'use_in_sample_residuals = True'
    """
    expected_metric = pd.DataFrame({'mean_squared_error': [0.06916732087926723]})
    expected_predictions = pd.DataFrame(
        data=np.array([
                [0.55717779, 0.24709348, 0.95368172],
                [0.43355138, 0.08322668, 0.78788097],
                [0.54969767, 0.18838016, 0.9154328 ],
                [0.52945466, 0.15832868, 0.88376994],
                [0.48308861, 0.1261428 , 0.86164685],
                [0.5096801 , 0.06681772, 0.86172991],
                [0.49519677, 0.08912781, 0.91684662],
                [0.47997916, 0.10273939, 0.90498899],
                [0.49177914, 0.10934448, 0.88893765],
                [0.495797  , 0.08762867, 0.86507873],
                [0.57738724, 0.19244588, 0.96178584],
                [0.44370472, 0.02305298, 0.8315352 ]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 5,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y,
                                        exog                    = None,
                                        cv                      = cv,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False
                                   )
    
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_interval_yes_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'use_in_sample_residuals = True'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.05663345135204598]})
    expected_predictions = pd.DataFrame(
        data = np.array([
            [0.59059622, 0.24604234, 0.95777604],
            [0.47257504, 0.11196002, 0.85369193],
            [0.53024098, 0.1682998 , 0.91037299],
            [0.46163343, 0.10223625, 0.8247796 ],
            [0.42295275, 0.07166562, 0.8225198 ],
            [0.46286083, 0.09490153, 0.84051961],
            [0.43618422, 0.06022958, 0.80710401],
            [0.43552906, 0.03948656, 0.80872364],
            [0.48687517, 0.14867238, 0.89104985],
            [0.55455072, 0.19944523, 0.94531687],
            [0.55577332, 0.16834461, 0.96642363],
            [0.53943402, 0.18402811, 0.94236828]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 4,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y,
                                        exog                    = exog,
                                        cv                      = cv,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False
                                   )
    
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_interval_yes_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    'use_in_sample_residuals = True'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.061723961096013524]})
    expected_predictions = pd.DataFrame(
        data = np.array([
                [0.59059622, 0.24604234, 0.95777604],
                [0.47257504, 0.11196002, 0.85369193],
                [0.53024098, 0.1682998 , 0.91037299],
                [0.46163343, 0.10223625, 0.8247796 ],
                [0.50035119, 0.14659333, 0.85331366],
                [0.43595809, 0.08041239, 0.83346131],
                [0.4349167 , 0.04980571, 0.83970229],
                [0.42381237, 0.03993929, 0.83760039],
                [0.55165332, 0.16991552, 0.92146443],
                [0.53442833, 0.15396293, 0.91161655],
                [0.65361802, 0.29167326, 1.07608747],
                [0.51297419, 0.15499493, 0.91544835]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 5,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y,
                                        exog                    = exog,
                                        cv                      = cv,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Out sample residuals                                                       *
# ******************************************************************************

def test_output_backtesting_forecaster_interval_out_sample_residuals_no_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    'use_in_sample_residuals = False'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.06598802629306816]})
    expected_predictions = pd.DataFrame(
        data = np.array([
                [0.55717779, 0.63654358, 1.5406994 ],
                [0.43355138, 0.5760887 , 1.53342317],
                [0.54969767, 0.68344933, 1.57809559],
                [0.52945466, 0.69548679, 1.63622589],
                [0.38969292, 0.46905871, 1.37321453],
                [0.52778339, 0.67719181, 1.64080772],
                [0.49152015, 0.65291457, 1.54977963],
                [0.4841678 , 0.63577789, 1.57188267],
                [0.4076433 , 0.48700909, 1.39116491],
                [0.50904672, 0.64496324, 1.58747561],
                [0.50249462, 0.63982498, 1.5237537 ],
                [0.49232817, 0.58731791, 1.50369442]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)
    forecaster.out_sample_residuals_ = out_sample_residuals
    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 4,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )

    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y,
                                        exog                    = None,
                                        cv                      = cv,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = False,
                                        verbose                 = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Callable metric                                                            *
# ******************************************************************************

def my_metric(y_true, y_pred):  # pragma: no cover
    """
    Callable metric
    """
    metric = ((y_true - y_pred) / len(y_true)).mean()
    
    return metric


def test_callable_metric_backtesting_forecaster_no_exog_no_remainder_with_mocked():
    """
    Test callable metric in _backtesting_forecaster with backtesting mocked, interval no. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    expected_metric = pd.DataFrame({"my_metric": [0.005283745900436151]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.55717779,
                    0.43355138,
                    0.54969767,
                    0.52945466,
                    0.38969292,
                    0.52778339,
                    0.49152015,
                    0.4841678,
                    0.4076433,
                    0.50904672,
                    0.50249462,
                    0.49232817,
                ]
            )
        },
        index=pd.RangeIndex(start=38, stop=50, step=1),
    )
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 4,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = None,
                                        cv                 = cv,
                                        metric             = my_metric,
                                        verbose            = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_list_metrics_backtesting_forecaster_no_exog_no_remainder_with_mocked():
    """
    Test list of metrics in _backtesting_forecaster with backtesting mocked, interval no. 
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    """
    expected_metrics = pd.DataFrame(
        data=[[0.06598802629306816, 0.06598802629306816]],
        columns=['mean_squared_error', 'mean_squared_error']
    )
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.55717779,
                    0.43355138,
                    0.54969767,
                    0.52945466,
                    0.38969292,
                    0.52778339,
                    0.49152015,
                    0.4841678,
                    0.4076433,
                    0.50904672,
                    0.50249462,
                    0.49232817,
                ]
            )
        },
        index=pd.RangeIndex(start=38, stop=50, step=1),
    )

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 4,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metrics, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = None,
                                        cv                 = cv,
                                        metric             = ['mean_squared_error', mean_squared_error],
                                        verbose            = False
                                   )

    pd.testing.assert_frame_equal(expected_metrics, metrics)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * fixed_train_size = True                                                    *
# ******************************************************************************

def test_output_backtesting_forecaster_fixed_train_size_no_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    fixed_train_size=True
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.06720844584333846]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.55717779,
                    0.43355138,
                    0.54969767,
                    0.52945466,
                    0.34597367,
                    0.50223873,
                    0.47833829,
                    0.46082257,
                    0.37810191,
                    0.49508366,
                    0.48808014,
                    0.47323313,
                ]
            )
        },
        index=pd.RangeIndex(start=38, stop=50, step=1),
    )

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 4,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = None,
                                        cv                 = cv,
                                        metric             = 'mean_squared_error',
                                        verbose            = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_fixed_train_size_no_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    fixed_train_size=True
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.07217085374372428]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.55717779,
                    0.43355138,
                    0.54969767,
                    0.52945466,
                    0.48308861,
                    0.4909399,
                    0.47942107,
                    0.46025344,
                    0.46649132,
                    0.47061725,
                    0.57603136,
                    0.41480551,
                ]
            )
        },
        index=pd.RangeIndex(start=38, stop=50, step=1),
    )
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 5,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster         = forecaster,
                                        y                  = y,
                                        exog               = None,
                                        cv                 = cv,
                                        metric             = 'mean_squared_error',
                                        verbose            = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_fixed_train_size_yes_exog_no_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked,
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error',
    fixed_train_size=True
    """
    expected_metric = pd.DataFrame({'mean_squared_error': [0.05758244401484334]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.59059622,
                    0.47257504,
                    0.53024098,
                    0.46163343,
                    0.37689967,
                    0.44267729,
                    0.42642836,
                    0.41604275,
                    0.45047245,
                    0.53784704,
                    0.53726274,
                    0.51516772,
                ]
            )
        },
        index=pd.RangeIndex(start=38, stop=50, step=1),
    )
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 4,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster = forecaster,
                                        y          = y,
                                        exog       = exog,
                                        cv         = cv,
                                        metric     = 'mean_squared_error',
                                        verbose    = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_fixed_train_size_yes_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked,
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error',
    fixed_train_size=True
    """
    expected_metric = pd.DataFrame({'mean_squared_error': [0.06425019123005545]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.59059622,
                    0.47257504,
                    0.53024098,
                    0.46163343,
                    0.50035119,
                    0.41975558,
                    0.4256614,
                    0.41176005,
                    0.52357817,
                    0.509974,
                    0.65354628,
                    0.48210726,
                ]
            )
        },
        index=pd.RangeIndex(start=38, stop=50, step=1),
    )
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)

    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 5,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster = forecaster,
                                        y          = y,
                                        exog       = exog,
                                        cv         = cv,
                                        metric     = 'mean_squared_error',
                                        verbose    = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Gap                                                                        *
# ******************************************************************************


def test_output_backtesting_forecaster_interval_yes_exog_yes_remainder_gap_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=3, metric='mean_squared_error',
    'use_in_sample_residuals = True'
    """
    expected_metric = pd.DataFrame({'mean_squared_error': [0.0839045861490063]})
    expected_predictions = pd.DataFrame(
        data = np.array([[0.65022999, 0.27183154, 1.06503914],
                         [0.52364637,  0.19453326, 0.88265459],
                         [0.46809256,  0.12440678, 0.75395058],
                         [0.48759732,  0.24079549, 0.86300276],
                         [0.47172445,  0.14533367, 0.80742469],
                         [0.54075007,  0.15924753, 1.01588603],
                         [0.50283999,  0.19115121, 0.85076388],
                         [0.49737535,  0.22956778, 0.7939908 ],
                         [0.49185456,  0.22055651, 0.84145859],
                         [0.48906044,  0.17652033, 0.80063491],
                         [0.27751064, -0.07159784, 0.70858646],
                         [0.25859617, -0.03545793, 0.57770947],
                         [0.32853669,  0.03467979, 0.64259882],
                         [0.43751884,  0.13949436, 0.82693778],
                         [0.33016371, -0.04785842, 0.64473596],
                         [0.58126262,  0.22355106, 1.01410171],
                         [0.52955296,  0.19886586, 0.93120556]]),
        columns = ['pred', 'lower_bound', 'upper_bound'],
        index = pd.RangeIndex(start=33, stop=50, step=1)
    )

    forecaster = ForecasterDirect(
                     regressor = LinearRegression(), 
                     lags      = 3,
                     steps     = 8
                 )
    n_backtest = 20
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 5,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = False,
            gap                   = 3,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y,
                                        exog                    = exog,
                                        cv                      = cv,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_interval_yes_exog_not_allow_remainder_gap_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=3, metric='mean_squared_error',
    'use_in_sample_residuals = True', allow_incomplete_fold = False
    """
    y_with_index = y.copy()
    y_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_with_index = exog.copy()
    exog_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    expected_metric = pd.DataFrame({'mean_squared_error': [0.09133694038363274]})
    expected_predictions = pd.DataFrame(
        data = np.array([[0.65022999,  0.27183154, 1.06503914],
                         [0.52364637,  0.19453326, 0.88265459],
                         [0.46809256,  0.12440678, 0.75395058],
                         [0.48759732,  0.24079549, 0.86300276],
                         [0.47172445,  0.14533367, 0.80742469],
                         [0.50952084,  0.17367515, 0.96199755],
                         [0.52131987,  0.17725791, 0.88095373],
                         [0.50289964,  0.13703358, 0.89080844],
                         [0.54941988,  0.2239702 , 0.81990615],
                         [0.51121195,  0.23007235, 0.76212766],
                         [0.25123452, -0.02303138, 0.68451499],
                         [0.2791409 ,  0.05980179, 0.55535995],
                         [0.28390161,  0.00743721, 0.64748321],
                         [0.38317935,  0.0929118 , 0.73290098],
                         [0.42183128,  0.10993796, 0.80993652]]),
        columns = ['pred', 'lower_bound', 'upper_bound'],
        index = pd.date_range(start='2022-02-03', periods=15, freq='D')
    )

    forecaster = ForecasterDirect(
                     regressor = LinearRegression(), 
                     lags      = 3,
                     steps     = 8
                 )
    cv = TimeSeriesFold(
            steps                 = 5,
            initial_train_size    = len(y_with_index) - 20,
            window_size           = None,
            differentiation       = None,
            refit                 = True,
            fixed_train_size      = True,
            gap                   = 3,
            skip_folds            = None,
            allow_incomplete_fold = False,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster              = forecaster,
                                        y                       = y_with_index,
                                        exog                    = exog_with_index,
                                        cv                      = cv,
                                        metric                  = 'mean_squared_error',
                                        interval                = [5, 95],
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


# ******************************************************************************
# * Refit int                                                                  *
# ******************************************************************************

def test_output_backtesting_forecaster_refit_int_interval_yes_exog_yes_remainder_with_mocked():
    """
    Test output of backtesting_forecaster refit with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=0, metric='mean_squared_error',
    'use_in_sample_residuals = True'. Refit int.
    """
    expected_metric = pd.DataFrame(
        {'mean_squared_error': [0.06099110404144631], 
         'mean_absolute_scaled_error': [0.791393]}
    )
    expected_predictions = pd.DataFrame(
        data = np.array([[0.55616986, 0.15288789, 0.89198752],
                         [0.48751797, 0.14866438, 0.83169303],
                         [0.57764391, 0.17436194, 0.91346157],
                         [0.51298667, 0.17413308, 0.85716173],
                         [0.47430051, 0.0796644 , 0.82587748],
                         [0.49192271, 0.14609696, 0.95959395],
                         [0.52213783, 0.12750172, 0.8737148 ],
                         [0.54492575, 0.1991    , 1.012597  ],
                         [0.52501537, 0.13641764, 0.86685356],
                         [0.4680474 , 0.08515461, 0.81792677],
                         [0.51059498, 0.12199725, 0.85243317],
                         [0.53067132, 0.14777853, 0.88055069],
                         [0.4430938 , 0.0509291 , 0.69854202],
                         [0.49911716, 0.1231365 , 0.8711497 ],
                         [0.44546347, 0.05329877, 0.70091169],
                         [0.46530749, 0.08932683, 0.83734003],
                         [0.46901878, 0.08173407, 0.82098555],
                         [0.55371362, 0.14618224, 0.98199137],
                         [0.60759064, 0.22030593, 0.9595574 ],
                         [0.50415336, 0.09662198, 0.93243111]]),
        columns = ['pred', 'lower_bound', 'upper_bound'],
        index = pd.RangeIndex(start=30, stop=50, step=1)
    )

    forecaster = ForecasterDirect(
                     regressor = Ridge(random_state=123), 
                     lags      = 3,
                     steps     = 8
                 )

    n_backtest = 20
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 2,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = 2,
            fixed_train_size      = False,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                       forecaster              = forecaster,
                                       y                       = y,
                                       exog                    = exog,
                                       cv                      = cv,
                                       metric                  = ['mean_squared_error', 'mean_absolute_scaled_error'],
                                       interval                = [5, 95],
                                       n_boot                  = 500,
                                       random_state            = 123,
                                       use_in_sample_residuals = True,
                                       verbose                 = False,
                                       n_jobs                  = 1
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_refit_int_interval_yes_exog_not_allow_remainder_gap_with_mocked():
    """
    Test output of _backtesting_forecaster_no_refit with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=3, metric='mean_squared_error',
    'use_in_sample_residuals = True', allow_incomplete_fold = False. Refit int.
    """
    y_with_index = y.copy()
    y_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_with_index = exog.copy()
    exog_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    expected_metric = pd.DataFrame({'mean_squared_error': [0.060991643719298785]})
    expected_predictions = pd.DataFrame(
        data = np.array([
                [0.51878642, 0.19491548, 0.86522758],
                [0.49269791, 0.15037971, 0.85925162],
                [0.49380441, 0.1639645 , 0.89759979],
                [0.51467463, 0.15567332, 0.88191791],
                [0.52690045, 0.20302951, 0.87334162],
                [0.51731996, 0.17500175, 0.88387367],
                [0.51290311, 0.18306321, 0.9166985 ],
                [0.50334306, 0.14434174, 0.87058633],
                [0.50171526, 0.17784432, 0.84815642],
                [0.50946908, 0.16715088, 0.87602279],
                [0.50200357, 0.17216366, 0.90579895],
                [0.50436041, 0.1453591 , 0.87160369],
                [0.4534189 , 0.08003548, 0.82515615],
                [0.52621695, 0.17498193, 0.91058562],
                [0.50477802, 0.14697488, 0.88219215],
                [0.52235258, 0.14985376, 0.90260343]]),
                columns = ['pred', 'lower_bound', 'upper_bound'],
        index = pd.date_range(start='2022-02-03', periods=16, freq='D')
    )

    forecaster = ForecasterRecursive(regressor=Ridge(random_state=123), 
                                   lags=3, binner_kwargs={'n_bins': 15})
    cv = TimeSeriesFold(
            steps                 = 4,
            initial_train_size    = len(y_with_index) - 20,
            window_size           = None,
            differentiation       = None,
            refit                 = 3,
            fixed_train_size      = True,
            gap                   = 3,
            skip_folds            = None,
            allow_incomplete_fold = False,
            return_all_indexes    = False,
        )

    warn_msg = re.escape(
        ("If `refit` is an integer other than 1 (intermittent refit). `n_jobs` "
         "is set to 1 to avoid unexpected results during parallelization.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        metric, backtest_predictions = _backtesting_forecaster(
                                           forecaster              = forecaster,
                                           y                       = y_with_index,
                                           exog                    = exog_with_index,
                                           cv                      = cv,
                                           metric                  = 'mean_squared_error',
                                           interval                = [5, 95],
                                           n_boot                  = 500,
                                           random_state            = 123,
                                           use_in_sample_residuals = True,
                                           verbose                 = False,
                                           n_jobs                  = 2
                                       )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)
