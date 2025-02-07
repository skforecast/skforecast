# Unit test _backtesting_forecaster No refit
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.model_selection._validation import _backtesting_forecaster
from skforecast.model_selection._split import TimeSeriesFold

# Fixtures
from ..fixtures_model_selection import y
from ..fixtures_model_selection import exog
from ..fixtures_model_selection import out_sample_residuals


# ******************************************************************************
# * Test _backtesting_forecaster No Interval                                   *
# ******************************************************************************

@pytest.mark.parametrize("n_jobs", [-1, 1, 'auto'],
                         ids=lambda n: f'n_jobs: {n}')
def test_output_backtesting_forecaster_no_exog_no_remainder_ForecasterRecursive_with_mocked(n_jobs):
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    ForecasterRecursive.
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.0646438286283131]})
    expected_predictions = pd.DataFrame({
        'pred':np.array([0.55717779, 0.43355138, 0.54969767, 0.52945466, 
                         0.39585199, 0.55935949, 0.45263533, 0.4578669 , 
                         0.36988237, 0.57912951, 0.48686057, 0.45709952])}, 
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
            refit                 = False,
            fixed_train_size      = True,
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
                                       n_jobs     = n_jobs,
                                       verbose    = False
                                   )
                                   
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_no_exog_no_remainder_ForecasterDirect_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    12 observations to backtest, steps=4 (no remainder), metric='mean_squared_error'
    ForecasterDirect.
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.06999891546733726]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.5468482,
                    0.44670961,
                    0.57651222,
                    0.52511275,
                    0.39745044,
                    0.53365885,
                    0.39390143,
                    0.47002119,
                    0.38080838,
                    0.56987255,
                    0.42302249,
                    0.45580163,
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
            refit                 = False,
            fixed_train_size      = True,
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


def test_output_backtesting_forecaster_no_exog_no_initial_train_size_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, no exog, 
    no initial_train_size, steps=1, ForecasterRecursive.
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.05194702533929101]})
    expected_predictions = pd.DataFrame({
        'pred': np.array([0.50394528, 0.53092847, 0.51638165, 0.47382814, 0.61996956,
                         0.47685471, 0.52565717, 0.50842469, 0.4925563 , 0.55717972,
                         0.45707228, 0.45990822, 0.53762873, 0.5230163 , 0.41661072,
                         0.51097738, 0.52448483, 0.48179537, 0.5307759 , 0.55580453,
                         0.51780297, 0.53189442, 0.55356883, 0.46142853, 0.52517734,
                         0.46241276, 0.49292214, 0.53169102, 0.40448875, 0.55686226,
                         0.46860633, 0.5098154 , 0.49041677, 0.48435035, 0.51152271,
                         0.56870534, 0.53226143, 0.49091506, 0.56878395, 0.42767269,
                         0.53335856, 0.48167273, 0.5658333 , 0.41464667, 0.56733702,
                         0.5724869 , 0.45299923])
        }, 
        index=pd.RangeIndex(start=3, stop=50, step=1)
    )
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)
    forecaster.fit(y=y)
    initial_train_size = None
    cv = TimeSeriesFold(
            steps                 = 1,
            initial_train_size    = initial_train_size,
            window_size           = None,
            differentiation       = None,
            refit                 = False,
            fixed_train_size      = True,
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
    expected_metric = pd.DataFrame({"mean_squared_error": [0.07085869503962372]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.55717779,
                    0.43355138,
                    0.54969767,
                    0.52945466,
                    0.48308861,
                    0.54252612,
                    0.46139434,
                    0.44730047,
                    0.50031862,
                    0.49831103,
                    0.55613172,
                    0.42970914,
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
            refit                 = False,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = None,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
                                        forecaster = forecaster,
                                        y          = y,
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
    expected_metric = pd.DataFrame({"mean_squared_error": [0.05585411566592716]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.59059622,
                    0.47257504,
                    0.53024098,
                    0.46163343,
                    0.41316188,
                    0.51416101,
                    0.42705363,
                    0.4384041,
                    0.42611891,
                    0.59547291,
                    0.5170294,
                    0.4982889,
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
            refit                 = False,
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


def test_output_backtesting_forecaster_yes_exog_yes_remainder_with_mocked():
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval no.
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.06313056651237414]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.59059622,
                    0.47257504,
                    0.53024098,
                    0.46163343,
                    0.50035119,
                    0.49157332,
                    0.43327346,
                    0.42420804,
                    0.53389427,
                    0.51693094,
                    0.60207937,
                    0.48227974,
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
            refit                 = False,
            fixed_train_size      = True,
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
    12 observations to backtest, steps=5 (2 remainder), metric='mean_squared_error'
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.044538146622722964]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.59059622,
                    0.47257504,
                    0.53024098,
                    0.46163343,
                    0.50035119,
                    0.60207937,
                    0.48227974,
                ]
            )
        },
        index=pd.Index([38, 39, 40, 41, 42, 48, 49], dtype="int64"),
    )
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)
    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 5,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = False,
            fixed_train_size      = True,
            gap                   = 0,
            skip_folds            = 2,
            allow_incomplete_fold = True,
            return_all_indexes    = False,
        )
    metric, backtest_predictions = _backtesting_forecaster(
        forecaster = forecaster,
        y                  = y,
        exog               = exog,
        cv                 = cv,
        metric             = "mean_squared_error",
        verbose            = True,
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
    expected_metric = pd.DataFrame({"mean_squared_error": [0.06464382862831312]})
    expected_predictions = pd.DataFrame(
        data = np.array([
            [0.55717779, 0.19882822, 0.87939731],
            [0.43355138, 0.08976463, 0.86652665],
            [0.54969767, 0.20849403, 0.98096724],
            [0.52945466, 0.16705061, 0.92717139],
            [0.39585199, 0.03750242, 0.71807151],
            [0.55935949, 0.21557274, 0.99233476],
            [0.45263533, 0.11143169, 0.8839049 ],
            [0.4578669 , 0.09546285, 0.85558363],
            [0.36988237, 0.01153279, 0.69210188],
            [0.57912951, 0.23534276, 1.01210478],
            [0.48686057, 0.14565693, 0.91813015],
            [0.45709952, 0.09469547, 0.85481624]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
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
            refit                 = False,
            fixed_train_size      = True,
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
                                        verbose                 = False,
                                        show_progress           = False
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
    expected_metric = pd.DataFrame({"mean_squared_error": [0.07085869503962372]})
    expected_predictions = pd.DataFrame(
        data = np.array([
            [0.55717779, 0.19882822, 0.87939731],
            [0.43355138, 0.08976463, 0.86652665],
            [0.54969767, 0.20849403, 0.98096724],
            [0.52945466, 0.16705061, 0.92717139],
            [0.48308861, 0.12658912, 0.89058516],
            [0.54252612, 0.18417655, 0.86474563],
            [0.46139434, 0.11760759, 0.89436961],
            [0.44730047, 0.10609683, 0.87857004],
            [0.50031862, 0.13791457, 0.89803535],
            [0.49831103, 0.14181154, 0.90580758],
            [0.55613172, 0.19778215, 0.87835123],
            [0.42970914, 0.08592239, 0.86268441]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)
    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 5,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = False,
            fixed_train_size      = True,
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
    expected_metric = pd.DataFrame({"mean_squared_error": [0.05585411566592716]})
    expected_predictions = pd.DataFrame(
    data = np.array([
            [0.59059622, 0.24316567, 0.93342085],
            [0.47257504, 0.12485033, 0.87643834],
            [0.53024098, 0.18273747, 0.93078198],
            [0.46163343, 0.10256219, 0.83853371],
            [0.41316188, 0.06573133, 0.75598651],
            [0.51416101, 0.16643631, 0.91802432],
            [0.42705363, 0.07955012, 0.82759464],
            [0.4384041 , 0.07933286, 0.81530438],
            [0.42611891, 0.07868836, 0.76894354],
            [0.59547291, 0.24774821, 0.99933622],
            [0.5170294 , 0.16952589, 0.91757041],
            [0.4982889 , 0.13921766, 0.87518918]]
        ),
        columns=['pred', 'lower_bound', 'upper_bound'],
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
            refit                 = False,
            fixed_train_size      = True,
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
    expected_metric = pd.DataFrame({"mean_squared_error": [0.06313056651237414]})
    expected_predictions = pd.DataFrame(
        data = np.array([
            [0.59059622, 0.24316567, 0.93342085],
            [0.47257504, 0.12485033, 0.87643834],
            [0.53024098, 0.18273747, 0.93078198],
            [0.46163343, 0.10256219, 0.83853371],
            [0.50035119, 0.14089436, 0.88160183],
            [0.49157332, 0.14414278, 0.83439795],
            [0.43327346, 0.08554876, 0.83713677],
            [0.42420804, 0.07670453, 0.82474905],
            [0.53389427, 0.17482303, 0.91079456],
            [0.51693094, 0.15747411, 0.89818158],
            [0.60207937, 0.25464882, 0.944904  ],
            [0.48227974, 0.13455504, 0.88614305]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )
    
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)
    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
            steps                 = 5,
            initial_train_size    = len(y_train),
            window_size           = None,
            differentiation       = None,
            refit                 = False,
            fixed_train_size      = True,
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
                                        interval                = (5, 95),
                                        n_boot                  = 500,
                                        random_state            = 123,
                                        use_in_sample_residuals = True,
                                        verbose                 = False,
                                        show_progress           = False
                                   )
    
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("initial_train_size", 
                         [len(y) - 20, "2022-01-30 00:00:00"],
                         ids=lambda init: f'initial_train_size: {init}')
def test_output_backtesting_forecaster_no_refit_interval_yes_exog_bootstrapping(initial_train_size):
    """
    Test output of _backtesting_forecaster with predicted intervals as bootstrapping.
    """
    y_with_index = y.copy()
    y_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_with_index = exog.copy()
    exog_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    expected_metric = pd.DataFrame({"mean_squared_error": [0.06392277343002713]})
    expected_predictions = pd.DataFrame(
        data = np.array([
                [0.55538506, 0.60367643, 0.88152611, 0.5634774 , 0.72025328,
                    0.2137087 , 0.6199332 , 0.3945439 , 0.52863287, 0.40661365,
                    0.52863287],
                [0.44124674, 0.30814369, 0.67552661, 0.59094798, 0.1530626 ,
                    0.47746399, 0.30400363, 0.5175231 , 0.16374181, 0.33787653,
                    0.09089052],
                [0.49752366, 0.49160155, 0.17901185, 0.3860609 , 0.20553126,
                    0.42645717, 0.16942951, 0.68746286, 0.47814313, 0.49592495,
                    0.48960277],
                [0.53822797, 0.54949064, 0.60649888, 0.46744448, 0.39636506,
                    0.57650316, 0.31502897, 0.28183223, 0.71848603, 0.59646825,
                    0.29032951],
                [0.4793552 , 0.58652515, 0.45968808, 0.35057928, 0.41631354,
                    0.56344941, 0.13469427, 0.41615485, 0.4767692 , 0.47616   ,
                    0.30475643],
                [0.4474213 , 0.49571267, 0.77356235, 0.45551363, 0.61228952,
                    0.10574494, 0.51196944, 0.28658014, 0.42066911, 0.29864989,
                    0.42066911],
                [0.51546009, 0.38235704, 0.74973996, 0.66516132, 0.22727595,
                    0.55167734, 0.37821697, 0.59173645, 0.23795516, 0.41208988,
                    0.16510387],
                [0.54066381, 0.53474169, 0.22215199, 0.42920105, 0.24867141,
                    0.46959732, 0.21256965, 0.73060301, 0.52128328, 0.53906509,
                    0.53274291],
                [0.51970475, 0.53096742, 0.58797566, 0.44892126, 0.37784183,
                    0.55797994, 0.29650575, 0.26330901, 0.69996281, 0.57794502,
                    0.27180629],
                [0.51420212, 0.62137208, 0.494535  , 0.38542621, 0.45116047,
                    0.59829634, 0.1695412 , 0.45100177, 0.51161613, 0.51100693,
                    0.33960336],
                [0.4699511 , 0.51824248, 0.79609215, 0.47804344, 0.63481932,
                    0.12827474, 0.53449924, 0.30910994, 0.44319891, 0.3211797 ,
                    0.44319891],
                [0.55077981, 0.41767676, 0.78505967, 0.70048104, 0.26259567,
                    0.58699706, 0.41353669, 0.62705617, 0.27327487, 0.4474096 ,
                    0.20042359],
                [0.54063198, 0.53470987, 0.22212017, 0.42916922, 0.24863958,
                    0.46956549, 0.21253783, 0.73057119, 0.52125146, 0.53903327,
                    0.53271109],
                [0.48476663, 0.49602931, 0.55303754, 0.41398315, 0.34290372,
                    0.52304182, 0.26156764, 0.2283709 , 0.66502469, 0.54300691,
                    0.23686818],
                [0.49651396, 0.60368391, 0.47684684, 0.36773805, 0.43347231,
                    0.58060818, 0.15185304, 0.43331361, 0.49392796, 0.49331876,
                    0.32191519],
                [0.59616217, 0.64445355, 0.92230322, 0.60425451, 0.76103039,
                    0.25448581, 0.66071031, 0.43532101, 0.56940998, 0.44739077,
                    0.56940998],
                [0.46166238, 0.32855932, 0.69594224, 0.61136361, 0.17347823,
                    0.49787962, 0.32441926, 0.53793873, 0.18415744, 0.35829216,
                    0.11130615],
                [0.55197041, 0.54604829, 0.23345859, 0.44050765, 0.25997801,
                    0.48090392, 0.22387625, 0.74190961, 0.53258988, 0.55037169,
                    0.54404951],
                [0.5694184 , 0.58068107, 0.63768931, 0.49863492, 0.42755549,
                    0.60769359, 0.34621941, 0.31302266, 0.74967646, 0.62765868,
                    0.32151994],
                [0.52086432, 0.62803427, 0.5011972 , 0.39208841, 0.45782267,
                    0.60495854, 0.1762034 , 0.45766397, 0.51827832, 0.51766912,
                    0.34626555]]),
        columns = ['pred'] + [f'pred_boot_{i}' for i in range(10)],
        index = pd.date_range(start='2022-01-31', periods=20, freq='D')
    )

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)
    cv = TimeSeriesFold(
             steps              = 5,
             initial_train_size = initial_train_size,
             refit              = False
         )
    metric, backtest_predictions = _backtesting_forecaster(
                                       forecaster              = forecaster,
                                       y                       = y_with_index,
                                       exog                    = exog_with_index,
                                       cv                      = cv,
                                       metric                  = 'mean_squared_error',
                                       interval                = 'bootstrapping',
                                       n_boot                  = 10,
                                       random_state            = 123,
                                       use_in_sample_residuals = True,
                                       verbose                 = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


def test_output_backtesting_forecaster_no_refit_interval_distribution_yes_exog():
    """
    Test output of _backtesting_forecaster with predicted intervals as distribution.
    """
    y_with_index = y.copy()
    y_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_with_index = exog.copy()
    exog_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    expected_metric = pd.DataFrame({"mean_absolute_error": [0.19221231167961977]})
    expected_predictions = pd.DataFrame(
        data = np.array([
            [0.55538506, 0.5265697 , 0.19122853],
            [0.44124674, 0.45485962, 0.21296327],
            [0.49752366, 0.48702575, 0.20938072],
            [0.53822797, 0.51715637, 0.21511474],
            [0.4793552 , 0.4814511 , 0.20866791],
            [0.4474213 , 0.41860594, 0.19122853],
            [0.51546009, 0.52907297, 0.21296327],
            [0.54066381, 0.5301659 , 0.20938072],
            [0.51970475, 0.49863315, 0.21511474],
            [0.51420212, 0.51629803, 0.20866791],
            [0.4699511 , 0.44113574, 0.19122853],
            [0.55077981, 0.56439269, 0.21296327],
            [0.54063198, 0.53013407, 0.20938072],
            [0.48476663, 0.46369504, 0.21511474],
            [0.49651396, 0.49860986, 0.20866791],
            [0.59616217, 0.56734681, 0.19122853],
            [0.46166238, 0.47527525, 0.21296327],
            [0.55197041, 0.5414725 , 0.20938072],
            [0.5694184 , 0.5483468 , 0.21511474],
            [0.52086432, 0.52296022, 0.20866791]]),
        columns = ['pred', 'loc', 'scale'],
        index = pd.date_range(start='2022-01-31', periods=20, freq='D')
    )

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=3)
    cv = TimeSeriesFold(
             steps              = 5,
             initial_train_size = len(y_with_index) - 20,
             refit              = False
         )
    metric, backtest_predictions = _backtesting_forecaster(
                                       forecaster              = forecaster,
                                       y                       = y_with_index,
                                       exog                    = exog_with_index,
                                       cv                      = cv,
                                       metric                  = 'mean_absolute_error',
                                       interval                = norm,
                                       n_boot                  = 250,
                                       random_state            = 123,
                                       use_in_sample_residuals = True,
                                       verbose                 = False
                                   )

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("interval", 
                         [0.90, (5, 95)], 
                         ids = lambda value: f'interval: {value}')
def test_output_backtesting_forecaster_interval_conformal_and_binned_with_mocked(interval):
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), conformal=True, binned=True.
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.063130566512374]})
    expected_predictions = pd.DataFrame(
        data = np.array([
            [0.59059622, 0.2974502 , 0.88374223],
            [0.47257504, 0.22356317, 0.72158691],
            [0.53024098, 0.14339956, 0.91708239],
            [0.46163343, 0.21262156, 0.7106453 ],
            [0.50035119, 0.35604735, 0.64465502],
            [0.49157332, 0.34726949, 0.63587716],
            [0.43327346, 0.18718506, 0.67936186],
            [0.42420804, 0.32142927, 0.52698681],
            [0.53389427, 0.2579754 , 0.80981315],
            [0.51693094, 0.13008952, 0.90377235],
            [0.60207937, 0.30893335, 0.89522538],
            [0.48227974, 0.0046491 , 0.95991039]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )
    
    forecaster = ForecasterRecursive(
        regressor=LinearRegression(), lags=3, binner_kwargs={'n_bins': 10}
    )
    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
             steps                 = 5,
             initial_train_size    = len(y_train),
             window_size           = None,
             differentiation       = None,
             refit                 = False,
             fixed_train_size      = True,
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
                                        interval                = interval,
                                        interval_method         = 'conformal',
                                        use_in_sample_residuals = True,
                                        use_binned_residuals    = True,
                                        random_state            = 123,
                                        verbose                 = False,
                                        show_progress           = False
                                   )
    
    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)


@pytest.mark.parametrize("interval", 
                         [0.90, (5, 95)], 
                         ids = lambda value: f'interval: {value}')
def test_output_backtesting_forecaster_interval_conformal_and_binned_with_mocked_ForecasterDirect(interval):
    """
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    12 observations to backtest, steps=5 (2 remainder), conformal=True, binned=True.
    """
    expected_metric = pd.DataFrame({"mean_squared_error": [0.061964730085838]})
    expected_predictions = pd.DataFrame(
        data = np.array([
            [0.55584775, 0.37659309, 0.73510242],
            [0.45718425, 0.08077419, 0.83359431],
            [0.57242372, 0.36697393, 0.77787352],
            [0.47236513, 0.10146645, 0.84326382],
            [0.50476389, 0.16013181, 0.84939597],
            [0.51445718, 0.1698251 , 0.85908926],
            [0.42759863, 0.14064966, 0.71454759],
            [0.42937664, 0.14242768, 0.71632561],
            [0.51632981, 0.17169773, 0.86096189],
            [0.57758034, 0.37213055, 0.78303014],
            [0.55914632, 0.37989165, 0.73840098],
            [0.46393165, 0.09303296, 0.83483033]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=38, stop=50, step=1)
    )
    
    forecaster = ForecasterDirect(
        regressor=LinearRegression(), steps=5, lags=3, binner_kwargs={'n_bins': 10}
    )
    n_backtest = 12
    y_train = y[:-n_backtest]
    cv = TimeSeriesFold(
             steps                 = 5,
             initial_train_size    = len(y_train),
             window_size           = None,
             differentiation       = None,
             refit                 = False,
             fixed_train_size      = True,
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
                                        interval                = interval,
                                        interval_method         = 'conformal',
                                        use_in_sample_residuals = True,
                                        use_binned_residuals    = True,
                                        random_state            = 123,
                                        verbose                 = False,
                                        show_progress           = False
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
    expected_metric = pd.DataFrame({"mean_squared_error": [0.0646438286283131]})
    expected_predictions = pd.DataFrame(
        data = np.array([
            [0.55717779, 0.63654358, 1.5406994 ],
            [0.43355138, 0.5760887 , 1.53342317],
            [0.54969767, 0.68344933, 1.57809559],
            [0.52945466, 0.69548679, 1.63622589],
            [0.39585199, 0.47521778, 1.3793736 ],
            [0.55935949, 0.70189681, 1.65923128],
            [0.45263533, 0.58638699, 1.48103325],
            [0.4578669 , 0.62389903, 1.56463813],
            [0.36988237, 0.44924816, 1.35340398],
            [0.57912951, 0.72166684, 1.67900131],
            [0.48686057, 0.62061223, 1.51525849],
            [0.45709952, 0.62313165, 1.56387074]]),
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
            refit                 = False,
            fixed_train_size      = True,
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
# * Callable metric                                                           *
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
    expected_metric = pd.DataFrame({"my_metric": [0.005603130564222017]})
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.55717779,
                    0.43355138,
                    0.54969767,
                    0.52945466,
                    0.39585199,
                    0.55935949,
                    0.45263533,
                    0.4578669,
                    0.36988237,
                    0.57912951,
                    0.48686057,
                    0.45709952,
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
            refit                 = False,
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
        {
            "mean_absolute_error": [0.20796216],
            "mean_squared_error": [0.0646438286283131],
        }
    )
    expected_predictions = pd.DataFrame(
        {
            "pred": np.array(
                [
                    0.55717779,
                    0.43355138,
                    0.54969767,
                    0.52945466,
                    0.39585199,
                    0.55935949,
                    0.45263533,
                    0.4578669,
                    0.36988237,
                    0.57912951,
                    0.48686057,
                    0.45709952,
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
            refit                 = False,
            fixed_train_size      = True,
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
                                        metric             = ['mean_absolute_error', mean_squared_error],
                                        verbose            = False
                                   )

    pd.testing.assert_frame_equal(expected_metrics, metrics)
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
    expected_metric = pd.DataFrame({"mean_squared_error": [0.07965887934114284]})
    expected_predictions = pd.DataFrame(
        {
            "pred": {
                33: 0.6502299892039047,
                34: 0.5236463733888286,
                35: 0.4680925560056311,
                36: 0.48759731640448023,
                37: 0.4717244484701478,
                38: 0.5758584470242223,
                39: 0.5359293308956417,
                40: 0.48995122056218404,
                41: 0.4527131747771053,
                42: 0.49519412870799573,
                43: 0.2716782430953111,
                44: 0.2781382241553876,
                45: 0.31569423787508544,
                46: 0.39832240179281636,
                47: 0.3482268061581242,
                48: 0.6209868551179476,
                49: 0.4719075668681217,
            },
            "lower_bound": {
                33: 0.2718315398459972,
                34: 0.19453326150156153,
                35: 0.12440677892063873,
                36: 0.12107577460096824,
                37: 0.14533367434351524,
                38: 0.19745999766631478,
                39: 0.2068162190083746,
                40: 0.14626544347719167,
                41: 0.08619163297359334,
                42: 0.1688033545813632,
                43: -0.10672020626259643,
                44: -0.05097488773187947,
                45: -0.027991539209906935,
                46: 0.031800859989304375,
                47: 0.021836032031491648,
                48: 0.2425884057600401,
                49: 0.14279445498085463,
            },
            "upper_bound": {
                33: 1.065039136744312,
                34: 0.9230439995748042,
                35: 0.7539505750052855,
                36: 0.8314505919311639,
                37: 0.7469755216091047,
                38: 0.9906675945646294,
                39: 0.9353269570816173,
                40: 0.7758092395618383,
                41: 0.796566450303789,
                42: 0.7704452018469528,
                43: 0.6864873906357182,
                44: 0.6775358503413632,
                45: 0.6015522568747398,
                46: 0.7421756773195001,
                47: 0.6234778792970812,
                48: 1.0357960026583548,
                49: 0.8713051930540974,
            },
        }
    )
    expected_predictions = pd.DataFrame(
        data = np.array([
            [0.65022999,  0.27183154, 1.06503914],
            [0.52364637,  0.19453326, 0.88265459],
            [0.46809256,  0.12440678, 0.75395058],
            [0.48759732,  0.24079549, 0.86300276],
            [0.47172445,  0.14533367, 0.80742469],
            [0.57585845,  0.19746   , 0.99066759],
            [0.53592933,  0.20681622, 0.89493754],
            [0.48995122,  0.14626544, 0.77580924],
            [0.45271317,  0.20591135, 0.82811861],
            [0.49519413,  0.16880335, 0.83089437],
            [0.27167824, -0.10672021, 0.68648739],
            [0.27813822, -0.05097489, 0.63714644],
            [0.31569424, -0.02799154, 0.60155226],
            [0.3983224 ,  0.15152057, 0.77372784],
            [0.34822681,  0.02183603, 0.68392705],
            [0.62098686,  0.24258841, 1.035796  ],
            [0.47190757,  0.14279445, 0.83091578]]),
        columns=['pred', 'lower_bound', 'upper_bound'],
        index=pd.RangeIndex(start=33, stop=50, step=1)
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
            refit                 = False,
            fixed_train_size      = True,
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
    Test output of _backtesting_forecaster with backtesting mocked, interval yes. 
    Regressor is LinearRegression with lags=3, Series y is mocked, exog is mocked, 
    20 observations to backtest, steps=5 and gap=3, metric='mean_squared_error',
    'use_in_sample_residuals = True', allow_incomplete_fold = False
    """
    y_with_index = y.copy()
    y_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')
    exog_with_index = exog.copy()
    exog_with_index.index = pd.date_range(start='2022-01-01', periods=50, freq='D')

    expected_metric = pd.DataFrame({"mean_squared_error": [0.08826806818628832]})
    expected_predictions = pd.DataFrame(
        data = np.array([[0.65022999,  0.27183154, 1.06503914],
                         [0.52364637,  0.19453326, 0.88265459],
                         [0.46809256,  0.12440678, 0.75395058],
                         [0.48759732,  0.24079549, 0.86300276],
                         [0.47172445,  0.14533367, 0.80742469],
                         [0.57585845,  0.19746   , 0.99066759],
                         [0.53592933,  0.20681622, 0.89493754],
                         [0.48995122,  0.14626544, 0.77580924],
                         [0.45271317,  0.20591135, 0.82811861],
                         [0.49519413,  0.16880335, 0.83089437],
                         [0.27167824, -0.10672021, 0.68648739],
                         [0.27813822, -0.05097489, 0.63714644],
                         [0.31569424, -0.02799154, 0.60155226],
                         [0.3983224 ,  0.15152057, 0.77372784],
                         [0.34822681,  0.02183603, 0.68392705]]),
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
            refit                 = False,
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
    backtest_predictions = backtest_predictions.asfreq('D')

    pd.testing.assert_frame_equal(expected_metric, metric)
    pd.testing.assert_frame_equal(expected_predictions, backtest_predictions)
