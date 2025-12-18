################################################################################
#                         Benchmarking ForecasterDirect                        #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import numpy as np
import pandas as pd
from packaging.version import parse
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from skforecast import __version__ as skforecast_version
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.utils import check_predict_input
from .benchmark_runner import BenchmarkRunner

# Global config
LEN_SERIES = 2000
STEPS = 10
RANDOM_STATE = 123


def _make_data(
    len_series: int = LEN_SERIES, 
    steps: int = STEPS, 
    random_state: int = RANDOM_STATE
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic time series data for benchmarking.
    
    Parameters
    ----------
    len_series : int
        Length of the time series.
    steps : int
        Number of prediction steps to generate.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    y : pandas Series
        Target time series.
    exog : pandas DataFrame
        Exogenous variables for training.
    exog_pred : pandas DataFrame
        Exogenous variables for prediction.
    
    """

    rng = np.random.default_rng(random_state)
    y = pd.Series(
        data=rng.normal(loc=20, scale=5, size=len_series),
        index=pd.date_range(start='2010-01-01', periods=len_series, freq='h'),
        name='y'
    )

    # Create exogenous variables with temporal features
    exog = pd.DataFrame(index=y.index)
    exog['day_of_week'] = exog.index.dayofweek
    exog['week_of_year'] = exog.index.isocalendar().week.astype(int)
    exog['month'] = exog.index.month

    # Create future exogenous variables for prediction
    exog_pred = pd.DataFrame(
        index=pd.date_range(
            start=exog.index.max() + pd.Timedelta(hours=1),
            periods=steps,
            freq='h'
        )
    )
    exog_pred['day_of_week'] = exog_pred.index.dayofweek
    exog_pred['week_of_year'] = exog_pred.index.isocalendar().week.astype(int)
    exog_pred['month'] = exog_pred.index.month

    return y, exog, exog_pred


def run_benchmark_ForecasterDirect(output_dir):
    """
    Run all benchmarks for the ForecasterDirect class and save the results.
    """

    # Setup
    # ==========================================================================
    y, exog, exog_pred = _make_data()
    y_values = y.to_numpy()

    forecaster = ForecasterDirect(
        estimator=DummyRegressor(strategy='constant', constant=1.),
        steps=STEPS,
        lags=20,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
        n_jobs=1
    )

    def ForecasterDirect__create_lags(forecaster, y):
        forecaster._create_lags(y=y, X_as_pandas=False, train_index=None)

    def ForecasterDirect__create_train_X_y(forecaster, y, exog):
        forecaster._create_train_X_y(y=y, exog=exog)

    def ForecasterDirect_fit(forecaster, y, exog):
        forecaster.fit(y=y, exog=exog)

    def ForecasterDirect_check_predict_inputs(forecaster, exog):
        if parse(skforecast_version) >= parse("0.17.0"):
            check_predict_input(
                forecaster_name = type(forecaster).__name__,
                steps           = list(np.arange(STEPS) + 1),
                is_fitted       = forecaster.is_fitted,
                exog_in_        = forecaster.exog_in_,
                index_type_     = forecaster.index_type_,
                index_freq_     = forecaster.index_freq_,
                window_size     = forecaster.window_size,
                last_window     = forecaster.last_window_,
                exog            = exog,
                exog_names_in_  = forecaster.exog_names_in_,
                interval        = None,
                max_step        = forecaster.max_step
            )
        else:
            check_predict_input(
                forecaster_name = type(forecaster).__name__,
                steps           = list(np.arange(STEPS) + 1),
                is_fitted       = forecaster.is_fitted,
                exog_in_        = forecaster.exog_in_,
                index_type_     = forecaster.index_type_,
                index_freq_     = forecaster.index_freq_,
                window_size     = forecaster.window_size,
                last_window     = forecaster.last_window_,
                exog            = exog,
                exog_names_in_  = forecaster.exog_names_in_,
                interval        = None,
                max_steps       = forecaster.steps
            )

    def ForecasterDirect__create_predict_inputs(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = STEPS,
                exog         = exog,
                check_inputs = True
            )

    def ForecasterDirect_predict(forecaster, exog):
        forecaster.predict(steps=STEPS, exog=exog)

    def ForecasterDirect_predict_interval_conformal(forecaster, exog):
        forecaster.predict_interval(
            steps=STEPS,
            exog=exog,
            interval=[5, 95],
            method='conformal'
        )
        
    def ForecasterDirect_backtesting(forecaster, y, exog):
        cv = TimeSeriesFold(
                 initial_train_size=1200,
                 steps=STEPS,
                 refit=False
             )
        _ = backtesting_forecaster(
                forecaster=forecaster,
                y=y,
                exog=exog,
                cv=cv,
                metric='mean_squared_error',
                n_jobs=1,
                show_progress=False
            )

    def ForecasterDirect_backtesting_conformal(forecaster, y, exog):
        cv = TimeSeriesFold(
                 initial_train_size=1200,
                 steps=STEPS,
                 refit=False
             )
        _ = backtesting_forecaster(
                forecaster=forecaster,
                y=y,
                exog=exog,
                interval=[5, 95],
                interval_method='conformal',
                cv=cv,
                metric='mean_squared_error',
                n_jobs=1,
                show_progress=False,
                suppress_warnings=True
            )

    runner = BenchmarkRunner(repeat=30, output_dir=output_dir)
    _ = runner.benchmark(ForecasterDirect__create_lags, forecaster=forecaster, y=y_values)
    _ = runner.benchmark(ForecasterDirect__create_train_X_y, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=10, output_dir=output_dir)
    _ = runner.benchmark(ForecasterDirect_fit, forecaster=forecaster, y=y, exog=exog)

    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True, suppress_warnings=True)
    runner = BenchmarkRunner(repeat=30, output_dir=output_dir)
    _ = runner.benchmark(ForecasterDirect_check_predict_inputs, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterDirect__create_predict_inputs, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterDirect_predict, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterDirect_predict_interval_conformal, forecaster=forecaster, exog=exog_pred)

    runner = BenchmarkRunner(repeat=5, output_dir=output_dir)
    _ = runner.benchmark(ForecasterDirect_backtesting, forecaster=forecaster, y=y, exog=exog)
    _ = runner.benchmark(ForecasterDirect_backtesting_conformal, forecaster=forecaster, y=y, exog=exog)
