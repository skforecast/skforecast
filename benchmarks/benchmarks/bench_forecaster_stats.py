################################################################################
#                         Benchmarking ForecasterStats                         #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterStats
from skforecast.stats import Arima
from skforecast.model_selection import TimeSeriesFold, backtesting_stats
from skforecast.utils import check_predict_input
from .benchmark_runner import BenchmarkRunner

# Global config
LEN_SERIES = 500
STEPS = 24
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


def run_benchmark_ForecasterStats(output_dir):
    """
    Run all benchmarks for the ForecasterStats class with Arima and save the results.
    """

    # Setup
    # ==========================================================================
    y, exog, exog_pred = _make_data()

    forecaster = ForecasterStats(
        estimator=Arima(order=(1, 1, 1), seasonal_order=(0, 0, 0), m=1),
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler()
    )

    # Forecaster with auto arima for parameter selection benchmarks
    forecaster_auto = ForecasterStats(
        estimator=Arima(order=None, seasonal_order=None, m=24),
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler()
    )

    # Warm-up calls (compile numba functions)
    forecaster.fit(y=y, exog=exog, suppress_warnings=True)
    forecaster.predict(steps=STEPS, exog=exog_pred, suppress_warnings=True)
    forecaster.predict_interval(
        steps=STEPS, exog=exog_pred, alpha=0.05, suppress_warnings=True
    )
    forecaster_auto.fit(y=y, exog=exog, suppress_warnings=True)
    forecaster_auto.predict(steps=STEPS, exog=exog_pred, suppress_warnings=True)
    forecaster_auto.predict_interval(
        steps=STEPS, exog=exog_pred, alpha=0.05, suppress_warnings=True
    )

    def ForecasterStats_fit(forecaster, y, exog):
        forecaster.fit(y=y, exog=exog, suppress_warnings=True)

    def ForecasterStats_fit_auto(forecaster, y, exog):
        forecaster.fit(y=y, exog=exog, suppress_warnings=True)

    def ForecasterStats_check_predict_inputs(forecaster, exog):
        check_predict_input(
            forecaster_name  = type(forecaster).__name__,
            steps            = STEPS,
            is_fitted        = forecaster.is_fitted,
            exog_in_         = forecaster.exog_in_,
            index_type_      = forecaster.index_type_,
            index_freq_      = forecaster.index_freq_,
            window_size      = forecaster.window_size,
            last_window      = forecaster.last_window_,
            last_window_exog = None,
            exog             = exog,
            exog_names_in_   = forecaster.exog_names_in_,
            interval         = None,
            alpha            = None
        )

    def ForecasterStats__create_predict_inputs(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps = STEPS,
                exog  = exog
            )

    def ForecasterStats_predict(forecaster, exog):
        forecaster.predict(steps=STEPS, exog=exog, suppress_warnings=True)

    def ForecasterStats_predict_interval(forecaster, exog):
        forecaster.predict_interval(
            steps=STEPS,
            exog=exog,
            alpha=0.05,
            suppress_warnings=True
        )

    def ForecasterStats_backtesting(forecaster, y):
        cv = TimeSeriesFold(
                 initial_train_size=300,
                 steps=24,
                 refit=True
             )
        _ = backtesting_stats(
                forecaster=forecaster,
                y=y,
                cv=cv,
                metric='mean_squared_error',
                n_jobs=1,
                show_progress=False,
                suppress_warnings=True
            )

    def ForecasterStats_backtesting_exog(forecaster, y, exog):
        cv = TimeSeriesFold(
                 initial_train_size=300,
                 steps=24,
                 refit=True
             )
        _ = backtesting_stats(
                forecaster=forecaster,
                y=y,
                exog=exog,
                cv=cv,
                metric='mean_squared_error',
                n_jobs=1,
                show_progress=False,
                suppress_warnings=True
            )

    def ForecasterStats_backtesting_interval_exog(forecaster, y, exog):
        cv = TimeSeriesFold(
                 initial_train_size=300,
                 steps=24,
                 refit=True
             )
        _ = backtesting_stats(
                forecaster=forecaster,
                y=y,
                exog=exog,
                alpha=0.05,
                cv=cv,
                metric='mean_squared_error',
                n_jobs=1,
                show_progress=False,
                suppress_warnings=True
            )

    # Fit benchmarks
    runner = BenchmarkRunner(repeat=15, output_dir=output_dir)
    _ = runner.benchmark(ForecasterStats_fit, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=8, output_dir=output_dir)
    _ = runner.benchmark(ForecasterStats_fit_auto, forecaster=forecaster_auto, y=y, exog=exog)

    # Predict benchmarks (fit first)
    forecaster.fit(y=y, exog=exog, suppress_warnings=True)
    runner = BenchmarkRunner(repeat=30, output_dir=output_dir)
    _ = runner.benchmark(ForecasterStats_check_predict_inputs, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterStats__create_predict_inputs, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterStats_predict, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterStats_predict_interval, forecaster=forecaster, exog=exog_pred)

    # Backtesting benchmarks
    runner = BenchmarkRunner(repeat=8, output_dir=output_dir)
    _ = runner.benchmark(ForecasterStats_backtesting, forecaster=forecaster, y=y)
    _ = runner.benchmark(ForecasterStats_backtesting_exog, forecaster=forecaster, y=y, exog=exog)
    _ = runner.benchmark(ForecasterStats_backtesting_interval_exog, forecaster=forecaster, y=y, exog=exog)
