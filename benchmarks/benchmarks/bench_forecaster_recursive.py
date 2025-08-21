################################################################################
#                       Benchmarking ForecasterRecursive                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from .common import BenchmarkRunner


def _make_data(
    len_series: int = 2000, 
    steps: int = 100, 
    random_state: int = 123
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
    exog_prediction : pandas DataFrame
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


def run_benchmark_ForecasterRecursive():
    """
    Run all benchmarks for the ForecasterRecursive class and save the results.
    """

    # Setup
    y, exog, exog_pred = _make_data()

    forecaster = ForecasterRecursive(
        regressor=DummyRegressor(strategy='constant', constant=1.),
        lags=50,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    def ForecasterRecursive_fit(forecaster, y, exog):
        forecaster.fit(y=y, exog=exog)

    def ForecasterRecursive__create_train_X_y(forecaster, y, exog):
        forecaster._create_train_X_y(y=y, exog=exog)

    def ForecasterRecursive__create_predict_inputs(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = 100,
                exog         = exog,
                check_inputs = True
            )

    def ForecasterRecursive_predict(forecaster, exog):
        forecaster.predict(steps=100, exog=exog)

    def ForecasterRecursive_predict_interval_conformal(forecaster, exog):
        forecaster.predict_interval(
            steps=100,
            exog=exog,
            interval=[5, 95],
            method='conformal'
        )
        
    def ForecasterRecursive_backtesting(forecaster, y, exog):
        cv = TimeSeriesFold(
                initial_train_size=1200,
                fixed_train_size=True,
                steps=50,
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

    def ForecasterRecursive_backtesting_conformal(forecaster, y, exog):
        cv = TimeSeriesFold(
                initial_train_size=1200,
                fixed_train_size=True,
                steps=50,
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
                show_progress=False
            )

    runner = BenchmarkRunner(repeat=30, output_dir="./benchmarks")
    _ = runner.benchmark(ForecasterRecursive__create_train_X_y, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=10, output_dir="./benchmarks")
    _ = runner.benchmark(ForecasterRecursive_fit, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=30, output_dir="./benchmarks")
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    _ = runner.benchmark(ForecasterRecursive__create_predict_inputs, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterRecursive_predict, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterRecursive_predict_interval_conformal, forecaster=forecaster, exog=exog_pred)

    runner = BenchmarkRunner(repeat=5, output_dir="./benchmarks")
    _ = runner.benchmark(ForecasterRecursive_backtesting, forecaster=forecaster, y=y, exog=exog)
    _ = runner.benchmark(ForecasterRecursive_backtesting_conformal, forecaster=forecaster, y=y, exog=exog)
