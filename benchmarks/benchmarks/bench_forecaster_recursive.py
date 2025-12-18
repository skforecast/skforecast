################################################################################
#                       Benchmarking ForecasterRecursive                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.utils import check_predict_input
from .benchmark_runner import BenchmarkRunner

# Global config
LEN_SERIES = 2000
STEPS = 100
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


def run_benchmark_ForecasterRecursive(output_dir):
    """
    Run all benchmarks for the ForecasterRecursive class and save the results.
    """

    # Setup
    # ==========================================================================
    y, exog, exog_pred = _make_data()
    y_values = y.to_numpy()

    forecaster = ForecasterRecursive(
        estimator=DummyRegressor(strategy='constant', constant=1.),
        lags=50,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler()
    )

    # NOTE: As a constant prediction doesn't represent a real use case, we include
    # a forecaster with a LinearRegression estimator for binned bootstrapping.
    forecaster_boot = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=50,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
        binner_kwargs={'n_bins': 10}
    )

    def ForecasterRecursive__create_lags(forecaster, y):
        forecaster._create_lags(y=y, X_as_pandas=False, train_index=None)

    def ForecasterRecursive__create_train_X_y(forecaster, y, exog):
        forecaster._create_train_X_y(y=y, exog=exog)

    def ForecasterRecursive_fit(forecaster, y, exog):
        forecaster.fit(y=y, exog=exog)

    def ForecasterRecursive_check_predict_inputs(forecaster, exog):
        check_predict_input(
            forecaster_name = type(forecaster).__name__,
            steps           = STEPS,
            is_fitted       = forecaster.is_fitted,
            exog_in_        = forecaster.exog_in_,
            index_type_     = forecaster.index_type_,
            index_freq_     = forecaster.index_freq_,
            window_size     = forecaster.window_size,
            last_window     = forecaster.last_window_,
            exog            = exog,
            exog_names_in_  = forecaster.exog_names_in_,
            interval        = None
        )

    def ForecasterRecursive__create_predict_inputs(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = STEPS,
                exog         = exog,
                check_inputs = True
            )

    def ForecasterRecursive_predict(forecaster, exog):
        forecaster.predict(steps=STEPS, exog=exog)

    def ForecasterRecursive_predict_interval_conformal(forecaster, exog):
        forecaster.predict_interval(
            steps=STEPS,
            exog=exog,
            interval=[5, 95],
            method='conformal'
        )
        
    def ForecasterRecursive_backtesting(forecaster, y, exog):
        cv = TimeSeriesFold(
                 initial_train_size=1200,
                 steps=50,
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

    def ForecasterRecursive_backtesting_conformal(forecaster, y, exog):
        cv = TimeSeriesFold(
                 initial_train_size=1200,
                 steps=50,
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
                show_progress=False
            )

    def ForecasterRecursive_predict_bootstrapping(forecaster, exog):
        forecaster.predict_bootstrapping(
            steps=STEPS,
            exog=exog,
            n_boot=250
        )

    runner = BenchmarkRunner(repeat=30, output_dir=output_dir)
    _ = runner.benchmark(ForecasterRecursive__create_lags, forecaster=forecaster, y=y_values)
    _ = runner.benchmark(ForecasterRecursive__create_train_X_y, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=10, output_dir=output_dir)
    _ = runner.benchmark(ForecasterRecursive_fit, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=30, output_dir=output_dir)
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    _ = runner.benchmark(ForecasterRecursive_check_predict_inputs, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterRecursive__create_predict_inputs, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterRecursive_predict, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterRecursive_predict_interval_conformal, forecaster=forecaster, exog=exog_pred)

    runner = BenchmarkRunner(repeat=5, output_dir=output_dir)
    _ = runner.benchmark(ForecasterRecursive_backtesting, forecaster=forecaster, y=y, exog=exog)
    _ = runner.benchmark(ForecasterRecursive_backtesting_conformal, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=30, output_dir=output_dir)
    forecaster_boot.fit(y=y, exog=exog, store_in_sample_residuals=True)
    _ = runner.benchmark(ForecasterRecursive_predict_bootstrapping, forecaster=forecaster_boot, exog=exog_pred)
