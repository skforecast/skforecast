################################################################################
#                  Benchmarking ForecasterDirectMultiVariate                   #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.utils import check_predict_input
from .benchmark_runner import BenchmarkRunner

# Global config
N_SERIES = 13
LEN_SERIES = 1000
STEPS = 10
RANDOM_STATE = 123


def _make_data(
    n_series: int = N_SERIES,
    len_series: int = LEN_SERIES, 
    steps: int = STEPS, 
    random_state: int = RANDOM_STATE
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic time series data for benchmarking.
    
    Parameters
    ----------
    n_series : int
        Number of time series to generate.
    len_series : int
        Length of the time series.
    steps : int
        Number of prediction steps to generate.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    series : pandas DataFrame
        Target time series.
    exog : pandas DataFrame
        Exogenous variables for training.
    exog_pred : pandas DataFrame
        Exogenous variables for prediction.
    
    """

    rng = np.random.default_rng(random_state)
    series = pd.DataFrame(
        data = rng.normal(
            loc = 10,
            scale = 3,
            size = (len_series, n_series)
        ),
        columns = [f'series_{i}' for i in range(n_series)],
        index = pd.date_range(
            start = '2020-01-01',
            periods = len_series,
            freq = 'h'
        )
    )

    # Create exogenous variables with temporal features
    exog = pd.DataFrame(index=series.index)
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

    return series, exog, exog_pred


def run_benchmark_ForecasterDirectMultiVariate():
    """
    Run all benchmarks for the ForecasterDirectMultiVariate class and save the results.
    """

    # Setup
    # ==========================================================================
    series, exog, exog_pred = _make_data()

    forecaster = ForecasterDirectMultiVariate(
        regressor=DummyRegressor(strategy='constant', constant=1.),
        level='series_1',
        steps=STEPS,
        lags=20,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
        n_jobs=1
    )

    def ForecasterDirectMultiVariate__create_train_X_y(forecaster, series, exog):
        forecaster._create_train_X_y(series=series, exog=exog)

    def ForecasterDirectMultiVariate__create_train_X_y_no_exog(forecaster, series):
        forecaster._create_train_X_y(series=series)

    def ForecasterDirectMultiVariate_fit(forecaster, series, exog):
        forecaster.fit(series=series, exog=exog)

    def ForecasterDirectMultiVariate_fit_series_no_exog(forecaster, series):
        forecaster.fit(series=series)

    def ForecasterDirectMultiVariate_check_predict_inputs(forecaster, exog):
        check_predict_input(
            forecaster_name  = type(forecaster).__name__,
            steps            = list(np.arange(STEPS) + 1),
            is_fitted        = forecaster.is_fitted,
            exog_in_         = forecaster.exog_in_,
            index_type_      = forecaster.index_type_,
            index_freq_      = forecaster.index_freq_,
            window_size      = forecaster.window_size,
            last_window      = forecaster.last_window_,
            exog             = exog,
            exog_names_in_   = forecaster.exog_names_in_,
            interval         = None,
            max_step         = forecaster.max_step,
            series_names_in_ = forecaster.X_train_series_names_in_
        )

    def ForecasterDirectMultiVariate__create_predict_inputs(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = STEPS,
                exog         = exog,
                check_inputs = True
            )

    def ForecasterDirectMultiVariate_predict(forecaster, exog):
        forecaster.predict(steps=STEPS, exog=exog, suppress_warnings=True)

    def ForecasterDirectMultiVariate_predict_interval_conformal(forecaster, exog):
        forecaster.predict_interval(
            steps=STEPS,
            exog=exog,
            interval=[5, 95],
            method='conformal',
            suppress_warnings=True
        )
        
    def ForecasterDirectMultiVariate_backtesting(forecaster, series, exog):
        cv = TimeSeriesFold(
                initial_train_size=900,
                fixed_train_size=True,
                steps=STEPS,
            )
        _ = backtesting_forecaster_multiseries(
                forecaster=forecaster,
                series=series,
                exog=exog,
                cv=cv,
                metric='mean_squared_error',
                n_jobs=1,
                show_progress=False
            )
            
    def ForecasterDirectMultiVariate_backtesting_no_exog(forecaster, series):
        cv = TimeSeriesFold(
                initial_train_size=900,
                fixed_train_size=True,
                steps=5,
            )
        _ = backtesting_forecaster_multiseries(
                forecaster=forecaster,
                series=series,
                cv=cv,
                metric='mean_squared_error',
                n_jobs=1,
                show_progress=False
            )

    def ForecasterDirectMultiVariate_backtesting_conformal(forecaster, series, exog):
        cv = TimeSeriesFold(
                initial_train_size=900,
                fixed_train_size=True,
                steps=STEPS,
            )
        _ = backtesting_forecaster_multiseries(
                forecaster=forecaster,
                series=series,
                exog=exog,
                interval=[5, 95],
                interval_method='conformal',
                cv=cv,
                metric='mean_squared_error',
                n_jobs=1,
                show_progress=False
            )

    runner = BenchmarkRunner(repeat=10, output_dir="./benchmarks")
    _ = runner.benchmark(ForecasterDirectMultiVariate__create_train_X_y, forecaster=forecaster, series=series, exog=exog)
    _ = runner.benchmark(ForecasterDirectMultiVariate__create_train_X_y_no_exog, forecaster=forecaster, series=series)

    runner = BenchmarkRunner(repeat=5, output_dir="./benchmarks")
    _ = runner.benchmark(ForecasterDirectMultiVariate_fit, forecaster=forecaster, series=series, exog=exog)
    _ = runner.benchmark(ForecasterDirectMultiVariate_fit_series_no_exog, forecaster=forecaster, series=series)

    runner = BenchmarkRunner(repeat=10, output_dir="./benchmarks")
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    _ = runner.benchmark(ForecasterDirectMultiVariate_check_predict_inputs, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterDirectMultiVariate__create_predict_inputs, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterDirectMultiVariate_predict, forecaster=forecaster, exog=exog_pred)
    _ = runner.benchmark(ForecasterDirectMultiVariate_predict_interval_conformal, forecaster=forecaster, exog=exog_pred)

    runner = BenchmarkRunner(repeat=5, output_dir="./benchmarks")
    _ = runner.benchmark(ForecasterDirectMultiVariate_backtesting, forecaster=forecaster, series=series, exog=exog)
    _ = runner.benchmark(ForecasterDirectMultiVariate_backtesting_no_exog, forecaster=forecaster, series=series)
    _ = runner.benchmark(ForecasterDirectMultiVariate_backtesting_conformal, forecaster=forecaster, series=series, exog=exog)
