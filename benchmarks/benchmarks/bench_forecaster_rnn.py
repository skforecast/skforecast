################################################################################
#                          Benchmarking ForecasterRnn                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np
import pandas as pd
from skforecast.deep_learning import create_and_compile_model, ForecasterRnn
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


def run_benchmark_ForecasterRnn(output_dir):
    """
    Run all benchmarks for the ForecasterRnn class and save the results.
    """

    # Setup
    # ==========================================================================
    series, exog, exog_pred = _make_data()

    model = create_and_compile_model(
        series=series, 
        lags=10, 
        steps=STEPS, 
        recurrent_layer="LSTM", 
        recurrent_units=64,
        dense_units=64, 
        model_name="benchmark_no_exog"
    )

    model_exog = create_and_compile_model(
        series=series, 
        exog=exog,
        lags=10, 
        steps=STEPS, 
        recurrent_layer="LSTM", 
        recurrent_units=64,
        dense_units=64, 
        model_name="benchmark_exog"
    )

    forecaster = ForecasterRnn(
                     regressor  = model,
                     levels     = list(series.columns),
                     lags       = 10,
                     fit_kwargs = {'epochs': 25, 'batch_size': 32}
                 )
    
    forecaster_exog = ForecasterRnn(
                          regressor  = model_exog,
                          levels     = list(series.columns),
                          lags       = 10,
                          fit_kwargs = {'epochs': 25, 'batch_size': 32}
                      )

    def ForecasterRnn__create_train_X_y(forecaster, series, exog):
        forecaster._create_train_X_y(series=series, exog=exog)

    def ForecasterRnn__create_train_X_y_no_exog(forecaster, series):
        forecaster._create_train_X_y(series=series)

    def ForecasterRnn_fit(forecaster, series, exog):
        forecaster.fit(series=series, exog=exog)

    def ForecasterRnn_fit_series_no_exog(forecaster, series):
        forecaster.fit(series=series)

    def ForecasterRnn_check_predict_inputs(forecaster, exog):
        check_predict_input(
            forecaster_name   = type(forecaster).__name__,
            steps             = STEPS,
            is_fitted         = forecaster.is_fitted,
            exog_in_          = forecaster.exog_in_,
            index_type_       = forecaster.index_type_,
            index_freq_       = forecaster.index_freq_,
            window_size       = forecaster.window_size,
            last_window       = forecaster.last_window_,
            exog              = exog,
            exog_names_in_    = forecaster.exog_names_in_,
            interval          = None,
            max_step          = forecaster.max_step,
            levels            = forecaster.levels,
            levels_forecaster = forecaster.levels,
            series_names_in_  = forecaster.series_names_in_,
        )

    def ForecasterRnn__create_predict_inputs(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = STEPS,
                exog         = exog,
                check_inputs = True
            )

    def ForecasterRnn_predict(forecaster, exog):
        forecaster.predict(steps=5, exog=exog, suppress_warnings=True)

    def ForecasterRnn_predict_interval_conformal(forecaster, exog):
        forecaster.predict_interval(
            steps=STEPS,
            exog=exog,
            method='conformal',
            interval=[5, 95],
            suppress_warnings=True
        )

    def ForecasterRnn_backtesting(forecaster, series, exog):
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
                show_progress=False
            )
            
    def ForecasterRnn_backtesting_no_exog(forecaster, series):
        cv = TimeSeriesFold(
                initial_train_size=900,
                fixed_train_size=True,
                steps=STEPS,
            )
        _ = backtesting_forecaster_multiseries(
                forecaster=forecaster,
                series=series,
                cv=cv,
                metric='mean_squared_error',
                show_progress=False
            )
            
    def ForecasterRnn_backtesting_conformal(forecaster, series, exog):
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
                interval=[5, 95],
                interval_method='conformal',
                metric='mean_squared_error',
                show_progress=False
            )

    runner = BenchmarkRunner(repeat=10, output_dir=output_dir)
    _ = runner.benchmark(ForecasterRnn__create_train_X_y, forecaster=forecaster_exog, series=series, exog=exog)
    _ = runner.benchmark(ForecasterRnn__create_train_X_y_no_exog, forecaster=forecaster, series=series)

    runner = BenchmarkRunner(repeat=5, output_dir=output_dir)
    _ = runner.benchmark(ForecasterRnn_fit, forecaster=forecaster_exog, series=series, exog=exog)
    _ = runner.benchmark(ForecasterRnn_fit_series_no_exog, forecaster=forecaster, series=series)

    forecaster_exog.fit(series=series, exog=exog, store_in_sample_residuals=True)
    runner = BenchmarkRunner(repeat=10, output_dir=output_dir)
    _ = runner.benchmark(ForecasterRnn_check_predict_inputs, forecaster=forecaster_exog, exog=exog_pred)
    _ = runner.benchmark(ForecasterRnn__create_predict_inputs, forecaster=forecaster_exog, exog=exog_pred)
    _ = runner.benchmark(ForecasterRnn_predict, forecaster=forecaster_exog, exog=exog_pred)
    _ = runner.benchmark(ForecasterRnn_predict_interval_conformal, forecaster=forecaster_exog, exog=exog_pred)

    runner = BenchmarkRunner(repeat=5, output_dir=output_dir)
    _ = runner.benchmark(ForecasterRnn_backtesting, forecaster=forecaster_exog, series=series, exog=exog)
    _ = runner.benchmark(ForecasterRnn_backtesting_no_exog, forecaster=forecaster, series=series)
    _ = runner.benchmark(ForecasterRnn_backtesting_conformal, forecaster=forecaster_exog, series=series, exog=exog)
