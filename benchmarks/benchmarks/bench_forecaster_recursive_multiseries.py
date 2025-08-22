################################################################################
#                  Benchmarking ForecasterRecursiveMultiSeries                 #
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
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.utils import check_predict_input
from .benchmark_runner import BenchmarkRunner

# Global config
N_SERIES = 600
LEN_SERIES = 2000
STEPS = 100
RANDOM_STATE = 123


def _make_data(
    n_series: int = N_SERIES,
    len_series: int = LEN_SERIES,
    steps: int = STEPS,
    random_state: int = RANDOM_STATE
) -> tuple[dict, pd.DataFrame, dict, dict, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
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
    series_dict : dict
        Dictionary containing the generated time series.
    series_df_long : pd.DataFrame
        Long format DataFrame containing the generated time series.
    series_dict_different_length : dict
        Dictionary containing the generated time series with different lengths.
    exog_dict : dict
        Dictionary containing the exogenous variables for training.
    exog_df_long : pd.DataFrame
        Long format DataFrame containing the exogenous variables for training.
    exog_df_wide : pd.DataFrame
        Wide format DataFrame containing the exogenous variables for training.
    exog_dict_pred : dict
        Dictionary containing the exogenous variables for prediction.
    exog_df_long_pred : pd.DataFrame
        Long format DataFrame containing the exogenous variables for prediction.
    exog_df_wide_pred : pd.DataFrame
        Wide format DataFrame containing the exogenous variables for prediction.

    """

    series_dict = {}
    rng = np.random.default_rng(random_state)
    for i in range(n_series):
        series_dict[f'series_{i}'] = pd.Series(
            data = rng.normal(loc=20, scale=5, size=len_series),
            index=pd.date_range(
                start='2010-01-01',
                periods=len_series,
                freq='h'
            ),
            name=f'series_{i}'
        )

    series_dict_different_length = {
        k: v.iloc[:-rng.integers(low=1, high=1000)].copy() 
        for k, v in series_dict.items()
    }

    series_df_wide = pd.DataFrame(series_dict, index=series_dict['series_0'].index)

    series_df_long = series_df_wide.stack()
    series_df_long.index = series_df_long.index.set_names(['datetime', 'series_id'])
    series_df_long = series_df_long.swaplevel().sort_index()
    series_df_long = series_df_long.to_frame(name='value')

    # Create exogenous variables with temporal features
    exog_df_wide = pd.DataFrame(index=series_df_wide.index)
    exog_df_wide['day_of_week'] = exog_df_wide.index.dayofweek
    exog_df_wide['week_of_year'] = exog_df_wide.index.isocalendar().week.astype(int)
    exog_df_wide['month'] = exog_df_wide.index.month

    exog_dict = {}
    for k in series_dict.keys():
        exog_dict[k] = exog_df_wide.copy()

    exog_df_long = (
        pd.concat([exog.assign(series_id=k) for k, exog in exog_dict.items()])
        .reset_index()
        .rename(columns={'index': 'datetime'})
        .set_index(['series_id', 'datetime'])
    )

    # Create future exogenous variables for prediction
    exog_df_wide_pred = pd.DataFrame(
        index=pd.date_range(
            start=series_df_wide.index.max() + pd.Timedelta(hours=1),
            periods=steps,
            freq='h'
        )
    )
    exog_df_wide_pred['day_of_week'] = exog_df_wide_pred.index.dayofweek
    exog_df_wide_pred['week_of_year'] = exog_df_wide_pred.index.isocalendar().week.astype(int)
    exog_df_wide_pred['month'] = exog_df_wide_pred.index.month

    exog_dict_pred = {}
    for k in series_dict.keys():
        exog_dict_pred[k] = exog_df_wide_pred.copy()
        
    exog_df_long_pred = (
        pd.concat([exog.assign(series_id=k) for k, exog in exog_dict_pred.items()])
        .reset_index()
        .rename(columns={'index': 'datetime'})
        .set_index(['series_id', 'datetime'])
    )

    return (
        series_dict, 
        series_df_long, 
        series_dict_different_length,
        exog_dict,
        exog_df_long,
        exog_df_wide,
        exog_dict_pred,
        exog_df_long_pred,
        exog_df_wide_pred
    )


def run_benchmark_ForecasterRecursiveMultiSeries():
    """
    Run all benchmarks for the ForecasterRecursiveMultiSeries class and save the results.
    """

    # Setup
    # ==========================================================================
    (
        series_dict, 
        series_df_long, 
        series_dict_different_length,
        exog_dict,
        exog_df_long,
        exog_df_wide,
        exog_dict_pred,
        exog_df_long_pred,
        exog_df_wide_pred
    ) = _make_data()

    forecaster = ForecasterRecursiveMultiSeries(
        regressor=DummyRegressor(strategy='constant', constant=1.),
        lags=50,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
        encoding="ordinal"
    )

    # _create_train_X_y and create_train_X_y_single_series
    # ==========================================================================
    def ForecasterRecursiveMultiSeries__create_train_X_y_series_is_dict_no_exog(forecaster, series):
        forecaster._create_train_X_y(series=series, exog=None)

    def ForecasterRecursiveMultiSeries__create_train_X_y_series_is_dict_exog_is_dict(forecaster, series, exog):
        forecaster._create_train_X_y(series=series, exog=exog)

    def ForecasterRecursiveMultiSeries__create_train_X_y_series_is_dict_different_length_exog_is_dict(forecaster, series, exog):
        forecaster._create_train_X_y(series=series, exog=exog)

    def ForecasterRecursiveMultiSeries__create_train_X_y_series_is_dict_exog_is_df_wide(forecaster, series, exog):
        forecaster._create_train_X_y(series=series, exog=exog)

    def ForecasterRecursiveMultiSeries__create_train_X_y_series_is_df_long_no_exog(forecaster, series):
        forecaster._create_train_X_y(series=series, exog=None)

    def ForecasterRecursiveMultiSeries__create_train_X_y_series_is_df_long_exog_is_df_long(forecaster, series, exog):
        forecaster._create_train_X_y(series=series, exog=exog)

    def ForecasterRecursiveMultiSeries__create_train_X_y_series_is_df_long_exog_is_df_wide(forecaster, series, exog):
        forecaster._create_train_X_y(series=series, exog=exog)

    def ForecasterRecursiveMultiSeries__create_train_X_y_single_series(forecaster, y, exog):
        _ = forecaster._create_train_X_y_single_series(
                y           = y,
                exog        = exog,
                ignore_exog = False,
            )

    # Fit
    # ==========================================================================
    def ForecasterRecursiveMultiSeries_fit_series_is_dict_no_exog(forecaster, series):
        forecaster.fit(series=series, exog=None)

    def ForecasterRecursiveMultiSeries_fit_series_is_dict_exog_is_dict(forecaster, series, exog):
        forecaster.fit(series=series, exog=exog)

    def ForecasterRecursiveMultiSeries_fit_series_is_dict_different_length_exog_is_dict(forecaster, series, exog):
        forecaster.fit(series=series, exog=exog)

    def ForecasterRecursiveMultiSeries_fit_series_is_dataframe_no_exog(forecaster, series):
        forecaster.fit(series=series, exog=None)

    def ForecasterRecursiveMultiSeries_fit_series_is_dataframe_exog_is_dataframe(forecaster, series, exog):
        forecaster.fit(series=series, exog=exog)

    def ForecasterRecursiveMultiSeries_fit_series_is_dataframe_exog_is_dict(forecaster, series, exog):
        forecaster.fit(series=series, exog=exog)

    # Predict
    # ==========================================================================
    def ForecasterRecursiveMultiSeries_check_predict_inputs(forecaster, exog):
        check_predict_input(
            forecaster_name  = type(forecaster).__name__,
            steps            = STEPS,
            is_fitted        = forecaster.is_fitted,
            exog_in_         = forecaster.exog_in_,
            index_type_      = forecaster.index_type_,
            index_freq_      = forecaster.index_freq_,
            window_size      = forecaster.window_size,
            last_window      = pd.DataFrame(forecaster.last_window_),
            exog             = exog,
            exog_names_in_   = forecaster.exog_names_in_,
            interval         = None,
            levels           = forecaster.series_names_in_,
            series_names_in_ = forecaster.series_names_in_,
            encoding         = forecaster.encoding
        )

    def ForecasterRecursiveMultiSeries__create_predict_inputs_exog_is_dict(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = STEPS,
                exog         = exog,
                check_inputs = True
            )

    def ForecasterRecursiveMultiSeries__create_predict_inputs_exog_is_df_long(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = STEPS,
                exog         = exog,
                check_inputs = True
            )
        
    def ForecasterRecursiveMultiSeries__create_predict_inputs_exog_is_df_wide(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = STEPS,
                exog         = exog,
                check_inputs = True
            )
    
    def ForecasterRecursiveMultiSeries_predict_exog_is_dict(forecaster, exog):
        forecaster.predict(steps=STEPS, exog=exog, suppress_warnings=True)

    def ForecasterRecursiveMultiSeries_predict_exog_is_df_long(forecaster, exog):
        forecaster.predict(steps=STEPS, exog=exog, suppress_warnings=True)

    def ForecasterRecursiveMultiSeries_predict_interval_exog_is_dict_conformal(forecaster, exog):
        forecaster.predict_interval(
            steps=STEPS,
            exog=exog,
            method='conformal',
            interval=[5, 95],
            suppress_warnings=True
        )

    # Backtesting
    # ==========================================================================
    def ForecasterRecursiveMultiSeries_backtesting_series_is_dict_no_exog(forecaster, series):
        cv = TimeSeriesFold(
                initial_train_size=1200,
                fixed_train_size=True,
                steps=50,
            )
        _ = backtesting_forecaster_multiseries(
                forecaster=forecaster,
                series=series,
                exog=None,
                cv=cv,
                metric='mean_squared_error',
                show_progress=False
            )
            
    def ForecasterRecursiveMultiSeries_backtesting_series_is_dict_exog_is_dict(forecaster, series, exog):
        cv = TimeSeriesFold(
                initial_train_size=1200,
                fixed_train_size=True,
                steps=50,
            )
        _ = backtesting_forecaster_multiseries(
                forecaster=forecaster,
                series=series,
                exog=exog,
                cv=cv,
                metric='mean_squared_error',
                show_progress=False
            )

    def ForecasterRecursiveMultiSeries_backtesting_series_is_dict_no_exog_conformal(forecaster, series):
        cv = TimeSeriesFold(
                initial_train_size=1200,
                fixed_train_size=True,
                steps=50,
            )
        _ = backtesting_forecaster_multiseries(
                forecaster=forecaster,
                series=series,
                exog=None,
                cv=cv,
                interval=[5, 95],
                interval_method='conformal',
                metric='mean_squared_error',
                show_progress=False
            )
            
    def ForecasterRecursiveMultiSeries_backtesting_series_is_dict_exog_dict_conformal(forecaster, series, exog):
        cv = TimeSeriesFold(
                initial_train_size=1200,
                fixed_train_size=True,
                steps=50,
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

    # Create train_X_y
    # ==========================================================================
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=DummyRegressor(strategy='constant', constant=1.),
        lags=50,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
        encoding="ordinal"
    )

    runner = BenchmarkRunner(repeat=10, output_dir="./benchmarks")
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries__create_train_X_y_series_is_dict_no_exog,
            forecaster=forecaster,
            series=series_dict
        )
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries__create_train_X_y_series_is_dict_exog_is_dict,
            forecaster=forecaster,
            series=series_dict,
            exog=exog_dict
        )
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries__create_train_X_y_series_is_dict_different_length_exog_is_dict,
            forecaster=forecaster,
            series=series_dict_different_length,
            exog=exog_dict
        )
    if parse(skforecast_version) >= parse("0.17.0"):
        _ = runner.benchmark(
                ForecasterRecursiveMultiSeries__create_train_X_y_series_is_dict_exog_is_df_wide,
                forecaster=forecaster,
                series=series_dict,
                exog=exog_df_wide
            )
    if parse(skforecast_version) >= parse("0.17.0"):
        _ = runner.benchmark(
            ForecasterRecursiveMultiSeries__create_train_X_y_series_is_df_long_no_exog,
                forecaster=forecaster,
                series=series_df_long,
            )

    if parse(skforecast_version) >= parse("0.17.0"):
        _ = runner.benchmark(
                ForecasterRecursiveMultiSeries__create_train_X_y_series_is_df_long_exog_is_df_long,
                forecaster=forecaster,
                series=series_df_long,
                exog=exog_df_long
            )
    if parse(skforecast_version) >= parse("0.17.0"):
        _ = runner.benchmark(
                ForecasterRecursiveMultiSeries__create_train_X_y_series_is_df_long_exog_is_df_wide,
                forecaster=forecaster,
                series=series_df_long,
                exog=exog_df_wide
            )
    _ = runner.benchmark(
        ForecasterRecursiveMultiSeries__create_train_X_y_single_series,
            forecaster=forecaster,
            y = series_dict['series_0'],
            exog = exog_dict['series_0'],
        )

    # Fit
    # ==========================================================================
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=DummyRegressor(strategy='constant', constant=1.),
        lags=50,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
        encoding="ordinal"
    )

    runner = BenchmarkRunner(repeat=5, output_dir="./benchmarks")
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_fit_series_is_dict_no_exog,
            forecaster=forecaster,
            series=series_dict
        )
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_fit_series_is_dict_exog_is_dict,
            forecaster=forecaster,
            series=series_dict,
            exog=exog_dict
        )
    _ = runner.benchmark(
        ForecasterRecursiveMultiSeries_fit_series_is_dict_different_length_exog_is_dict,
            forecaster=forecaster,
            series=series_dict_different_length,
            exog=exog_dict
        )
    
    if parse(skforecast_version) >= parse("0.17.0"):
        _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_fit_series_is_dataframe_no_exog,
                forecaster=forecaster,
                series=series_df_long
            )
        
    if parse(skforecast_version) >= parse("0.17.0"):
        _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_fit_series_is_dataframe_exog_is_dataframe,
                forecaster=forecaster,
                series=series_df_long,
                exog=exog_df_long
            )
    if parse(skforecast_version) >= parse("0.17.0"):
        _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_fit_series_is_dataframe_exog_is_dict,
                forecaster=forecaster,
                series=series_df_long,
                exog=exog_dict
            )

    # Predict
    # ==========================================================================
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=DummyRegressor(strategy='constant', constant=1.),
        lags=50,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
        encoding="ordinal"
    )
    forecaster.fit(series=series_dict, exog=exog_dict, store_in_sample_residuals = True)

    runner = BenchmarkRunner(repeat=10, output_dir="./benchmarks")
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_check_predict_inputs,
            forecaster=forecaster,
            exog=exog_dict_pred
        )
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries__create_predict_inputs_exog_is_dict,
            forecaster=forecaster,
            exog=exog_dict_pred
        )
    
    if parse(skforecast_version) >= parse("0.17.0"):
        _ = runner.benchmark(
                ForecasterRecursiveMultiSeries__create_predict_inputs_exog_is_df_long,
                forecaster=forecaster,
                exog=exog_df_long_pred
            )
    
    if parse(skforecast_version) >= parse("0.17.0"):
        # NOTE: Only when the forecaster is fitted with a wide dataframe, the exogenous variables can be
        # passed as a wide dataframe.
        forecaster.fit(series=series_dict, exog=exog_df_wide, store_in_sample_residuals = True)
        _ = runner.benchmark(
                ForecasterRecursiveMultiSeries__create_predict_inputs_exog_is_df_wide,
                forecaster=forecaster,
                exog=exog_df_wide_pred
            )

    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_predict_exog_is_dict,
            forecaster=forecaster,
            exog=exog_dict_pred
        )
    
    if parse(skforecast_version) >= parse("0.17.0"):
        _ = runner.benchmark(
                ForecasterRecursiveMultiSeries_predict_exog_is_df_long,
                forecaster=forecaster,
                exog=exog_df_long_pred
            )
    
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_predict_interval_exog_is_dict_conformal,
            forecaster=forecaster,
            exog=exog_dict_pred
        )

    # Backtesting
    # ==========================================================================
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=DummyRegressor(strategy='constant', constant=1.),
        lags=50,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
        encoding="ordinal"
    )

    runner = BenchmarkRunner(repeat=5, output_dir="./benchmarks")
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_backtesting_series_is_dict_no_exog,
            forecaster=forecaster,
            series=series_dict
        )
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_backtesting_series_is_dict_exog_is_dict,
            forecaster=forecaster,
            series=series_dict,
            exog=exog_dict
        )
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_backtesting_series_is_dict_no_exog_conformal,
            forecaster=forecaster,
            series=series_dict
        )
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_backtesting_series_is_dict_exog_dict_conformal,
            forecaster=forecaster,
            series=series_dict,
            exog=exog_dict
        )
