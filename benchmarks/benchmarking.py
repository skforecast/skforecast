################################################################################
#                                 Benchmarking                                 #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import time
import platform
import joblib
import os
import warnings
import hashlib
import inspect
import psutil
import sklearn
import numpy as np
import pandas as pd
import lightgbm
import skforecast
import plotly.graph_objects as go
from packaging.version import parse as parse_version
from plotly.express.colors import qualitative

from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from skforecast.utils import *
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.deep_learning import create_and_compile_model, ForecasterRnn
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster, backtesting_forecaster_multiseries


class BenchmarkRunner:
    def __init__(self, output_dir="./", repeat=10):
        self.output_dir = output_dir
        self.repeat = repeat
        os.makedirs(self.output_dir, exist_ok=True)

    def get_system_info(self):
        return {
            'datetime': pd.Timestamp.now(),
            'python_version': platform.python_version(),
            'skforecast_version': skforecast.__version__,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'sklearn_version': sklearn.__version__,
            'lightgbm_version': lightgbm.__version__,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(logical=True),
            'memory_gb': round(psutil.virtual_memory().total / 1e9, 2),
        }

    def hash_function_code(self, func):
        src = inspect.getsource(func)
        return hashlib.md5(src.encode()).hexdigest()

    def time_function(self, func, *args, **kwargs):
        times = []
        try:
            for _ in range(self.repeat):
                start = time.perf_counter()
                func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)

            return {'avg_time': np.mean(times), 'std_dev': np.std(times)}
        except Exception as e:
            warnings.warn(f"The function {func.__name__} raised an exception: {e}")
            return {'avg_time': np.nan, 'std_dev': np.nan}

    def benchmark(self, func, forecaster=None, allow_repeated_execution=True, *args, **kwargs):
        """
        Benchmark a function by measuring its execution time and saving the results to a file.
        """
        forecaster_name = type(forecaster).__name__ if forecaster else np.nan
        if forecaster_name == 'ForecasterRnn':
            regressor_name = forecaster.regressor.name if forecaster else np.nan
        else:
            regressor_name = type(forecaster.regressor).__name__ if forecaster else np.nan
        func_name = func.__name__
        hash_code = self.hash_function_code(func)
        timing = self.time_function(func, forecaster, *args, **kwargs)
        system_info = self.get_system_info()

        print(f"Benchmarking function: {func_name}")
        entry = {
            'forecaster_name': forecaster_name,
            'regressor_name': regressor_name,
            'function_name': func_name,
            'function_hash': hash_code,
            'run_time_avg': timing['avg_time'],
            'run_time_std_dev': timing['std_dev'],
            **system_info
        }

        result_file = os.path.join(self.output_dir, "benchmark.joblib")
        df_new = pd.DataFrame([entry])
        if os.path.exists(result_file):
            df_existing = joblib.load(result_file)
            if not allow_repeated_execution:
                cols_to_ignore = ['run_time_avg', 'run_time_std_dev', 'datetime']
                mask = (
                    df_existing
                    .drop(columns = cols_to_ignore)
                    .eq(df_new.drop(columns = cols_to_ignore).loc[0, :])
                    .all(axis=1)
                )
                if mask.any():
                    warnings.warn(
                        "This benchmark already exists with the same hash and system info. Skipping save."
                    )
                    return df_existing
            
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            joblib.dump(df_combined, result_file)

            return df_combined
        
        else:
            joblib.dump(df_new, result_file)

            return df_new


def plot_benchmark_results(df, function_name, add_median=True, add_mean=True):
    """
    Plot benchmark results with jittered strip points and per-point standard deviation error bars,
    with error bars matching point colors.
    """

    np.random.seed(42164)
    sorted_versions = sorted(df['skforecast_version'].unique(), key=parse_version)
    df['skforecast_version'] = pd.Categorical(df['skforecast_version'], categories=sorted_versions, ordered=True)
    df = df.sort_values("skforecast_version")
    version_to_num = {v: i for i, v in enumerate(sorted_versions)}
    df['x_jittered'] = df['skforecast_version'].map(version_to_num).astype(float) + np.random.uniform(-0.05, 0.05, size=len(df))
    version_colors = {
        v: qualitative.Plotly[i % len(qualitative.Plotly)] 
        for i, v in enumerate(sorted_versions)
    }
    platform_symbols = {
        pltf: symbol for pltf, symbol in zip(df['platform'].unique(), [
            'circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star'
        ])
    }

    fig = go.Figure()
    for version in sorted_versions:
        for pltf, symbol in platform_symbols.items():
            sub_df = df[(df['skforecast_version'] == version) & (df['platform'] == pltf)]
            if sub_df.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub_df['x_jittered'],
                    y=sub_df['run_time_avg'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=version_colors[version],
                        symbol=symbol,
                        opacity=0.7
                    ),
                    error_y=dict(
                        type='data',
                        array=sub_df['run_time_std_dev'],
                        visible=True,
                        color=version_colors[version],
                        thickness=1.5,
                        width=5
                    ),
                    name=f'{version} - {pltf}',
                    text=sub_df.apply(lambda row: (
                        f"Forecaster: {row['forecaster_name']}<br>"
                        f"Regressor: {row['regressor_name']}<br>"
                        f"Function: {row['function_name']}<br>"
                        f"Function_hash: {row['function_hash']}<br>"
                        f"Datetime: {row['datetime']}<br>"
                        f"Python version: {row['python_version']}<br>"
                        f"skforecast version: {row['skforecast_version']}<br>"
                        f"numpy version: {row['numpy_version']}<br>"
                        f"pandas version: {row['pandas_version']}<br>"
                        f"sklearn version: {row['sklearn_version']}<br>"
                        f"lightgbm version: {row['lightgbm_version']}<br>"
                        f"Platform: {row['platform']}<br>"
                        f"Processor: {row['processor']}<br>"
                        f"CPU count: {row['cpu_count']}<br>"
                        f"Memory (GB): {row['memory_gb']:.2f}<br>"
                        f"Run time avg: {row['run_time_avg']:.4f} seconds<br>"
                        f"Run time std dev: {row['run_time_std_dev']:.4f} seconds"
                    ), axis=1),
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=True
                )
            )

    if add_median:
        medians = df.groupby('skforecast_version', observed=True)['run_time_avg'].median().reset_index()
        medians['x_center'] = medians['skforecast_version'].map(version_to_num)
        fig.add_trace(
            go.Scatter(
                x=medians['x_center'],
                y=medians['run_time_avg'],
                mode='lines+markers',
                line=dict(color='black', width=2),
                marker=dict(size=8, color='black'),
                name='Median (per version)'
            )
        )

    if add_mean:
        means = df.groupby('skforecast_version', observed=True)['run_time_avg'].mean().reset_index()
        means['x_center'] = means['skforecast_version'].map(version_to_num)
        fig.add_trace(
            go.Scatter(
                x=means['x_center'],
                y=means['run_time_avg'],
                mode='lines+markers',
                line=dict(color='black', width=2, dash='dash'),
                marker=dict(size=8, color='black'),
                name='Mean (per version)'
            )
        )

    fig.update_layout(
        title=f"Execution time of {function_name}",
        xaxis=dict(
            tickmode='array',
            tickvals=list(version_to_num.values()),
            ticktext=list(version_to_num.keys()),
            title='skforecast version',
            tickangle=-45
        ),
        yaxis_title='Execution time (seconds)',
        showlegend=True
    )
    fig.show()


def run_benchmark_ForecasterRecursiveMultiSeries(
    series_dict,
    series_df_long,
    series_dict_different_length,
    exog_dict,
    exog_df_long,
    exog_df_wide,
    exog_dict_prediction,
    exog_df_long_prediction,
    exog_df_wide_prediction
):
    """
    Run all benchmarks for the ForecasterRecursiveMultiSeries class and save the results.
    """

    print("Running benchmarks for ForecasterRecursiveMultiSeries...")
    print("skforecast version:", skforecast.__version__)

    # Fit
    # --------------------------------------------------------------------------
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

    # _create_train_X_y and create_train_X_y_single_series
    # --------------------------------------------------------------------------
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

    # Predict
    # --------------------------------------------------------------------------
    def ForecasterRecursiveMultiSeries_predict_exog_is_dict(forecaster, exog):
        forecaster.predict(steps=100, exog=exog, suppress_warnings=True)

    def ForecasterRecursiveMultiSeries_predict_exog_is_df_long(forecaster, exog):
        forecaster.predict(steps=100, exog=exog, suppress_warnings=True)

    def ForecasterRecursiveMultiSeries_predict_interval_exog_is_dict_conformal(forecaster, exog):
        forecaster.predict_interval(
            steps=100,
            exog=exog,
            method='conformal',
            interval=[5, 95],
            suppress_warnings=True
        )

    def ForecasterRecursiveMultiSeries__create_predict_inputs_exog_is_dict(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = 100,
                exog         = exog,
                check_inputs = True
            )

    def ForecasterRecursiveMultiSeries__create_predict_inputs_exog_is_df_long(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = 100,
                exog         = exog,
                check_inputs = True
            )
        
    def ForecasterRecursiveMultiSeries__create_predict_inputs_exog_is_df_wide(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = 100,
                exog         = exog,
                check_inputs = True
            )

    def ForecasterRecursiveMultiSeries__check_predict_inputs(forecaster, exog):
        check_predict_input(
            forecaster_name  = type(forecaster).__name__,
            steps            = 100,
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

    # Backtesting
    # --------------------------------------------------------------------------
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
    # --------------------------------------------------------------------------
    regressor = LGBMRegressor(random_state=8520, verbose=-1)
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=regressor,
        lags=50,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
        encoding="ordinal"
    )

    runner = BenchmarkRunner(repeat=10, output_dir="./")
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
    if skforecast.__version__ >= "0.17.0":
        _ = runner.benchmark(
                ForecasterRecursiveMultiSeries__create_train_X_y_series_is_dict_exog_is_df_wide,
                forecaster=forecaster,
                series=series_dict,
                exog=exog_df_wide
            )
    if skforecast.__version__ >= "0.17.0":
        _ = runner.benchmark(
            ForecasterRecursiveMultiSeries__create_train_X_y_series_is_df_long_no_exog,
                forecaster=forecaster,
                series=series_df_long,
            )

    if skforecast.__version__ >= "0.17.0":
        _ = runner.benchmark(
                ForecasterRecursiveMultiSeries__create_train_X_y_series_is_df_long_exog_is_df_long,
                forecaster=forecaster,
                series=series_df_long,
                exog=exog_df_long
            )
    if skforecast.__version__ >= "0.17.0":
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
    # --------------------------------------------------------------------------
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=regressor,
        lags=50,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
        encoding="ordinal"
    )

    runner = BenchmarkRunner(repeat=5, output_dir="./")
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
    
    if skforecast.__version__ >= "0.17.0":
        _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_fit_series_is_dataframe_no_exog,
                forecaster=forecaster,
                series=series_df_long
            )
        
    if skforecast.__version__ >= "0.17.0":
        _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_fit_series_is_dataframe_exog_is_dataframe,
                forecaster=forecaster,
                series=series_df_long,
                exog=exog_df_long
            )
    if skforecast.__version__ >= "0.17.0":
        _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_fit_series_is_dataframe_exog_is_dict,
                forecaster=forecaster,
                series=series_df_long,
                exog=exog_dict
            )

    # Predict
    # --------------------------------------------------------------------------
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=regressor,
        lags=50,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
        encoding="ordinal"
    )
    forecaster.fit(series=series_dict, exog=exog_dict, store_in_sample_residuals = True)

    runner = BenchmarkRunner(repeat=10, output_dir="./")
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_predict_exog_is_dict,
            forecaster=forecaster,
            exog=exog_dict_prediction
        )
    if skforecast.__version__ >= "0.17.0":
        _ = runner.benchmark(
                ForecasterRecursiveMultiSeries_predict_exog_is_df_long,
                forecaster=forecaster,
                exog=exog_df_long_prediction
            )   
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries_predict_interval_exog_is_dict_conformal,
            forecaster=forecaster,
            exog=exog_dict_prediction
        )
    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries__create_predict_inputs_exog_is_dict,
            forecaster=forecaster,
            exog=exog_dict_prediction
        )
    if skforecast.__version__ >= "0.17.0":
        _ = runner.benchmark(
                ForecasterRecursiveMultiSeries__create_predict_inputs_exog_is_df_long,
                forecaster=forecaster,
                exog=exog_df_long_prediction
            )

    _ = runner.benchmark(
            ForecasterRecursiveMultiSeries__check_predict_inputs,
            forecaster=forecaster,
            exog=exog_dict_prediction
        )
    
    if skforecast.__version__ >= "0.17.0":
        # NOTE: Only when the forecaster is fitted with a wide dataframe, the exogenous variables can be
        # passed as a wide dataframe.
        forecaster.fit(series=series_dict, exog=exog_df_wide, store_in_sample_residuals = True)
        _ = runner.benchmark(
                ForecasterRecursiveMultiSeries__create_predict_inputs_exog_is_df_wide,
                forecaster=forecaster,
                exog=exog_df_wide_prediction
            )

    # Backtesting
    # --------------------------------------------------------------------------
    forecaster = ForecasterRecursiveMultiSeries(
        regressor=regressor,
        lags=50,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
        encoding="ordinal"
    )

    runner = BenchmarkRunner(repeat=5, output_dir="./")
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
    

def run_benchmark_ForecasterRecursive(
    y,
    exog,
    exog_prediction
):
    """
    Run all benchmarks for the ForecasterRecursive class and save the results.
    """

    regressor = LGBMRegressor(random_state=8520, verbose=-1)
    forecaster = ForecasterRecursive(
        regressor=regressor,
        lags=50,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    def ForecasterRecursive_fit(forecaster, y, exog):
        forecaster.fit(y=y, exog=exog)

    def ForecasterRecursive__create_train_X_y(forecaster, y, exog):
        forecaster._create_train_X_y(y=y, exog=exog)

    def ForecasterRecursive_predict(forecaster, exog):
        forecaster.predict(steps=100, exog=exog)

    def ForecasterRecursive_predict_interval_conformal(forecaster, exog):
        forecaster.predict_interval(
            steps=100,
            exog=exog,
            interval=[5, 95],
            method='conformal'
        )

    def ForecasterRecursive__create_predict_inputs(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = 100,
                exog         = exog,
                check_inputs = True
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

                show_progress=False
            )

    runner = BenchmarkRunner(repeat=30, output_dir="./")
    _ = runner.benchmark(ForecasterRecursive__create_train_X_y, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=10, output_dir="./")
    _ = runner.benchmark(ForecasterRecursive_fit, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=30, output_dir="./")
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    _ = runner.benchmark(ForecasterRecursive_predict, forecaster=forecaster, exog=exog_prediction)
    _ = runner.benchmark(ForecasterRecursive_predict_interval_conformal, forecaster=forecaster, exog=exog_prediction)
    _ = runner.benchmark(ForecasterRecursive__create_predict_inputs, forecaster=forecaster, exog=exog_prediction)

    runner = BenchmarkRunner(repeat=5, output_dir="./")
    _ = runner.benchmark(ForecasterRecursive_backtesting, forecaster=forecaster, y=y, exog=exog)
    _ = runner.benchmark(ForecasterRecursive_backtesting_conformal, forecaster=forecaster, y=y, exog=exog)


def run_benchmark_ForecasterDirectMultiVariate(
    series,
    exog,
    exog_prediction
):
    """
    Run all benchmarks for the ForecasterDirectMultiVariate class and save the results.
    """

    regressor = LGBMRegressor(random_state=8520, verbose=-1)
    forecaster = ForecasterDirectMultiVariate(
        regressor=regressor,
        level='series_1',
        steps=5,
        lags=20,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    def ForecasterDirectMultiVariate_fit(forecaster, series, exog):
        forecaster.fit(series=series, exog=exog)

    def ForecasterDirectMultiVariate__create_train_X_y(forecaster, series, exog):
        forecaster._create_train_X_y(series=series, exog=exog)

    def ForecasterDirectMultiVariate_fit_series_no_exog(forecaster, series):
        forecaster.fit(series=series)

    def ForecasterDirectMultiVariate__create_train_X_y_no_exog(forecaster, series):
        forecaster._create_train_X_y(series=series)

    def ForecasterDirectMultiVariate_predict(forecaster, exog):
        forecaster.predict(steps=5, exog=exog, suppress_warnings=True)

    def ForecasterDirectMultiVariate_predict_interval_conformal(forecaster, exog):
        forecaster.predict_interval(
            steps=5,
            exog=exog,
            method='conformal',
            interval=[5, 95],
            suppress_warnings=True
        )

    def ForecasterDirectMultiVariate__create_predict_inputs(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = 5,
                exog         = exog,
                check_inputs = True
            )

    def ForecasterDirectMultiVariate__check_predict_inputs(forecaster, exog):
        check_predict_input(
            forecaster_name  = type(forecaster).__name__,
            steps            = 5,
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

    def ForecasterDirectMultiVariate_backtesting(forecaster, series, exog):
        cv = TimeSeriesFold(
                initial_train_size=900,
                fixed_train_size=True,
                steps=5,
            )
        _ = backtesting_forecaster_multiseries(
                forecaster=forecaster,
                series=series,
                exog=exog,
                cv=cv,
                metric='mean_squared_error',
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
                show_progress=False
            )
            
    def ForecasterDirectMultiVariate_backtesting_conformal(forecaster, series, exog):
        cv = TimeSeriesFold(
                initial_train_size=900,
                fixed_train_size=True,
                steps=5,
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

    runner = BenchmarkRunner(repeat=10, output_dir="./")
    _ = runner.benchmark(ForecasterDirectMultiVariate__create_train_X_y, forecaster=forecaster, series=series, exog=exog)
    _ = runner.benchmark(ForecasterDirectMultiVariate__create_train_X_y_no_exog, forecaster=forecaster, series=series)

    runner = BenchmarkRunner(repeat=5, output_dir="./")
    _ = runner.benchmark(ForecasterDirectMultiVariate_fit, forecaster=forecaster, series=series, exog=exog)
    _ = runner.benchmark(ForecasterDirectMultiVariate_fit_series_no_exog, forecaster=forecaster, series=series)

    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    runner = BenchmarkRunner(repeat=10, output_dir="./")
    _ = runner.benchmark(ForecasterDirectMultiVariate_predict, forecaster=forecaster, exog=exog_prediction)
    _ = runner.benchmark(ForecasterDirectMultiVariate_predict_interval_conformal, forecaster=forecaster, exog=exog_prediction)
    _ = runner.benchmark(ForecasterDirectMultiVariate__create_predict_inputs, forecaster=forecaster, exog=exog_prediction)
    _ = runner.benchmark(ForecasterDirectMultiVariate__check_predict_inputs, forecaster=forecaster, exog=exog_prediction)

    runner = BenchmarkRunner(repeat=5, output_dir="./")
    _ = runner.benchmark(ForecasterDirectMultiVariate_backtesting, forecaster=forecaster, series=series, exog=exog)
    _ = runner.benchmark(ForecasterDirectMultiVariate_backtesting_no_exog, forecaster=forecaster, series=series)
    _ = runner.benchmark(ForecasterDirectMultiVariate_backtesting_conformal, forecaster=forecaster, series=series, exog=exog)


def run_benchmark_ForecasterDirect(
    y,
    exog,
    exog_prediction
):
    """
    Run all benchmarks for the ForecasterDirect class and save the results.
    """

    regressor = LGBMRegressor(random_state=8520, verbose=-1)
    forecaster = ForecasterDirect(
        regressor=regressor,
        steps=5,
        lags=20,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    def ForecasterDirect_fit(forecaster, y, exog):
        forecaster.fit(y=y, exog=exog)

    def ForecasterDirect__create_train_X_y(forecaster, y, exog):
        forecaster._create_train_X_y(y=y, exog=exog)

    def ForecasterDirect_predict(forecaster, exog):
        forecaster.predict(steps=5, exog=exog)

    def ForecasterDirect_predict_interval_conformal(forecaster, exog):
        forecaster.predict_interval(
            steps=5,
            exog=exog,
            interval=[5, 95],
            method='conformal'
        )

    def ForecasterDirect__create_predict_inputs(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = 5,
                exog         = exog,
                check_inputs = True
            )
        
    def ForecasterDirect_backtesting(forecaster, y, exog):
        cv = TimeSeriesFold(
                initial_train_size=1200,
                fixed_train_size=True,
                steps=5,
            )
        _ = backtesting_forecaster(
                forecaster=forecaster,
                y=y,
                exog=exog,
                cv=cv,
                metric='mean_squared_error',
                show_progress=False
            )

    def ForecasterDirect_backtesting_conformal(forecaster, y, exog):
        cv = TimeSeriesFold(
                initial_train_size=1200,
                fixed_train_size=True,
                steps=5,
            )
        _ = backtesting_forecaster(
                forecaster=forecaster,
                y=y,
                exog=exog,
                interval=[5, 95],
                interval_method='conformal',
                cv=cv,
                metric='mean_squared_error',

                show_progress=False
            )

    runner = BenchmarkRunner(repeat=30, output_dir="./")
    _ = runner.benchmark(ForecasterDirect__create_train_X_y, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=10, output_dir="./")
    _ = runner.benchmark(ForecasterDirect_fit, forecaster=forecaster, y=y, exog=exog)

    runner = BenchmarkRunner(repeat=30, output_dir="./")
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    _ = runner.benchmark(ForecasterDirect_predict, forecaster=forecaster, exog=exog_prediction)
    _ = runner.benchmark(ForecasterDirect_predict_interval_conformal, forecaster=forecaster, exog=exog_prediction)
    _ = runner.benchmark(ForecasterDirect__create_predict_inputs, forecaster=forecaster, exog=exog_prediction)

    runner = BenchmarkRunner(repeat=5, output_dir="./")
    _ = runner.benchmark(ForecasterDirect_backtesting, forecaster=forecaster, y=y, exog=exog)
    _ = runner.benchmark(ForecasterDirect_backtesting_conformal, forecaster=forecaster, y=y, exog=exog)
    

def run_benchmark_ForecasterRnn(
    series,
    exog,
    exog_prediction
):
    """
    Run all benchmarks for the ForecasterRnn class and save the results.
    """

    model = create_and_compile_model(
        series=series, 
        lags=10, 
        steps=5, 
        recurrent_layer="LSTM", 
        recurrent_units=64,
        dense_units=64, 
        model_name="benchmark_no_exog"
    )

    model_exog = create_and_compile_model(
        series=series, 
        exog=exog,
        lags=10, 
        steps=5, 
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

    def ForecasterRnn_fit(forecaster, series, exog):
        forecaster.fit(series=series, exog=exog)

    def ForecasterRnn__create_train_X_y(forecaster, series, exog):
        forecaster._create_train_X_y(series=series, exog=exog)

    def ForecasterRnn_fit_series_no_exog(forecaster, series):
        forecaster.fit(series=series)

    def ForecasterRnn__create_train_X_y_no_exog(forecaster, series):
        forecaster._create_train_X_y(series=series)

    def ForecasterRnn_predict(forecaster, exog):
        forecaster.predict(steps=5, exog=exog, suppress_warnings=True)

    def ForecasterRnn_predict_interval_conformal(forecaster, exog):
        forecaster.predict_interval(
            steps=5,
            exog=exog,
            method='conformal',
            interval=[5, 95],
            suppress_warnings=True
        )

    def ForecasterRnn__create_predict_inputs(forecaster, exog):
        _ = forecaster._create_predict_inputs(
                steps        = 5,
                exog         = exog,
                check_inputs = True
            )

    def ForecasterRnn__check_predict_inputs(forecaster, exog):
        check_predict_input(
            forecaster_name   = type(forecaster).__name__,
            steps             = 5,
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

    def ForecasterRnn_backtesting(forecaster, series, exog):
        cv = TimeSeriesFold(
                initial_train_size=900,
                fixed_train_size=True,
                steps=5,
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
                steps=5,
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
                steps=5,
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

    runner = BenchmarkRunner(repeat=10, output_dir="./")
    _ = runner.benchmark(ForecasterRnn__create_train_X_y, forecaster=forecaster_exog, series=series, exog=exog)
    _ = runner.benchmark(ForecasterRnn__create_train_X_y_no_exog, forecaster=forecaster, series=series)

    runner = BenchmarkRunner(repeat=5, output_dir="./")
    _ = runner.benchmark(ForecasterRnn_fit, forecaster=forecaster_exog, series=series, exog=exog)
    _ = runner.benchmark(ForecasterRnn_fit_series_no_exog, forecaster=forecaster, series=series)

    forecaster_exog.fit(series=series, exog=exog, store_in_sample_residuals=True)
    runner = BenchmarkRunner(repeat=10, output_dir="./")
    _ = runner.benchmark(ForecasterRnn_predict, forecaster=forecaster_exog, exog=exog_prediction)
    _ = runner.benchmark(ForecasterRnn_predict_interval_conformal, forecaster=forecaster_exog, exog=exog_prediction)
    _ = runner.benchmark(ForecasterRnn__create_predict_inputs, forecaster=forecaster_exog, exog=exog_prediction)
    _ = runner.benchmark(ForecasterRnn__check_predict_inputs, forecaster=forecaster_exog, exog=exog_prediction)

    runner = BenchmarkRunner(repeat=5, output_dir="./")
    _ = runner.benchmark(ForecasterRnn_backtesting, forecaster=forecaster_exog, series=series, exog=exog)
    _ = runner.benchmark(ForecasterRnn_backtesting_no_exog, forecaster=forecaster, series=series)
    _ = runner.benchmark(ForecasterRnn_backtesting_conformal, forecaster=forecaster_exog, series=series, exog=exog)
