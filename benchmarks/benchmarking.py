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
import plotly.express as px
import plotly.graph_objects as go
from packaging.version import parse as parse_version
from plotly.express.colors import qualitative


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
        regressor_name = type(forecaster.regressor).__name__ if forecaster else np.nan
        func_name = func.__name__
        hash_code = self.hash_function_code(func)
        timing = self.time_function(func, *args, **kwargs)
        system_info = self.get_system_info()

        print(f"Benchmarking function: {func_name} with skforecast version {skforecast.__version__}")
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

    fig = go.Figure()
    for version in sorted_versions:
        sub_df = df[df['skforecast_version'] == version]
        fig.add_trace(
            go.Scatter(
                x=sub_df['x_jittered'],
                y=sub_df['run_time_avg'],
                mode='markers',
                marker=dict(size=10, color=version_colors[version], opacity=0.7),
                error_y=dict(
                    type='data',
                    array=sub_df['run_time_std_dev'],
                    visible=True,
                    color=version_colors[version],  # color matches marker
                    thickness=1.5,
                    width=5
                ),
                name=f'Run time - {version}',
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
