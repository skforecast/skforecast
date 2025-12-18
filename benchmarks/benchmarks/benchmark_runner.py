################################################################################
#                                BenchmarkRunner                               #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import time
import platform
import joblib
import os
import hashlib
import inspect
import psutil
import sklearn
import numpy as np
import pandas as pd
import lightgbm
import skforecast


class BenchmarkRunner:
    """"
    Class to run benchmarks on skforecast forecasters and save the results.
    """
    def __init__(self, output_dir="./benchmarks", repeat=10):

        self.results_filename = "benchmark.joblib"
        self.output_dir = output_dir
        self.repeat = repeat
        self._system_info_cache = None

        os.makedirs(self.output_dir, exist_ok=True)

    def get_system_info(self):
        """
        Collect system and package version information (cached except datetime).
        """
        if self._system_info_cache is None:
            self._system_info_cache = {
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
        return {
            'datetime': pd.Timestamp.now(),
            **self._system_info_cache
        }

    def hash_function_code(self, func):
        """
        Generate MD5 hash of a function's source code.
        """
        src = inspect.getsource(func)
        return hashlib.md5(src.encode()).hexdigest()

    def time_function(self, func, *args, **kwargs):
        """
        Measure execution time of a function over multiple runs.
        """
        times = []
        try:
            for _ in range(self.repeat):
                start = time.perf_counter()
                func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)

            return {
                'avg_time': np.mean(times), 
                'median_time': np.median(times),
                'p95_time': np.percentile(times, 95),
                'std_time': np.std(times)
            }
        except Exception as e:
            print(f"::warning::Benchmark FAILED - {func.__name__}: {e}")
            return {
                'avg_time': np.nan, 
                'median_time': np.nan, 
                'p95_time': np.nan, 
                'std_time': np.nan
            }

    def benchmark(self, func, forecaster=None, allow_repeated_execution=True, *args, **kwargs):
        """
        Benchmark a function by measuring its execution time and saving the results to a file.
        """
        forecaster_name = type(forecaster).__name__ if forecaster else np.nan
        if forecaster_name == 'ForecasterRnn':
            estimator_name = forecaster.estimator.name if forecaster else np.nan
        else:
            estimator_name = type(forecaster.estimator).__name__ if forecaster else np.nan
        
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        func_name = func.__name__
        hash_code = self.hash_function_code(func)
        method_name = func_name.replace(f'{forecaster_name}_', '') 
        system_info = self.get_system_info()

        print(f"[{timestamp}] Benchmarking function: {func_name}", flush=True)
        timing = self.time_function(func, forecaster, *args, **kwargs)

        entry = {
            'forecaster_name': forecaster_name,
            'estimator_name': estimator_name,
            'function_name': func_name,
            'function_hash': hash_code,
            'method_name': method_name,
            'run_time_avg': timing['avg_time'],
            'run_time_median': timing['median_time'],
            'run_time_p95': timing['p95_time'],
            'run_time_std': timing['std_time'],
            'n_repeats': self.repeat,
            **system_info
        }

        result_file = os.path.join(self.output_dir, self.results_filename)
        df_new = pd.DataFrame([entry])
        if os.path.exists(result_file):
            df_existing = joblib.load(result_file)
            if not allow_repeated_execution:
                cols_to_ignore = [
                    'run_time_avg', 'run_time_median', 'run_time_p95', 
                    'run_time_std', 'n_repeats', 'datetime'
                ]
                mask = (
                    df_existing
                    .drop(columns = cols_to_ignore)
                    .eq(df_new.drop(columns = cols_to_ignore).loc[0, :])
                    .all(axis=1)
                )
                if mask.any():
                    print(
                        f"::notice::Benchmark skipped: identical entry already exists in {result_file}"
                    )
                    return df_existing
            
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            joblib.dump(df_combined, result_file)
            print(f"Appended new benchmark entry to existing file: {result_file}")

            return df_combined
        
        else:
            joblib.dump(df_new, result_file)
            print(f"Created new benchmark file: {result_file}")

            return df_new
