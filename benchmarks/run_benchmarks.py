################################################################################
#                              Run Benchmarking                                #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import os

# Limitar hilos para reducir ruido en tiempos
if os.getenv('GITHUB_ACTIONS') != 'true':
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from skforecast import __version__ as skforecast_version
from benchmarks import (
    run_benchmark_ForecasterRecursive,
    run_benchmark_ForecasterRecursiveClassifier,
    run_benchmark_ForecasterRecursiveMultiSeries,
    run_benchmark_ForecasterStats,
    run_benchmark_ForecasterDirect,
    run_benchmark_ForecasterDirectMultiVariate
)

# Fijar semillas reproducibles
np.random.seed(123)


def verify_all_benchmarks_completed(output_dir: str, start_time: pd.Timestamp) -> bool:
    """
    Verify that all benchmarks from the current run completed successfully 
    by checking for NaN values in the benchmark results.
    
    Parameters
    ----------
    output_dir : str
        Directory where benchmark.joblib is saved.
    start_time : pd.Timestamp
        Timestamp when the current benchmark run started. Only benchmarks
        with datetime >= start_time will be verified.
        
    Returns
    -------
    bool
        True if all benchmarks completed successfully, False otherwise.
    """
    result_file = f"{output_dir}/benchmark.joblib"
    
    if not pd.io.common.file_exists(result_file):
        print("::error::Benchmark file not found!")
        return False
    
    df = joblib.load(result_file)
    
    # Filter only benchmarks from the current run (started after start_time)
    df_current_run = df[df['datetime'] >= start_time]
    
    if len(df_current_run) == 0:
        print("::error::No benchmarks found for the current run!")
        return False
    
    # Check for NaN values in timing columns (indicates failed benchmarks)
    timing_cols = ['run_time_avg', 'run_time_median', 'run_time_p95', 'run_time_std']
    failed_benchmarks = df_current_run[df_current_run[timing_cols].isna().any(axis=1)]
    
    if len(failed_benchmarks) > 0:
        print("::error::The following benchmarks FAILED (contain NaN values):")
        for _, row in failed_benchmarks.iterrows():
            print(f"  - {row['function_name']}")
        return False
    
    print(f"Verified {len(df_current_run)} benchmarks from current run.")
    return True


def main():
    """
    Run all benchmarks for skforecast.
    """

    output_dir = "benchmarks"
    
    # Record start time to filter only current run benchmarks
    start_time = pd.Timestamp.now()

    print(
        f"Running skforecast benchmarks (skforecast={skforecast_version}), "
        f"output will be saved in '{output_dir}'"
    )
    
    run_benchmark_ForecasterRecursive(output_dir)
    run_benchmark_ForecasterRecursiveClassifier(output_dir)
    run_benchmark_ForecasterRecursiveMultiSeries(output_dir)
    run_benchmark_ForecasterStats(output_dir)
    run_benchmark_ForecasterDirect(output_dir)
    run_benchmark_ForecasterDirectMultiVariate(output_dir)

    # Verify all benchmarks completed successfully (only from current run)
    if not verify_all_benchmarks_completed(output_dir, start_time):
        print("::error::Not all benchmarks completed successfully!")
        raise SystemExit(1)
    
    print("All benchmarks completed successfully!")


if __name__ == "__main__":
    main()
