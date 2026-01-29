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


def verify_all_benchmarks_completed(output_dir: str) -> bool:
    """
    Verify that all benchmarks completed successfully by checking for NaN values
    in the benchmark results.
    
    Parameters
    ----------
    output_dir : str
        Directory where benchmark.joblib is saved.
        
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
    
    # Check for NaN values in timing columns (indicates failed benchmarks)
    timing_cols = ['run_time_avg', 'run_time_median', 'run_time_p95', 'run_time_std']
    failed_benchmarks = df[df[timing_cols].isna().any(axis=1)]
    
    if len(failed_benchmarks) > 0:
        print("::error::The following benchmarks FAILED (contain NaN values):")
        for _, row in failed_benchmarks.iterrows():
            print(f"  - {row['function_name']}")
        return False
    
    return True


def main():
    """
    Run all benchmarks for skforecast.
    """

    output_dir = "benchmarks"

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

    # Verify all benchmarks completed successfully
    if not verify_all_benchmarks_completed(output_dir):
        print("::error::Not all benchmarks completed successfully!")
        raise SystemExit(1)
    
    print("All benchmarks completed successfully!")


if __name__ == "__main__":
    main()
