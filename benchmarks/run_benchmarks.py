################################################################################
#                              Run Benchmarking                                #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import os
import numpy as np
from skforecast import __version__ as skforecast_version
from benchmarks import (
    run_benchmark_ForecasterRecursive,
    run_benchmark_ForecasterRecursiveMultiSeries,
    run_benchmark_ForecasterDirect,
    run_benchmark_ForecasterDirectMultiVariate
)

# Fijar semillas reproducibles
np.random.seed(123)

# Limitar hilos para reducir ruido en tiempos
if not os.getenv('GITHUB_ACTIONS') == 'true':
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def main():
    """
    Run all benchmarks for skforecast.
    """

    output_dir = "benchmarks"

    print(
        f"Running skforecast benchmarks (skforecast={skforecast_version}), "
        f"output will be saved in {output_dir}"
    )
    
    run_benchmark_ForecasterRecursive(output_dir)
    # run_benchmark_ForecasterRecursiveMultiSeries(output_dir)
    # run_benchmark_ForecasterDirect(output_dir)
    # run_benchmark_ForecasterDirectMultiVariate(output_dir)


if __name__ == "__main__":
    main()
