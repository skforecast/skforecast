# .github/scripts/run_benchmarks.py
import os
import numpy as np
from skforecast import __version__ as skf_version
from benchmarks.bench_forecaster_recursive import run_benchmark_ForecasterRecursive

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
    print(f"Running skforecast benchmarks (skforecast={skf_version})")
    run_benchmark_ForecasterRecursive()


if __name__ == "__main__":
    main()
