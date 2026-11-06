"""
Helpers shared by the experiments of this folder: versions of the environment,
synthetic series and a timer. Run the scripts from this folder.
"""
from __future__ import annotations
import importlib.metadata as md
import os
import platform
import resource
import sys
import time
import numpy as np
import pandas as pd

CHECKPOINTS = ["h2oai/timeh2o-v1-tiny", "h2oai/timeh2o-v1"]


def versions() -> str:
    """
    One line with the versions of the environment and the hardware.
    """
    import torch

    def v(package: str) -> str:
        try:
            return md.version(package)
        except md.PackageNotFoundError:
            return "-"

    return (
        f"Python {sys.version.split()[0]} | torch {torch.__version__} "
        f"| numpy {np.__version__} | pandas {pd.__version__} "
        f"| h2ofm {v('h2ofm')} | huggingface-hub {v('huggingface-hub')} "
        f"| skforecast {v('skforecast')} | threads {torch.get_num_threads()} "
        f"| cpus {os.cpu_count()} | {platform.processor() or platform.machine()} "
        f"| {platform.system()} {platform.release().split('-')[0]}"
    )


def peak_rss_gb() -> float:
    """
    Peak resident set size of this process, in GB (Linux reports KB).
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def hourly_series(n: int, seed: int = 0, level: float = 50.0) -> np.ndarray:
    """
    Hourly series with trend, daily and weekly seasonality and noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (
        level
        + 0.01 * t
        + 5 * np.sin(2 * np.pi * t / 24)
        + 2 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 0.5, n)
    )


def timeit(fn, repeat: int = 3) -> float:
    """
    Median wall time of `fn()` in seconds, after one warm-up call.
    """
    fn()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))
