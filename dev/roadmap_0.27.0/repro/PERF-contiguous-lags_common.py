"""
PERF-contiguous-lags: helpers shared by scripts 09 to 11 (loaded with
`importlib`, the file name is not a valid module name).

- `environment()`: versions of Python and the packages involved, machine and
  power source (`pmset -g batt`).
- `wait_for_load(max_load)`: waits until the 1-minute load average is below
  `max_load` before a measurement, and stops the script if it does not drop
  within `max_wait` seconds (measuring with the machine loaded is not valid).
- `set_mode(value)`: the flag `_CONTIGUOUS_X_TRAIN` of the prototype in the
  four forecaster modules (`False`: 0.27.x, `True`: always copy, `"D"`: copy only
  for the estimators of option D).
"""

import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn

import skforecast.direct._forecaster_direct as m_direct
import skforecast.direct._forecaster_direct_multivariate as m_dmv
import skforecast.recursive._forecaster_recursive as m_rec
import skforecast.recursive._forecaster_recursive_classifier as m_cls

MODULES = (m_rec, m_cls, m_direct, m_dmv)
if not all(hasattr(m, "_CONTIGUOUS_X_TRAIN") for m in MODULES):
    sys.exit(
        "Prototype missing: git apply "
        "dev/roadmap_0.27.0/repro/PERF-contiguous-lags_prototype.diff"
    )


def set_mode(value) -> None:
    for m in MODULES:
        m._CONTIGUOUS_X_TRAIN = value


def power_source() -> str:
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return "unknown"
    if "AC Power" in out:
        return "AC"
    if "Battery Power" in out:
        return "battery"
    return "unknown"


def environment() -> str:
    versions = [
        f"Python {platform.python_version()}",
        f"numpy {np.__version__}",
        f"pandas {pd.__version__}",
        f"scikit-learn {sklearn.__version__}",
        f"scipy {scipy.__version__}",
    ]
    for name in ("lightgbm", "xgboost", "catboost"):
        try:
            versions.append(f"{name} {__import__(name).__version__}")
        except ImportError:
            pass
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except OSError:
        cpu = platform.processor()
    return (
        ", ".join(versions)
        + f"; {cpu}, {platform.platform()}; power: {power_source()}; "
        + "load %.2f %.2f %.2f" % os.getloadavg()
    )


def wait_for_load(max_load: float = 4.0, max_wait: float = 1800, step: float = 15):
    """Wait until the 1-minute load average is below `max_load`."""
    t0 = time.time()
    while os.getloadavg()[0] >= max_load:
        if time.time() - t0 > max_wait:
            sys.exit(
                f"Load {os.getloadavg()[0]:.1f} above {max_load} for more than "
                f"{max_wait:.0f} s: stopped without measuring."
            )
        time.sleep(step)
    return os.getloadavg()[0]


def make_y(n_rows: int, max_lag: int) -> pd.Series:
    """Same series as script 03: daily and weekly seasonality plus AR(1) noise."""
    rng = np.random.default_rng(123)
    n = n_rows + max_lag
    t = np.arange(n)
    noise = np.zeros(n)
    eps = rng.normal(scale=1.0, size=n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + eps[i]
    y = 50 + 10 * np.sin(2 * np.pi * t / 24) + 5 * np.sin(2 * np.pi * t / 168) + noise
    return pd.Series(y, index=pd.date_range("2000-01-01", periods=n, freq="h"), name="y")


def to_classes(y: pd.Series) -> pd.Series:
    q = y.quantile([1 / 3, 2 / 3]).to_numpy()
    return pd.Series(
        np.where(y < q[0], "low", np.where(y < q[1], "mid", "high")),
        index=y.index, name="y",
    )


def run_case(make_forecaster, fit, predict, on_value, min_time=0.2):
    """
    Time `fit(forecaster)` with the flag off and with `on_value`, alternating.

    `fit(forecaster)` runs the measured call, `predict(forecaster)` returns the
    values compared between the two versions (after the first fit). Each
    measurement is the mean of `inner` consecutive calls (at least `min_time`
    seconds); the number of rounds depends on the duration of one call.
    """
    import gc

    first, out, objs = {}, {}, {}
    for flag in (False, True):
        set_mode(on_value if flag else False)
        f = make_forecaster()
        t0 = time.perf_counter()
        fit(f)
        first[flag] = time.perf_counter() - t0
        out[flag] = np.asarray(predict(f), dtype=float)
        objs[flag] = f
    t_first = max(first.values())
    inner = max(1, int(min_time / max(t_first, 1e-6)))
    rounds = 7 if t_first < 0.5 else (5 if t_first < 3 else (3 if t_first < 15 else 2))
    times = {False: [], True: []}
    for _ in range(rounds):
        for flag in (False, True):
            set_mode(on_value if flag else False)
            f = objs[flag]
            gc.collect()
            t0 = time.perf_counter()
            for _ in range(inner):
                fit(f)
            times[flag].append((time.perf_counter() - t0) / inner)
    set_mode(on_value)
    diff = np.abs(out[True] - out[False])
    return dict(
        off_ms=min(times[False]) * 1e3, on_ms=min(times[True]) * 1e3,
        med_off_ms=float(np.median(times[False])) * 1e3,
        med_on_ms=float(np.median(times[True])) * 1e3,
        ratio=min(times[True]) / min(times[False]),
        max_abs_diff=float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0,
        identical=bool(np.array_equal(out[True], out[False], equal_nan=True)),
        t_first_s=t_first, inner=inner, rounds=rounds,
    )


def peak_memory(make_forecaster, fit, on_value, repeats=2):
    """Peak `tracemalloc` memory (MB) of `fit(forecaster)`, flag off and on."""
    import gc
    import tracemalloc

    peaks = {False: np.inf, True: np.inf}
    for _ in range(repeats):
        for flag in (False, True):
            set_mode(on_value if flag else False)
            f = make_forecaster()
            gc.collect()
            tracemalloc.start()
            fit(f)
            peak = tracemalloc.get_traced_memory()[1] / 2**20
            tracemalloc.stop()
            peaks[flag] = min(peaks[flag], peak)
            del f
    set_mode(on_value)
    return peaks[False], peaks[True]
