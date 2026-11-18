"""
Shared setup of the ForecasterRecursiveMultiSeries measurements. Import it
first: it blocks pyarrow and pins the process before pandas is imported.

Env: NO_PYARROW=1 blocks pyarrow; BENCH_CPU=<n> pins the process to one core
and raises its priority. The tree under test is the one on PYTHONPATH.
"""
import os
import sys

if os.environ.get("NO_PYARROW") == "1":
    class _BlockPyarrow:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")
            return None

    assert "pyarrow" not in sys.modules and "pandas" not in sys.modules
    sys.meta_path.insert(0, _BlockPyarrow)

if os.environ.get("BENCH_CPU"):
    import ctypes

    _k = ctypes.windll.kernel32
    _k.GetCurrentProcess.restype = ctypes.c_void_p
    _k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    assert _k.SetProcessAffinityMask(
        _k.GetCurrentProcess(), 1 << int(os.environ["BENCH_CPU"])
    )
    _k.SetPriorityClass(_k.GetCurrentProcess(), 0x00000080)

import warnings

import numpy as np
import pandas as pd
import skforecast
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from skforecast.model_selection import (
    TimeSeriesFold, backtesting_forecaster_multiseries,
)
from skforecast.recursive import ForecasterRecursiveMultiSeries

warnings.simplefilter("ignore")

LAGS = 24
N_EXOG = 3


def env_info():
    if pd.__version__ >= "3":
        storage = pd.Series(["a"]).dtype.storage
    else:
        storage = "object"
    return {
        "pandas": pd.__version__, "storage": storage, "numpy": np.__version__,
        "pyarrow_loaded": "pyarrow" in sys.modules,
        "python": sys.version.split()[0],
        "tree": os.path.dirname(os.path.dirname(skforecast.__file__)),
    }


def series_names(n_series):
    # Names like the ones of real data: an identifier, not one or two letters.
    return [f"series_{i:04d}" for i in range(n_series)]


def make_wide(n_series, n_rows, seed=123):
    """Series as a wide DataFrame and one exog DataFrame shared by all series."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="h")
    series = pd.DataFrame(
        np.cumsum(rng.normal(size=(n_rows, n_series)), axis=0) + 100,
        index=idx, columns=series_names(n_series),
    )
    exog = pd.DataFrame(
        rng.normal(size=(n_rows, N_EXOG)), index=idx,
        columns=[f"exog_{i}" for i in range(N_EXOG)],
    )
    return series, exog


def to_format(series, exog, fmt):
    """
    Same data as `make_wide` in another input format. With `dict` and `long`
    every series has its own exog (a copy of the shared one, so results are the
    same as with the wide format).
    """
    names = list(series.columns)
    if fmt == "wide":
        return series, exog
    if fmt == "dict":
        s = {k: series[k].copy() for k in names}
        e = None if exog is None else {k: exog.copy() for k in names}
        return s, e
    if fmt == "long":
        s = long_from_dict({k: series[k].to_frame("value") for k in names})
        e = None if exog is None else long_from_dict({k: exog for k in names})
        return s, e
    raise ValueError(fmt)


def long_from_dict(frames):
    out = pd.concat(frames, names=["series_id", "datetime"])
    return out


def exog_for_predict(exog_rows, names, kind):
    """`exog_rows`: wide exog with the rows of the steps to predict."""
    if kind == "none":
        return None
    if kind == "wide":
        return exog_rows
    if kind == "dict":  # the same object for all the series
        return {k: exog_rows for k in names}
    if kind == "dictown":  # one object per series, as with an exog of each series
        return {k: exog_rows.copy() for k in names}
    if kind == "long":
        return long_from_dict({k: exog_rows for k in names})
    raise ValueError(kind)


def make_forecaster(est="linear", encoding="ordinal"):
    if est == "linear":
        estimator = LinearRegression()
    elif est == "lgbm":
        estimator = LGBMRegressor(n_estimators=100, verbose=-1, random_state=1, n_jobs=1)
    else:
        raise ValueError(est)
    return ForecasterRecursiveMultiSeries(estimator, lags=LAGS, encoding=encoding)
