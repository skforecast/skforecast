"""
Shared helpers for the `ForecasterRecursiveMultiSeries.fit()` profiling study.

Everything here is study code: it never modifies `skforecast/`. Scripts import the
working tree (repository root is inserted in `sys.path`) and print the module path so
that a stale site-packages install cannot be measured by mistake.

Scenarios (500 series x 2000 daily observations, seed 123):

    A  no exogenous variables
    B  10 numeric exog (float64) per series
    C  5 numeric + 5 categorical exog (category dtype, cardinalities 3, 7, 12, 30, 100)

Run every script with the interpreter of the `skforecast_24_py13` conda environment:

    $env:PYTHONIOENCODING = "utf-8"
    C:\\Users\\Joaquin\\miniconda3\\envs\\skforecast_24_py13\\python.exe dev\\profiling_multiseries_fit\\<script>.py
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402
import sklearn  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402

import skforecast  # noqa: E402
from skforecast.preprocessing import RollingFeatures  # noqa: E402
from skforecast.recursive import ForecasterRecursiveMultiSeries  # noqa: E402

assert Path(skforecast.__file__).resolve().is_relative_to(REPO_ROOT), (
    f"skforecast imported from outside the working tree: {skforecast.__file__}"
)

SCENARIOS = {"A": "none", "B": "numeric", "C": "mixed"}
DEFAULT_N_JOBS = 4


# --------------------------------------------------------------------------- env
def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - git missing
        return "n/a"


def env_info() -> dict:
    cpu = platform.processor()
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            cpu = out
    except Exception:
        pass
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "skforecast": skforecast.__version__,
        "skforecast_file": skforecast.__file__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": lgb.__version__,
        "git_branch": _git("branch", "--show-current"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "cpu": cpu,
        "cpu_count_logical": os.cpu_count(),
        "os": platform.platform(),
    }


def print_env() -> dict:
    info = env_info()
    print("Environment")
    for k, v in info.items():
        print(f"  {k:18s} {v}")
    print()
    return info


# --------------------------------------------------------------------------- data
def make_data(
    n_series: int = 500,
    n_obs: int = 2000,
    exog: str = "none",
    random_state: int = 123,
) -> dict:
    """
    Synthetic multi-series workload. Returns a dict with keys:

    - `series_wide`: DataFrame (n_obs x n_series), DatetimeIndex freq 'D'.
    - `series_dict`: {name: pd.Series} with the same data.
    - `exog_dict`: {name: pd.DataFrame} or None.
    - `exog_long`: long-format DataFrame with MultiIndex (series_id, date), only
      for `exog='numeric'` (long format with category columns is not measured).
    """
    if exog not in {"none", "numeric", "mixed"}:
        raise ValueError(exog)

    rng = np.random.default_rng(random_state)
    idx = pd.date_range("2015-01-01", periods=n_obs, freq="D")
    t = np.arange(n_obs, dtype=float)
    doy = idx.dayofyear.to_numpy(dtype=float)
    dow = idx.dayofweek.to_numpy(dtype=float)

    scale = rng.uniform(1.0, 100.0, n_series)
    level = rng.uniform(0.5, 2.0, n_series)
    trend = rng.uniform(-0.0005, 0.001, n_series)
    amp_y = rng.uniform(0.1, 0.5, n_series)
    amp_w = rng.uniform(0.05, 0.3, n_series)
    phase = rng.uniform(0.0, 2.0 * np.pi, n_series)
    noise = rng.normal(0.0, 0.1, size=(n_obs, n_series))

    values = scale[None, :] * (
        level[None, :]
        + trend[None, :] * t[:, None]
        + amp_y[None, :] * np.sin(2.0 * np.pi * doy[:, None] / 365.25 + phase[None, :])
        + amp_w[None, :] * np.sin(2.0 * np.pi * dow[:, None] / 7.0)
        + noise
    )
    names = [f"series_{i:04d}" for i in range(n_series)]
    series_wide = pd.DataFrame(values, index=idx, columns=names)
    series_dict = {c: series_wide[c].copy() for c in names}

    exog_dict = None
    exog_long = None
    if exog != "none":
        n_num = 10 if exog == "numeric" else 5
        cards = [3, 7, 12, 30, 100] if exog == "mixed" else []
        num_cols = [f"num_{j}" for j in range(n_num)]
        exog_dict = {}
        for c in names:
            df = pd.DataFrame(
                rng.normal(size=(n_obs, n_num)), index=idx, columns=num_cols
            )
            for k in cards:
                codes = rng.integers(0, k, n_obs)
                df[f"cat_{k}"] = pd.Categorical.from_codes(
                    codes, categories=[f"c{k}_{i}" for i in range(k)]
                )
            exog_dict[c] = df
        if exog == "numeric":
            exog_long = pd.concat(exog_dict, names=["series_id", None])

    return {
        "series_wide": series_wide,
        "series_dict": series_dict,
        "exog_dict": exog_dict,
        "exog_long": exog_long,
        "names": names,
    }


def inputs_for(data: dict, series_format: str = "dict", exog_format: str = "dict"):
    """Pick the (series, exog) objects for one input-format combination."""
    series = data["series_dict"] if series_format == "dict" else data["series_wide"]
    if data["exog_dict"] is None:
        exog = None
    elif exog_format == "dict":
        exog = data["exog_dict"]
    elif exog_format == "long":
        if data["exog_long"] is None:
            raise ValueError("long exog only generated for the numeric scenario")
        exog = data["exog_long"]
    else:
        raise ValueError(exog_format)
    return series, exog


# --------------------------------------------------------------------------- forecaster
def build_forecaster(
    n_jobs: int = DEFAULT_N_JOBS,
    n_estimators: int = 25,
    lags: int | list | None = 24,
    window_features: str | None = "default",
    encoding: str | None = "ordinal",
    categorical_features: str | list | None = "auto",
    **kwargs,
) -> ForecasterRecursiveMultiSeries:
    if window_features == "default":
        window_features = RollingFeatures(stats=["mean", "std"], window_sizes=[7, 28])
    return ForecasterRecursiveMultiSeries(
        estimator=LGBMRegressor(
            n_estimators=n_estimators, random_state=123, verbose=-1, n_jobs=n_jobs
        ),
        lags=lags,
        window_features=window_features,
        encoding=encoding,
        categorical_features=categorical_features,
        **kwargs,
    )


def forecaster_defaults(f: ForecasterRecursiveMultiSeries) -> dict:
    """The constructor / fit defaults that stay active in the base configuration."""
    return {
        "encoding": f.encoding,
        "transformer_series": f.transformer_series,
        "transformer_exog": f.transformer_exog,
        "calendar_features": f.calendar_features,
        "categorical_features": f.categorical_features,
        "weight_func": f.weight_func,
        "series_weights": f.series_weights,
        "differentiation": f.differentiation,
        "dropna_from_series": f.dropna_from_series,
        "fit_kwargs": f.fit_kwargs,
        "binner_kwargs": f.binner_kwargs,
        "_probabilistic_mode": f._probabilistic_mode,
        "window_size": f.window_size,
        "lags_are_contiguous": f.lags_are_contiguous,
        "fit(store_last_window)": True,
        "fit(store_in_sample_residuals)": False,
    }


# --------------------------------------------------------------------------- timing
def timeit(fn, reps: int = 5, warmup: int = 1) -> dict:
    """Wall clock with perf_counter. Returns median / min / max / all."""
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return summarize(t)


def summarize(t: list[float]) -> dict:
    return {
        "median": statistics.median(t),
        "min": min(t),
        "max": max(t),
        "n": len(t),
        "all": list(t),
    }


def ab_interleaved(fa, fb, reps: int = 5, warmup: int = 1, label_a="A", label_b="B") -> dict:
    """
    Same-process A/B: warm both, then alternate A, B, A, B, ... (order swapped every
    round) so that CPU boost, allocator state and caches hit both sides equally.
    """
    for _ in range(warmup):
        fa()
        fb()
    ta, tb = [], []
    for i in range(reps):
        order = [(fa, ta), (fb, tb)] if i % 2 == 0 else [(fb, tb), (fa, ta)]
        for fn, bucket in order:
            t0 = time.perf_counter()
            fn()
            bucket.append(time.perf_counter() - t0)
    sa, sb = summarize(ta), summarize(tb)
    return {
        label_a: sa,
        label_b: sb,
        "speedup_median": sa["median"] / sb["median"] if sb["median"] else float("nan"),
    }


def fmt(sec: float) -> str:
    return f"{sec * 1000:8.1f} ms" if sec < 1 else f"{sec:8.3f} s "


# --------------------------------------------------------------------------- identity
def future_exog(exog, steps: int):
    """Exog for the `steps` periods after training: the last `steps` rows of each
    series' exog with the index shifted forward (same dtypes and categories)."""
    if exog is None:
        return None
    if isinstance(exog, dict):
        return {k: future_exog(v, steps) for k, v in exog.items()}
    if isinstance(exog.index, pd.MultiIndex):
        return {k: future_exog(g.droplevel(0), steps)
                for k, g in exog.groupby(level=0, sort=True, observed=True)}
    tail = exog.iloc[-steps:].copy()
    freq = exog.index.freq or pd.infer_freq(exog.index)
    tail.index = pd.date_range(exog.index[-1] + pd.tseries.frequencies.to_offset(freq),
                               periods=steps, freq=freq)
    return tail


def assert_identical_fits(fa: ForecasterRecursiveMultiSeries, fb: ForecasterRecursiveMultiSeries,
                          series, exog, steps: int = 10) -> None:
    """Bit-identical check between two fitted forecasters."""
    Xa, ya = fa.create_train_X_y(series=series, exog=exog)
    Xb, yb = fb.create_train_X_y(series=series, exog=exog)
    pd.testing.assert_frame_equal(Xa, Xb, check_exact=True)
    pd.testing.assert_series_equal(ya, yb, check_exact=True)
    ex = future_exog(exog, steps)
    pa = fa.predict(steps=steps, exog=ex)
    pb = fb.predict(steps=steps, exog=ex)
    pd.testing.assert_frame_equal(pa, pb, check_exact=True)
    assert fa.X_train_series_names_in_ == fb.X_train_series_names_in_
    assert fa.binner_intervals_.keys() == fb.binner_intervals_.keys()
    for k in fa.binner_intervals_:
        assert fa.binner_intervals_[k] == fb.binner_intervals_[k], k
    for k in fa.in_sample_residuals_:
        ra, rb = fa.in_sample_residuals_[k], fb.in_sample_residuals_[k]
        if ra is None or rb is None:
            assert ra is rb, k
        else:
            np.testing.assert_array_equal(ra, rb)
    if fa.in_sample_residuals_by_bin_:
        for k in fa.in_sample_residuals_by_bin_:
            ba, bb = fa.in_sample_residuals_by_bin_[k], fb.in_sample_residuals_by_bin_[k]
            if ba is None or bb is None:
                assert ba is bb, k
            else:
                assert ba.keys() == bb.keys()
                for b in ba:
                    np.testing.assert_array_equal(ba[b], bb[b])


# --------------------------------------------------------------------------- io
def save_json(obj, name: str) -> Path:
    path = RESULTS_DIR / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def md_table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)
