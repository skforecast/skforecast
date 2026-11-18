"""
Synthetic data and forecaster used to validate the preprocessing speedups of
`dev/preprocessing_speedups/PLAN.md`. Same data as the performance study
of November 2026 (sizes S, M, L).
"""

from __future__ import annotations

import os
import platform
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Lengths are drawn uniformly in [len_min, len_max] and all series end at the
# same timestamp, so they start at different dates.
SIZES = {
    "T": dict(n_series=6, len_min=400, len_max=600),  # smoke tests
    "S": dict(n_series=100, len_min=1000, len_max=2000),
    "M": dict(n_series=300, len_min=1000, len_max=5000),
    "L": dict(n_series=1000, len_min=1000, len_max=5000),
}

N_CATS = {"cat_common": 5, "cat_store": 20, "cat_promo": 50}


def make_data(n_series, len_min, len_max, freq="h", seed=0):
    """
    Dict of pandas Series (series) and dict of pandas DataFrames (exog). Exog
    has 8 columns: 2 numeric + 1 categorical common to all series and 3
    numeric + 2 categorical specific of each series. Categorical columns use
    the pandas `category` dtype with the same categories in every series.
    """
    rng = np.random.default_rng(seed)
    lengths = rng.integers(len_min, len_max + 1, size=n_series)
    end = pd.Timestamp("2025-01-01 00:00") if freq == "h" else pd.Timestamp("2025-01-01")
    full_len = int(lengths.max())
    full_index = pd.date_range(end=end, periods=full_len, freq=freq)

    t = np.arange(full_len)
    period = 24 if freq == "h" else 7
    common_num_1 = np.sin(2 * np.pi * t / period) + rng.normal(0, 0.1, full_len)
    common_num_2 = rng.normal(15, 5, full_len).cumsum() / np.sqrt(t + 1)
    cat_common_codes = rng.integers(0, N_CATS["cat_common"], full_len)
    cats = {k: [f"{k}_{i}" for i in range(n)] for k, n in N_CATS.items()}

    series_dict, exog_dict = {}, {}
    for i, n in enumerate(lengths):
        name = f"series_{i:04d}"
        idx = full_index[-n:]
        sl = slice(full_len - n, full_len)
        y = (
            rng.uniform(10, 100)
            + 5 * common_num_1[sl]
            + 0.1 * np.cumsum(rng.normal(0, 0.3, n))
            + rng.normal(0, 1, n)
        )
        series_dict[name] = pd.Series(y, index=idx, name=name)
        store = rng.integers(0, N_CATS["cat_store"])
        exog_dict[name] = pd.DataFrame(
            {
                "num_common_1": common_num_1[sl],
                "num_common_2": common_num_2[sl],
                "num_spec_1": rng.normal(0, 1, n),
                "num_spec_2": rng.uniform(0, 1, n),
                "num_spec_3": rng.normal(50, 10, n).round(1),
                "cat_common": pd.Categorical.from_codes(
                    cat_common_codes[sl], categories=cats["cat_common"]
                ),
                "cat_store": pd.Categorical.from_codes(
                    np.full(n, store), categories=cats["cat_store"]
                ),
                "cat_promo": pd.Categorical.from_codes(
                    rng.integers(0, N_CATS["cat_promo"], n), categories=cats["cat_promo"]
                ),
            },
            index=idx,
        )
    return series_dict, exog_dict


def split_train_future(series_dict, exog_dict, steps=168):
    """Training data and the last `steps` rows of exog as future exog."""
    s_train = {k: v.iloc[:-steps] for k, v in series_dict.items()}
    e_train = {k: v.iloc[:-steps] for k, v in exog_dict.items()}
    e_future = {k: v.iloc[-steps:] for k, v in exog_dict.items()}
    return s_train, e_train, e_future


def make_estimator(name):
    if name == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=100, verbose=-1, random_state=123)
    if name == "hgbr":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=100, random_state=123)
    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge()
    raise ValueError(name)


def make_forecaster(estimator="ridge", encoding="ordinal"):
    """Configuration of the study: lags=48, 3 rolling stats, StandardScaler."""
    from sklearn.preprocessing import StandardScaler

    from skforecast.preprocessing import RollingFeatures
    from skforecast.recursive import ForecasterRecursiveMultiSeries

    return ForecasterRecursiveMultiSeries(
        estimator=make_estimator(estimator),
        lags=48,
        window_features=RollingFeatures(stats=["mean", "std", "min"], window_sizes=24),
        encoding=encoding,
        transformer_series=StandardScaler(),
        categorical_features="auto",
    )


def env_report():
    import numba
    import sklearn

    import skforecast

    lines = [
        f"Python {sys.version.split()[0]} ({platform.platform()})",
        f"numpy {np.__version__}, pandas {pd.__version__}, "
        f"scikit-learn {sklearn.__version__}, numba {numba.__version__}, "
        f"skforecast {skforecast.__version__}",
    ]
    try:
        import lightgbm

        lines.append(f"lightgbm {lightgbm.__version__}")
    except ImportError:
        pass
    try:
        import pyarrow

        lines.append(f"pyarrow {pyarrow.__version__}")
    except ImportError:
        lines.append("pyarrow not installed")
    cpu = "?"
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor() or "?"
    lines.append(f"CPU {cpu}, {os.cpu_count()} logical cores")
    return "\n".join(lines)
