"""
Synthetic sample data shared across the scikit-learn compatibility checks.

Each generator uses a fixed seed so the checks are deterministic. Exogenous
data is produced with extra future rows so the prediction checks can pass future
exog that starts one step after the training window ends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_series(n: int = 120) -> pd.Series:
    """Single univariate series with a weekly seasonal component."""

    rng = np.random.default_rng(123)
    index = pd.date_range(start="2020-01-01", periods=n, freq="D")
    values = (
        10
        + np.sin(np.arange(n) * 2 * np.pi / 7)
        + rng.normal(scale=0.5, size=n)
    )
    return pd.Series(values, index=index, name="y")


def make_exog(n: int = 120, horizon: int = 5) -> pd.DataFrame:
    """Exogenous variables covering ``n`` training points plus ``horizon`` future steps.

    The extra ``horizon`` rows let the prediction checks pass future exog that
    starts exactly one step after the training window ends.
    """

    total = n + horizon
    rng = np.random.default_rng(456)
    index = pd.date_range(start="2020-01-01", periods=total, freq="D")
    return pd.DataFrame(
        {
            "exog_num": rng.normal(size=total),
            "exog_cat": rng.choice(["a", "b", "c"], size=total),
        },
        index=index,
    )


def make_multi_series(n: int = 120) -> pd.DataFrame:
    """Several series as columns, for multi-series and multivariate forecasters."""

    rng = np.random.default_rng(789)
    index = pd.date_range(start="2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "series_1": rng.normal(loc=10, size=n),
            "series_2": rng.normal(loc=20, size=n),
            "series_3": rng.normal(loc=30, size=n),
        },
        index=index,
    )


def make_class_series(n: int = 120) -> pd.Series:
    """Integer-labelled series for the classification-based forecaster."""

    rng = np.random.default_rng(321)
    index = pd.date_range(start="2020-01-01", periods=n, freq="D")
    return pd.Series(rng.integers(low=1, high=4, size=n), index=index, name="y")
