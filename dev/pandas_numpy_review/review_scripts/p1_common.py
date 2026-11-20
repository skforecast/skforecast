"""
Shared helpers of the P1 review scripts (multiseries: DataFrame in `fit`,
numpy in `predict`).

- `make_data`: two series whose value is `level effect + effect of exog c + noise`.
- `spy_fit`: records what `estimator.fit` receives.
- `numpy_fit`: in-memory prototype of the fix proposed in the study (the
  estimator receives `X_train.to_numpy(dtype=float)` in `fit`). Nothing under
  `skforecast/` is modified.
"""
import _env  # noqa: F401
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import skforecast.recursive._forecaster_recursive_multiseries as m
from skforecast.utils import utils as sk_utils

warnings.simplefilter("ignore")

EFFECT3 = np.array([-50.0, 0.0, 50.0])
# Non monotone effect of 8 categories and of 6 series (levels).
EFFECT8 = np.array([30.0, -50.0, 10.0, 50.0, -30.0, 0.0, 40.0, -10.0])
LEVEL6 = np.array([0.0, 60.0, -40.0, 20.0, -80.0, 100.0])


def make_data(categories=(10, 20, 30), n=300, seed=0, n_cat_cols=1, n_series=2, extra=None):
    """
    Series a, b, s2, ... with value `level effect + effect of c + noise`. `c` is a
    `category` exog whose k-th category (in the order of `categories`) adds
    EFFECT3[k] (3 categories) or EFFECT8[k] (8 categories). The level effect is
    5 * j with 2 series and LEVEL6[j] with 6. With `n_cat_cols=2`, `c2` adds
    0.2 * effect. Returns series, exog and exog_pred (one step per category,
    in the order of `categories`; c2 fixed to the second category), so the
    expected prediction of series 'a' is the effect vector.
    """
    rng = np.random.default_rng(seed)
    categories = list(categories)
    k = len(categories)
    effect = EFFECT3 if k == 3 else EFFECT8
    idx = pd.date_range("2020", periods=n, freq="D")
    fidx = pd.date_range(idx[-1] + idx.freq, periods=k, freq="D")
    y = rng.normal(size=n)
    exog = pd.DataFrame({"x": rng.normal(size=n)}, index=idx)
    exog_p = pd.DataFrame({"x": np.zeros(k)}, index=fidx)
    for j in range(n_cat_cols):
        pos = rng.integers(0, k, n)
        y = y + (1.0 if j == 0 else 0.2) * effect[pos]
        name = "c" if j == 0 else f"c{j + 1}"
        exog[name] = pd.Categorical(np.array(categories, dtype=object)[pos], categories=categories)
        exog_p[name] = pd.Categorical(
            categories if j == 0 else [categories[1]] * k, categories=categories
        )
    if extra is not None:
        extra(exog, exog_p, rng)
    level = 5.0 * np.arange(n_series) if n_series == 2 else LEVEL6[:n_series]
    series = {
        ("aBCDEF"[j].lower() if j < 2 else "s" + str(j)): pd.Series(y + level[j] + (rng.normal(size=n) if j else 0.0), index=idx)
        for j in range(n_series)
    }
    return series, exog, exog_p


@contextmanager
def spy_fit(estimator_cls, log):
    """Record type, dtypes and kwargs of every `estimator_cls.fit` call in `log`."""
    orig = estimator_cls.fit

    def fit(self, X, y, **kw):
        info = {"type": type(X).__name__}
        if isinstance(X, pd.DataFrame):
            info["dtypes"] = {c: str(t) for c, t in X.dtypes.items()}
        else:
            info["dtypes"] = str(X.dtype)
        info["kwargs"] = {k: v for k, v in kw.items() if k != "sample_weight"}
        log.append(info)
        return orig(self, X, y, **kw)

    estimator_cls.fit = fit
    try:
        yield
    finally:
        estimator_cls.fit = orig


@contextmanager
def numpy_fit():
    """
    Prototype of the fix: `ForecasterRecursiveMultiSeries.fit` gives the
    estimator `X_train.to_numpy(dtype=float)` (CatBoost: then the numpy cast
    that the single series forecasters use). The hook is the last function
    called on the matrix before `estimator.fit` (as scripts/ms_numpy_fit.py).
    """
    o_cast = m.cast_catboost_categorical_columns_dataframe
    o_copy, o_check = m._copy_rows_to_check, m._check_in_place_fit

    def cast(X, fit_kwargs, estimator, feature_names):
        X = X.to_numpy(dtype=float)
        return sk_utils.cast_catboost_categorical_columns(
            X=X, fit_kwargs=fit_kwargs, estimator=estimator
        )

    m.cast_catboost_categorical_columns_dataframe = cast
    m._copy_rows_to_check = lambda X: None
    m._check_in_place_fit = lambda X, X_rows: None
    try:
        yield
    finally:
        m.cast_catboost_categorical_columns_dataframe = o_cast
        m._copy_rows_to_check, m._check_in_place_fit = o_copy, o_check


def attempt(fn):
    """Return `fn()` or a short text with the exception."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"


def r(x, nd=1):
    """Round an array (or pass a text through) to print it."""
    return x if isinstance(x, str) else np.round(np.asarray(x, dtype=float), nd).tolist()
