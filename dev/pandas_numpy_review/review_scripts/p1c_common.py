"""
Shared helpers of the `p1c_*` scripts: adversarial review of option (c) of
D-NP2 (ForecasterRecursiveMultiSeries, `categorical_features=None`, a
`category` exog: fit the estimator with numpy and pass the column index as
categorical). Nothing under `skforecast/` is modified.

- `option_c(variant)`: in-memory prototype. Variants of the numpy matrix:
    'values': the category VALUES (what `predict` builds today); `predict`
              is not touched.
    'codes' : the pandas codes (position in `dtype.categories`, NaN for
              missing), which is what LightGBM trains on today; `predict`
              maps values to codes (unknown or NaN -> NaN).
  The (sorted) codes of the OrdinalEncoder are `categorical_features='auto'`
  with `p1_common.numpy_fit`, so they need no prototype.
  The index of every `category` column of X_train (the level column of
  `encoding='ordinal_category'` included) is given to the estimator with
  `configure_estimator_categorical_features`, and CatBoost gets the integer
  cast of `cast_catboost_categorical_columns`, as the single series forecasters.
- `make_big`: 30 series, c40 (40 categories), c3 (3 categories), numeric exog.
- `lgbm_structure`: tree structure of a LightGBM booster without the category
  identifiers, to compare models fitted with different labels.
"""
import _env  # noqa: F401
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import skforecast.recursive._forecaster_recursive_multiseries as m
from skforecast.utils import utils as sk_utils

warnings.simplefilter("ignore")


def category_columns(X):
    """Positions of the `category` columns of a DataFrame."""
    return [j for j, t in enumerate(X.dtypes) if isinstance(t, pd.CategoricalDtype)]


def to_matrix(X, variant):
    """
    Float matrix of a DataFrame with `category` columns. Returns the matrix
    and {column position: categories} for the columns encoded as codes.
    """
    arr = np.empty(X.shape, dtype=float)
    maps = {}
    for j in range(X.shape[1]):
        col = X.iloc[:, j]
        if isinstance(col.dtype, pd.CategoricalDtype):
            if variant == "values":
                arr[:, j] = col.to_numpy(dtype=float, na_value=np.nan)
            else:
                codes = col.cat.codes.to_numpy().astype(float)
                codes[codes < 0] = np.nan
                arr[:, j] = codes
                maps[j] = col.cat.categories
        else:
            arr[:, j] = col.to_numpy(dtype=float)
    return arr, maps


def values_to_codes(X, maps):
    """Copy of the float matrix `X` with the values of `maps` columns as codes."""
    X = X.copy()
    for j, cats in maps.items():
        v = X[:, j]
        if pd.api.types.is_integer_dtype(cats.dtype):
            ok = np.isfinite(v) & (v == np.floor(np.where(np.isfinite(v), v, 0)))
            code = np.full(len(v), -1)
            code[ok] = cats.get_indexer(v[ok].astype(np.int64))
        else:
            code = cats.get_indexer(v)
        X[:, j] = np.where(code < 0, np.nan, code)
    return X


@contextmanager
def option_c(variant="codes", log=None):
    """
    Prototype of option (c). Only acts on forecasters with
    `categorical_features=None`; the rest get the plain numpy fit
    (`p1_common.numpy_fit`). `log`, if given, receives what the estimator gets.
    """
    assert variant in ("values", "codes")
    state = {}
    o_cast = m.cast_catboost_categorical_columns_dataframe
    o_copy, o_check = m._copy_rows_to_check, m._check_in_place_fit
    o_build = m._build_predict_function
    o_fit = m.ForecasterRecursiveMultiSeries.fit

    def fit(self, *args, **kwargs):
        state["f"] = self
        self._p1c_maps = {}
        return o_fit(self, *args, **kwargs)

    def cast(X, fit_kwargs, estimator, feature_names):
        f = state["f"]
        if f.categorical_features is not None:
            X_np = X.to_numpy(dtype=float)
        else:
            cat_pos = category_columns(X)
            X_np, maps = to_matrix(X, variant)
            f._p1c_maps = maps
            if cat_pos:
                # NOTE: `fit_kwargs` is the dict later unpacked in `estimator.fit`,
                # so it is updated in place.
                sk_utils.configure_estimator_categorical_features(
                    estimator=estimator,
                    categorical_features_names_in_=[feature_names[j] for j in cat_pos],
                    X_train_features_names_out_=feature_names,
                    fit_kwargs=fit_kwargs,
                )
        if log is not None:
            log.append({"cat_pos": category_columns(X), "kwargs": dict(fit_kwargs)})
        return sk_utils.cast_catboost_categorical_columns(
            X=X_np, fit_kwargs=fit_kwargs, estimator=estimator
        )

    def build(estimator, forecaster=None):
        fn = o_build(estimator=estimator, forecaster=forecaster)
        maps = getattr(forecaster, "_p1c_maps", None)
        if not maps:
            return fn
        return lambda X: fn(values_to_codes(X, maps))

    m.ForecasterRecursiveMultiSeries.fit = fit
    m.cast_catboost_categorical_columns_dataframe = cast
    m._copy_rows_to_check = lambda X: None
    m._check_in_place_fit = lambda X, X_rows: None
    m._build_predict_function = build
    try:
        yield
    finally:
        m.ForecasterRecursiveMultiSeries.fit = o_fit
        m.cast_catboost_categorical_columns_dataframe = o_cast
        m._copy_rows_to_check, m._check_in_place_fit = o_copy, o_check
        m._build_predict_function = o_build


def lgbm_structure(est):
    """
    Structure of the trees of a LightGBM model that does not depend on the
    labels of the categories: split feature, gain, counts and leaf values.
    """
    df = est.booster_.trees_to_dataframe()
    cols = ["tree_index", "node_depth", "split_feature", "split_gain", "decision_type",
            "value", "weight", "count"]
    out = df[cols].copy()
    out["split_feature"] = out["split_feature"].astype(str)
    out["n_cats_left"] = [
        len(str(t).split("||")) if d == "==" else -1
        for t, d in zip(df["threshold"], df["decision_type"])
    ]
    return out


def same_structure(a, b):
    """Compare two `lgbm_structure` tables (feature names are replaced by order)."""
    if a.shape != b.shape:
        return f"DIFFERENT (nodes {len(a)} vs {len(b)})"
    num = ["split_gain", "value", "weight", "count", "n_cats_left", "node_depth", "tree_index"]
    fa, fb = pd.factorize(a["split_feature"])[0], pd.factorize(b["split_feature"])[0]
    if not np.array_equal(fa, fb) or not (a["decision_type"].fillna("") == b["decision_type"].fillna("")).all():
        return "DIFFERENT (split features)"
    d = np.nanmax(np.abs(a[num].to_numpy(dtype=float) - b[num].to_numpy(dtype=float)))
    return "identical" if d == 0 else f"DIFFERENT (max abs diff {d:.3g})"


def make_big(n=400, n_series=30, ordinal_share=0.5, seed=0, unseen_from=None, declare_unseen=True):
    """
    30 series over `n` days. Value of series j at t:
        level_j + 0.5 * y[t-1] part + effect40[c40] + effect3[c3] + 3 * x + noise(sd 1)
    - c40: 40 categories with values 100, 107, ..., 373 (not 0..n-1), declared in
      a shuffled order. Effect = ordinal_share * (ordinal ramp, 60 from first to
      last value) + (1 - ordinal_share) * N(0, 20) per category. It changes every
      day and is different for every series.
    - c3: categories 10, 20, 30 declared as [30, 10, 20], effect -15, 0, 15.
    - x: numeric, shared.
    - `unseen_from`: from that row on, 20% of the rows of c40 take a new
      category (value 380, effect +40), which never appears before. With
      `declare_unseen=False` it is not in the dtype of the rows before.
    Returns series (dict), exog (dict of DataFrames), and the expected value
    without noise (dict) to measure the error against the signal.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020", periods=n, freq="D")
    vals40 = 100 + 7 * np.arange(40)
    ramp = np.linspace(-30, 30, 40)
    eff40 = ordinal_share * ramp + (1 - ordinal_share) * rng.normal(0, 20, 40)
    vals3, eff3 = np.array([10, 20, 30]), np.array([-15.0, 0.0, 15.0])
    decl40 = list(rng.permutation(vals40))
    if unseen_from is not None:
        vals40 = np.append(vals40, 380)
        eff40 = np.append(eff40, 40.0)
        if declare_unseen:
            decl40 = decl40[:20] + [380] + decl40[20:]
    x = rng.normal(size=n)
    levels = rng.normal(0, 30, n_series)
    series, exog, signal = {}, {}, {}
    for j in range(n_series):
        p40 = rng.integers(0, 40, n)
        if unseen_from is not None:
            late = np.arange(n) >= unseen_from
            p40 = np.where(late & (rng.random(n) < 0.2), 40, p40)
        p3 = rng.integers(0, 3, n)
        sig = levels[j] + eff40[p40] + eff3[p3] + 3.0 * x
        name = f"s{j:02d}"
        series[name] = pd.Series(sig + rng.normal(size=n), index=idx, name=name)
        signal[name] = pd.Series(sig, index=idx)
        c40 = pd.Categorical(vals40[p40], categories=decl40 if (unseen_from is None or declare_unseen) else None)
        exog[name] = pd.DataFrame(
            {"x": x, "c40": c40, "c3": pd.Categorical(vals3[p3], categories=[30, 10, 20])}, index=idx
        )
    return series, exog, signal


def attempt(fn):
    """Return `fn()` or a short text with the exception."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).strip().splitlines()[0][:150]}"
