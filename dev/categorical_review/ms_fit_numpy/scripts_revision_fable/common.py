"""Shared data and estimator factories for the independent review scripts."""
import os
import warnings
import numpy as np
import pandas as pd

PD3 = int(pd.__version__.split(".")[0]) >= 3
EFFECT = {0: 0.0, 1: 30.0, 2: -20.0}

DTYPES = [
    "float", "int", "bool", "nullable", "cat_int_codes", "cat_int_vals",
    "cat_int_unordered", "cat_float", "cat_str", "object", "str", "datetime",
    "pyarrow", "mixed", "nullable_nona", "pyarrow_nona",
]
CATFEATS = ["auto", "none", "full", "incomplete"]
ENCODINGS = ["ordinal", "ordinal_category", "onehot", None]
ESTIMATORS = [
    "lgbm", "xgb", "hgb", "ridge", "linreg", "pipe_ridge", "pipe_lgbm",
    "ttr_ridge", "ttr_lgbm", "rf", "catboost",
]


def make_estimator(name):
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import (
        HistGradientBoostingRegressor, RandomForestRegressor
    )
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=15, num_leaves=8, min_child_samples=3,
            min_data_per_group=1, cat_smooth=1, cat_l2=0, verbose=-1,
            n_jobs=1, random_state=1, deterministic=True, force_row_wise=True
        )
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=15, max_depth=3, n_jobs=1, random_state=1,
            tree_method="hist"
        )
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=15, depth=3, verbose=0, random_seed=1, thread_count=1,
            allow_writing_files=False
        )
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=15, min_samples_leaf=3, random_state=1
        )
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "linreg":
        return LinearRegression()
    if name == "pipe_ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "pipe_lgbm":
        return make_pipeline(StandardScaler(), make_estimator("lgbm"))
    if name == "ttr_ridge":
        return TransformedTargetRegressor(
            regressor=Ridge(alpha=1.0), transformer=StandardScaler()
        )
    if name == "ttr_lgbm":
        return TransformedTargetRegressor(
            regressor=make_estimator("lgbm"), transformer=StandardScaler()
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=5, max_depth=4, random_state=1, n_jobs=1
        )
    raise ValueError(name)


def _cat_codes(n, rng):
    return rng.integers(0, 3, size=n)


def make_exog_cols(dtype_name, codes, x_f, index):
    """Exog DataFrame for a dtype scenario. `codes` drives the target."""
    n = len(codes)
    df = pd.DataFrame({"x_f": x_f}, index=index)
    cat_like = []      # columns 'auto'/'full' should treat as categorical
    cat_dtype_cols = []  # columns of dtype category
    if dtype_name == "float":
        df["x_c"] = codes.astype(float)
    elif dtype_name == "int":
        df["x_c"] = codes.astype("int64")
    elif dtype_name == "bool":
        df["x_c"] = codes > 0
    elif dtype_name in ("nullable", "nullable_nona"):
        na = dtype_name == "nullable"
        v = pd.array(codes, dtype="Int64")
        if na:
            v[::17] = pd.NA
        df["x_c"] = v
        f = pd.array(x_f * 2, dtype="Float64")
        if na:
            f[5::19] = pd.NA
        df["x_nf"] = f
        b = pd.array(codes > 1, dtype="boolean")
        if na:
            b[3::23] = pd.NA
        df["x_nb"] = b
    elif dtype_name == "cat_int_codes":
        df["x_c"] = pd.Categorical(codes, categories=[0, 1, 2])
        cat_like = ["x_c"]
        cat_dtype_cols = ["x_c"]
    elif dtype_name == "cat_int_vals":
        df["x_c"] = pd.Categorical(codes * 10 + 10, categories=[10, 20, 30])
        cat_like = ["x_c"]
        cat_dtype_cols = ["x_c"]
    elif dtype_name == "cat_int_unordered":
        df["x_c"] = pd.Categorical(codes, categories=[2, 0, 1])
        cat_like = ["x_c"]
        cat_dtype_cols = ["x_c"]
    elif dtype_name == "cat_float":
        df["x_c"] = pd.Categorical(codes + 0.5, categories=[0.5, 1.5, 2.5])
        cat_like = ["x_c"]
        cat_dtype_cols = ["x_c"]
    elif dtype_name == "cat_str":
        df["x_c"] = pd.Categorical(
            np.array(["k", "m", "z"])[codes], categories=["z", "k", "m"]
        )
        cat_like = ["x_c"]
        cat_dtype_cols = ["x_c"]
    elif dtype_name == "object":
        df["x_c"] = pd.Series(
            np.array(["k", "m", "z"], dtype=object)[codes], index=index,
            dtype=object
        )
        cat_like = ["x_c"]
    elif dtype_name == "str":
        dt = "str" if PD3 else "string"
        df["x_c"] = pd.Series(
            np.array(["k", "m", "z"], dtype=object)[codes], index=index
        ).astype(dt)
        cat_like = ["x_c"]
    elif dtype_name == "datetime":
        df["x_c"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(codes, "D")
        cat_like = ["x_c"]
    elif dtype_name in ("pyarrow", "pyarrow_nona"):
        import pyarrow as pa
        v = pd.array(codes, dtype=pd.ArrowDtype(pa.int64()))
        if dtype_name == "pyarrow":
            v[::17] = pd.NA
        df["x_c"] = v
        df["x_pf"] = pd.array(x_f * 2, dtype=pd.ArrowDtype(pa.float64()))
        df["x_pb"] = pd.array(codes > 1, dtype=pd.ArrowDtype(pa.bool_()))
    elif dtype_name == "mixed":
        df["x_i"] = (codes * 3).astype("int32")
        df["x_b"] = codes > 0
        df["x_c"] = pd.Categorical(codes * 10 + 10, categories=[10, 20, 30])
        df["x_s"] = pd.Categorical(
            np.array(["k", "m", "z"])[(codes + 1) % 3], categories=["z", "k", "m"]
        )
        cat_like = ["x_c", "x_s"]
        cat_dtype_cols = ["x_c", "x_s"]
    else:
        raise ValueError(dtype_name)
    if cat_like:
        # Extra integer column, used by the 'incomplete' list.
        df["x_k"] = (np.arange(n) % 4).astype("int64")
    return df, cat_like, cat_dtype_cols


def make_data(dtype_name, as_dict=False, n=90, n_future=8, seed=0):
    """Series, exog (train) and exog (future) for a scenario.

    Wide: one exog DataFrame shared by 3 series. Dict: one exog per series,
    series of different lengths and NaN inside one of them.
    """
    rng = np.random.default_rng(seed)
    idx_all = pd.date_range("2021-01-01", periods=n + n_future, freq="D")
    names = ["s_a", "s_b", "s_c"]
    if not as_dict:
        codes = _cat_codes(n + n_future, rng)
        x_f = rng.normal(size=n + n_future)
        exog_all, cat_like, cat_cols = make_exog_cols(
            dtype_name, codes, x_f, idx_all
        )
        series = pd.DataFrame(index=idx_all[:n])
        for i, s in enumerate(names):
            series[s] = (
                10 * (i + 1) + np.vectorize(EFFECT.get)(codes[:n])
                + 2 * x_f[:n] + rng.normal(scale=0.5, size=n)
            )
        return series, exog_all.iloc[:n], exog_all.iloc[n:], cat_like, cat_cols

    series, exog, exog_fut = {}, {}, {}
    for i, s in enumerate(names):
        codes = _cat_codes(n + n_future, rng)
        x_f = rng.normal(size=n + n_future)
        exog_all, cat_like, cat_cols = make_exog_cols(
            dtype_name, codes, x_f, idx_all
        )
        y = pd.Series(
            10 * (i + 1) + np.vectorize(EFFECT.get)(codes[:n])
            + 2 * x_f[:n] + rng.normal(scale=0.5, size=n),
            index=idx_all[:n], name=s
        )
        start = [0, 13, 27][i]
        y = y.iloc[start:].copy()
        if s == "s_b":
            y.iloc[20:24] = np.nan
        series[s] = y
        exog[s] = exog_all.iloc[start:n]
        exog_fut[s] = exog_all.iloc[n:]
    return series, exog, exog_fut, cat_like, cat_cols


def catfeat_value(mode, cat_like, exog_cols):
    if mode == "auto":
        return "auto"
    if mode == "none":
        return None
    if mode == "full":
        return list(cat_like) if cat_like else ["x_c"]
    if mode == "incomplete":
        # A list that leaves the `x_c` column out.
        rest = [c for c in cat_like if c != "x_c"]
        if "x_k" in exog_cols:
            rest.append("x_k")
        return rest
    raise ValueError(mode)


def summarize_exception(e):
    return ("ERR", type(e).__name__, str(e)[:160])


class Capture:
    """Run a callable capturing its result or exception, plus warnings."""

    def __init__(self):
        self.warns = set()

    def __call__(self, fn):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                out = fn()
            except Exception as e:  # noqa: BLE001
                out = summarize_exception(e)
        for x in w:
            self.warns.add((x.category.__name__, str(x.message)[:90]))
        return out


def pin_core(core=12):
    import ctypes
    k = ctypes.windll.kernel32
    h = k.GetCurrentProcess()
    k.SetProcessAffinityMask(h, 1 << core)
    k.SetPriorityClass(h, 0x00000080)  # HIGH_PRIORITY_CLASS
