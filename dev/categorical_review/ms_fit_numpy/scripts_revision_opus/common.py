import os
import sys
import warnings
import numpy as np
import pandas as pd


def banner():
    import skforecast
    print("skforecast:", skforecast.__file__, "pandas", pd.__version__, flush=True)


def make_series(kind="dict", n=120, n_series=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    data = {}
    for k in range(n_series):
        v = 10 + k + np.sin(np.arange(n) / (3 + k)) * 3 + rng.normal(0, 0.5, n)
        s = pd.Series(v, index=idx, name=f"s{k}")
        data[f"s{k}"] = s
    if kind == "df":
        return pd.DataFrame(data)
    # dict with different lengths and NaN
    data["s1"] = data["s1"].iloc[15:]
    data["s2"] = data["s2"].copy()
    data["s2"].iloc[40:44] = np.nan
    data["s3"] = data["s3"].iloc[:-0 or None].iloc[5:]
    return data


def make_exog_frame(idx, cols, seed=1):
    rng = np.random.default_rng(seed)
    n = len(idx)
    out = {}
    cycle = np.arange(n) % 3
    for c in cols:
        if c == "f":
            out[c] = rng.normal(0, 1, n)
        elif c == "i":
            out[c] = rng.integers(0, 5, n).astype("int64")
        elif c == "b":
            out[c] = (cycle == 1)
        elif c == "ci":   # category of ints, values != codes, effect big
            out[c] = pd.Categorical(np.array([10, 20, 30])[cycle])
        elif c == "ci0":  # category of ints, codes == values
            out[c] = pd.Categorical(np.array([0, 1, 2])[cycle])
        elif c == "cneg":  # category of ints with negatives
            out[c] = pd.Categorical(np.array([-5, 0, 7])[cycle])
        elif c == "cf":   # category of floats
            out[c] = pd.Categorical(np.array([1.5, 2.5, 3.5])[cycle])
        elif c == "cs":
            out[c] = pd.Categorical(np.array(["a", "b", "c"])[cycle])
        elif c == "o":
            out[c] = np.array(["x", "y", "z"], dtype=object)[cycle]
        elif c == "str":
            out[c] = pd.array(np.array(["x", "y", "z"])[cycle], dtype="string")
        elif c == "nI":
            a = pd.array(rng.integers(0, 5, n), dtype="Int64")
            a[7] = pd.NA
            out[c] = a
        elif c == "nIfull":
            out[c] = pd.array(rng.integers(0, 5, n), dtype="Int64")
        elif c == "nF":
            out[c] = pd.array(rng.normal(0, 1, n), dtype="Float64")
        elif c == "nB":
            out[c] = pd.array(cycle == 2, dtype="boolean")
        elif c == "paI":
            out[c] = pd.array(rng.integers(0, 5, n), dtype="int64[pyarrow]")
        elif c == "paF":
            out[c] = pd.array(rng.normal(0, 1, n), dtype="double[pyarrow]")
        elif c == "paB":
            out[c] = pd.array(cycle == 2, dtype="bool[pyarrow]")
        elif c == "dt":
            out[c] = pd.date_range("2000-01-01", periods=n, freq="D")
        else:
            raise ValueError(c)
    return pd.DataFrame(out, index=idx)


def series_effect(series, exog, col):
    """Add a strong effect of categorical column `col` to the series."""
    if col is None:
        return series
    if isinstance(series, dict):
        out = {}
        for k, s in series.items():
            e = exog[k] if isinstance(exog, dict) else exog
            vals = e.loc[s.index, col].astype(float).to_numpy()
            out[k] = s + vals / 5
        return out
    e = exog
    vals = e.loc[series.index, col].astype(float).to_numpy()
    return series.add(vals[:, None] / 5)


def make_case(series_kind, cols, steps=5, effect_col=None, exog_kind="dict"):
    series = make_series(series_kind)
    n_full = 120 + steps
    idx_full = pd.date_range("2020-01-01", periods=n_full, freq="D")
    base = make_exog_frame(idx_full, cols)
    if exog_kind == "dict" and series_kind == "dict":
        exog_full = {}
        for k, s in series.items():
            exog_full[k] = base.loc[s.index[0]:]
    else:
        exog_full = base
    if effect_col is not None:
        series = series_effect(series, exog_full, effect_col)
    end = idx_full[119]
    if isinstance(exog_full, dict):
        exog_train = {k: v.loc[:end] for k, v in exog_full.items()}
        exog_pred = {k: v.loc[end:].iloc[1:] for k, v in exog_full.items()}
    else:
        exog_train = exog_full.loc[:end]
        exog_pred = exog_full.loc[end:].iloc[1:]
    return series, exog_train, exog_pred


def make_estimator(name):
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    if name == "lgbm":
        return LGBMRegressor(n_estimators=20, verbose=-1, random_state=1, n_jobs=1)
    if name == "xgb":
        return XGBRegressor(n_estimators=20, random_state=1, n_jobs=1)
    if name == "cat":
        return CatBoostRegressor(iterations=20, verbose=0, random_seed=1,
                                 thread_count=1, allow_writing_files=False)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=20, random_state=1)
    if name == "ridge":
        return Ridge()
    if name == "lr":
        return LinearRegression()
    if name == "rf":
        return RandomForestRegressor(n_estimators=5, random_state=1, n_jobs=1)
    if name == "pipe_ridge":
        return make_pipeline(StandardScaler(), Ridge())
    if name == "pipe_lgbm":
        return make_pipeline(StandardScaler(), LGBMRegressor(n_estimators=20, verbose=-1, random_state=1, n_jobs=1))
    if name == "ttr_lgbm":
        return TransformedTargetRegressor(
            regressor=LGBMRegressor(n_estimators=20, verbose=-1, random_state=1, n_jobs=1),
            func=np.log1p, inverse_func=np.expm1)
    raise ValueError(name)


def run(fn):
    """Run fn, return ('ok', result, warnings) or ('err', 'Type: msg', warnings)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = fn()
            status = "ok"
        except Exception as e:
            res = f"{type(e).__name__}: {str(e)[:300]}"
            status = "err"
    wl = sorted({f"{x.category.__name__}" for x in w})
    return status, res, wl
