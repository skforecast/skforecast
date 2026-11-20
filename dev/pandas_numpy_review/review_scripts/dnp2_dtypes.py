"""
D-NP2: exog dtypes other than float64 (integers, float32, bool, nullable,
pyarrow, `category` with and without NaN, ordered `category`), with
`categorical_features` 'auto' and None and four estimators: what `fit` and
`predict` of `ForecasterRecursiveMultiSeries` do with each tree.

Run it with each tree and compare with `dnp2_cmp.py dtypes OUTDIR`:
    SK_TREE=<tree> bash run3.sh dnp2_dtypes.py OUTDIR
"""
import _env  # noqa: F401  (first: it can block pyarrow)
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
n = 200
idx = pd.date_range("2020", periods=n, freq="D")
fidx = pd.date_range(idx[-1] + idx.freq, periods=3, freq="D")
series = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n) + 1}, index=idx)
base = rng.integers(0, 4, n + 3)
x_values = rng.normal(size=n + 3)


def with_nan(values, dtype=None):
    values = pd.Series(values, dtype=dtype)
    values.iloc[5] = np.nan if dtype is None else pd.NA
    return values


COLUMNS = {
    "int64": lambda v: v.astype("int64"),
    "int32": lambda v: v.astype("int32"),
    "float32": lambda v: (v / 3).astype("float32"),
    "bool": lambda v: v > 1,
    "Int64": lambda v: pd.array(v, dtype="Int64"),
    "Int64 with NA": lambda v: with_nan(v, "Int64").array,
    "Float64": lambda v: pd.array(v / 3, dtype="Float64"),
    "boolean": lambda v: pd.array(v > 1, dtype="boolean"),
    "int64[pyarrow]": lambda v: pd.array(v, dtype="int64[pyarrow]"),
    "double[pyarrow]": lambda v: pd.array(v / 3, dtype="double[pyarrow]"),
    "bool[pyarrow]": lambda v: pd.array(v > 1, dtype="bool[pyarrow]"),
    "category int": lambda v: pd.Categorical(v),
    "category int with NaN": lambda v: pd.Categorical(with_nan(v.astype(float)).array, categories=[0, 1, 2, 3]),
    "category ordered": lambda v: pd.Categorical(v, ordered=True),
}


def col(kind):
    return COLUMNS[kind](base)


KINDS = list(COLUMNS)
EST = {"lgbm": lambda: LGBMRegressor(verbose=-1, n_estimators=20, min_child_samples=5, random_state=1, n_jobs=1),
       "hgb": lambda: HistGradientBoostingRegressor(max_iter=20, random_state=1),
       "xgb": lambda: XGBRegressor(n_estimators=20, random_state=1, n_jobs=1), "ridge": Ridge}
res = {}
for kind in KINDS:
    try:
        values = col(kind)
    except Exception as e:  # noqa (pyarrow blocked)
        for est in EST:
            for cf in ("auto", None):
                res[(kind, est, cf)] = f"not available: {type(e).__name__}"
        continue
    exog = pd.DataFrame({"x": x_values}, index=idx.append(fidx))
    exog["z"] = values
    for est in EST:
        for cf in ("auto", None):
            try:
                f = ForecasterRecursiveMultiSeries(EST[est](), lags=3, categorical_features=cf)
                f.fit(series=series, exog=exog.loc[idx])
                try:
                    res[(kind, est, cf)] = f.predict(steps=3, exog=exog.loc[fidx])["pred"].to_numpy()
                except Exception as e:  # noqa
                    res[(kind, est, cf)] = f"PREDICT {type(e).__name__}: {str(e).splitlines()[0][:70]}"
            except Exception as e:  # noqa
                res[(kind, est, cf)] = f"FIT {type(e).__name__}: {str(e).splitlines()[0][:70]}"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
path = out / f"dtypes_{_env.tree()}_{_env.tag()}.pkl"
pickle.dump({"versions": _env.versions(), "results": res}, open(path, "wb"))
print(len(res), "cases ->", path, _env.versions())
