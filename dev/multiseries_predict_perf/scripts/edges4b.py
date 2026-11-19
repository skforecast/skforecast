"""
Change 4b edge cases through the public API (predict, predict_interval and
backtesting) for the tree on PYTHONPATH: predictions, dtypes, labels, warnings
(category, text) and errors, pickled. Compare two trees with edges4b_compare.py.

Every case has an exog whose index is not `identical` to the index of the
predictions (named, without `freq`, or both) unless stated, combined with:
Series instead of DataFrame, missing / extra / reordered columns, categorical
columns, `transformer_exog`, levels without exog, one level, an object shared
by several levels (change 6), time zone index, integer index, more rows than
steps, an exog that starts earlier.

Usage: [NO_PYARROW=1] python edges4b.py <out.pkl>
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
    sys.meta_path.insert(0, _BlockPyarrow)

import itertools
import pickle
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.recursive import ForecasterRecursiveMultiSeries

N_TRAIN, STEPS, LAGS = 60, 3, 4
NAMES = ["s0", "s1", "s2", "s3"]


def make_index(kind, n):
    if kind == "h":
        return pd.date_range("2021-01-01", periods=n, freq="h")
    if kind == "tz":
        return pd.date_range("2021-03-27", periods=n, freq="h", tz="Europe/Madrid")
    if kind == "D_us":
        return pd.date_range("2021-01-01", periods=n, freq="D", unit="us")
    if kind == "int":
        return pd.RangeIndex(n)
    raise ValueError(kind)


def make_data(index_kind, exog_kind, n):
    rng = np.random.default_rng(1)
    idx = make_index(index_kind, n)
    series = pd.DataFrame(
        np.cumsum(rng.normal(size=(n, len(NAMES))), axis=0) + 50, index=idx,
        columns=NAMES,
    )
    exog = pd.DataFrame({"e0": rng.normal(size=n)}, index=idx)
    if exog_kind in ("float3", "mixed"):
        exog["e1"] = rng.normal(size=n)
        exog["e2"] = rng.normal(size=n)
    if exog_kind == "mixed":
        exog["e1"] = (np.arange(n) % 24).astype("int64")
        exog["e2"] = pd.Categorical(np.arange(n) % 7)
    return series, exog


def mod_index(ix, how):
    if how == "same":
        return ix
    if how == "named":
        return ix.rename("date")
    if how == "nofreq":
        return pd.DatetimeIndex(ix, freq=None) if isinstance(ix, pd.DatetimeIndex) \
            else pd.Index(np.asarray(ix), dtype="int64")
    if how == "named_nofreq":
        return mod_index(ix, "nofreq").rename("date")
    raise ValueError(how)


def exog_rows(exog, start, how, extra=0, early=0):
    e = exog.iloc[start - early: start + STEPS + extra]
    return e.set_axis(mod_index(e.index, how))


def shape(e, what):
    if what == "frame":
        return e
    if what == "series":
        return e["e0"]
    if what == "reversed":
        return e.iloc[:, ::-1]
    if what == "missing_col":
        return e.drop(columns=e.columns[-1])
    if what == "extra_col":
        return e.assign(zz=1.0)
    raise ValueError(what)


def capture(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fn()
            err = None
        except Exception as e:  # noqa: BLE001
            out, err = None, f"{type(e).__name__}: {e}"
    warns = [(x.category.__name__, str(x.message)) for x in w]
    return out, err, warns


def forecaster(est, transformer, exog_kind):
    if est == "lgbm":
        estimator = LGBMRegressor(n_estimators=20, verbose=-1, random_state=1, n_jobs=1)
    else:
        estimator = LinearRegression()
    if transformer == "scaler":
        tr = StandardScaler()
    elif transformer == "column":
        tr = make_column_transformer(
            (StandardScaler(), ["e0"]), remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    else:
        tr = None
    return ForecasterRecursiveMultiSeries(
        estimator, lags=LAGS, transformer_exog=tr,
        categorical_features="auto" if exog_kind == "mixed" else None,
    )


res = {}
CONFIGS = []
for index_kind, exog_kind, transformer in itertools.product(
    ("h", "tz", "D_us", "int"), ("one", "float3", "mixed"), ("none", "scaler", "column")
):
    if transformer == "scaler" and exog_kind == "mixed":
        continue
    CONFIGS.append((index_kind, exog_kind, transformer))

for index_kind, exog_kind, transformer in CONFIGS:
    est = "lgbm" if exog_kind == "mixed" else "linear"
    n = N_TRAIN + 2 * STEPS + 12
    series, exog = make_data(index_kind, exog_kind, n)
    cfg = (index_kind, exog_kind, transformer)
    f = forecaster(est, transformer, exog_kind)
    out, err, _ = capture(lambda: f.fit(series=series.iloc[:N_TRAIN],
                                        exog=exog.iloc[:N_TRAIN],
                                        store_in_sample_residuals=True))
    if err:
        res[cfg + ("fit",)] = (None, err, [])
        continue
    lw = series.iloc[N_TRAIN - LAGS: N_TRAIN]
    shapes = ["frame", "reversed", "missing_col", "extra_col"]
    if exog_kind == "one":
        shapes = ["frame", "series", "extra_col"]
    for how in ("same", "named", "nofreq", "named_nofreq"):
        for what in shapes:
            for layout in ("own", "shared", "missing_level", "one_level", "mixed_hows"):
                def build(extra=0, early=0):
                    e = shape(exog_rows(exog, N_TRAIN, how, extra, early), what)
                    if layout == "shared":
                        return {k: e for k in NAMES}
                    d = {k: e.copy() for k in NAMES}
                    if layout == "missing_level":
                        del d["s1"]
                    if layout == "mixed_hows":
                        d["s2"] = shape(exog_rows(exog, N_TRAIN, "same"), what)
                    return d
                levels = ["s2"] if layout == "one_level" else None
                for extra, early in ((0, 0), (5, 0), (0, 1)):
                    key = cfg + (how, what, layout, extra, early)
                    ex = build(extra, early)
                    res[key + ("predict",)] = capture(
                        lambda: f.predict(steps=STEPS, levels=levels, last_window=lw,
                                          exog=ex, suppress_warnings=False))
                if what == "frame" and layout in ("own", "shared"):
                    ex = build()
                    res[cfg + (how, what, layout, "interval")] = capture(
                        lambda: f.predict_interval(
                            steps=STEPS, last_window=lw, exog=ex, method="conformal",
                            interval=[0.1, 0.9],
                            suppress_warnings=False))
    # Backtesting: dict (own objects) and wide (one shared object) exog.
    for how in ("same", "named", "nofreq", "named_nofreq"):
        e = exog.set_axis(mod_index(exog.index, how))
        for fmt in ("dict", "wide"):
            if fmt == "dict":
                s_in = {k: series[k].set_axis(mod_index(series.index, "same")) for k in NAMES}
                e_in = {k: e.copy() for k in NAMES}
            else:
                s_in, e_in = series, e
            cv = TimeSeriesFold(steps=STEPS, initial_train_size=N_TRAIN, refit=False)
            res[cfg + (how, fmt, "backtesting")] = capture(
                lambda: backtesting_forecaster_multiseries(
                    forecaster=forecaster(est, transformer, exog_kind), series=s_in,
                    exog=e_in, cv=cv, metric="mean_absolute_error", n_jobs=1,
                    verbose=False, show_progress=False, suppress_warnings=False))

with open(sys.argv[1], "wb") as fh:
    pickle.dump({"pandas": pd.__version__, "res": res}, fh)
n_err = sum(1 for v in res.values() if v[1])
n_warn = sum(1 for v in res.values() if v[2])
print(pd.__version__, f"{len(res)} cases, {n_err} with error, {n_warn} with warnings")
