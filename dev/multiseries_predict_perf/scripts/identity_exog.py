"""
Identity battery of exog with columns that are not float: int, nullable Int64
and Float64 (with and without NA), pyarrow dtypes, bool and boolean with NA,
category with integer and text categories, text, object, datetime, several
dtypes at once, and a dict whose series give the predicted rows with
different dtypes. With NA in the predicted rows every case raises (pandas NA
cannot be converted to float): those kinds cover the errors.

Every kind of exog is fitted with Ridge and with LightGBM and goes through:
- `predict` with the exog wide, as a dict that repeats one object (shared),
  with one object per series (own), with more rows than steps (extra), with
  the columns in another order (reordered), without the exog of some series
  (partial) and in long format; with the window given and stored in fit.
- `backtesting_forecaster_multiseries` with series and exog wide, as dicts and
  in long format, without refit, and wide with refit.
Each case also records whether the exog passed by the user was modified.
Compare two trees with identity_compare_ms.py.

Usage: python identity_exog.py <out.pkl>
Env: NO_PYARROW=1 blocks pyarrow (the pyarrow dtypes are then skipped).
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

import pickle
import warnings

import numpy as np
import pandas as pd
import skforecast
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.recursive import ForecasterRecursiveMultiSeries

SKFORECAST_ROOT = os.path.dirname(skforecast.__file__)
RESULTS = {}

try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ModuleNotFoundError:
    HAS_PYARROW = False

N_SERIES, T, STEPS, LAGS = 4, 40, 3, 3
NAMES = [f"s{i}" for i in range(N_SERIES)]
IDX = pd.date_range("2021-01-01", periods=T + 12, freq="D")
SERIES = pd.DataFrame(
    np.random.default_rng(3).normal(size=(len(IDX), N_SERIES)).cumsum(axis=0) + 50,
    index=IDX, columns=NAMES,
)

KINDS = [
    "float", "int", "Int64", "Int64_na", "Float64", "Float64_na", "int_pyarrow",
    "float_pyarrow", "bool", "boolean", "boolean_na", "category_int", "category_str",
    "str_default", "object", "datetime", "mixed",
]
PREDICT_FORMATS = [
    "wide", "wide_stored", "dict_shared", "dict_own", "dict_extra", "dict_reordered",
    "dict_partial", "long",
]
FORECASTERS = {
    "ridge": lambda: ForecasterRecursiveMultiSeries(Ridge(), lags=LAGS),
    "lgbm": lambda: ForecasterRecursiveMultiSeries(
        LGBMRegressor(n_estimators=10, max_depth=2, verbose=-1, random_state=1, n_jobs=1),
        lags=LAGS,
    ),
}


def extra_column(kind, n, seed):
    """Column `x` of the kind; NA (in the `_na` kinds) in the first predicted row."""
    rng = np.random.default_rng(seed)
    ints = rng.integers(0, 4, size=n)
    if kind == "float":
        col = pd.Series(ints + rng.normal(size=n))
    elif kind == "int":
        col = pd.Series(ints.astype("int64"))
    elif kind == "Int64":
        col = pd.Series(ints).astype("Int64")
    elif kind == "Float64":
        col = pd.Series(ints + rng.normal(size=n)).astype("Float64")
    elif kind == "boolean":
        col = pd.Series(ints > 1).astype("boolean")
    elif kind == "Int64_na":
        col = pd.Series(ints).astype("Int64")
        col.iloc[T] = pd.NA
    elif kind == "Float64_na":
        col = pd.Series(ints + rng.normal(size=n)).astype("Float64")
        col.iloc[T] = pd.NA
    elif kind == "int_pyarrow":
        col = pd.Series(ints).astype("int64[pyarrow]")
    elif kind == "float_pyarrow":
        col = pd.Series(ints + rng.normal(size=n)).astype("float64[pyarrow]")
    elif kind == "bool":
        col = pd.Series(ints > 1)
    elif kind == "boolean_na":
        col = pd.Series(ints > 1).astype("boolean")
        col.iloc[T] = pd.NA
    elif kind == "category_int":
        col = pd.Series(pd.Categorical(ints))
    elif kind == "category_str":
        col = pd.Series(pd.Categorical(np.array(["a", "b", "c", "d"])[ints]))
    elif kind == "str_default":
        col = pd.Series(np.array(["a", "b", "c", "d"])[ints]).astype(str)
    elif kind == "object":
        col = pd.Series(ints.astype(float)).astype(object)
    elif kind == "datetime":
        col = pd.Series(pd.date_range("2000-01-01", periods=n, freq="D"))
    else:
        raise ValueError(kind)
    return col.to_numpy() if kind in ("int", "bool") else col.array


def exog_of(kind, seed=11):
    """Wide exog: a float column and one or more columns of the kind."""
    n = len(IDX)
    rng = np.random.default_rng(seed)
    exog = pd.DataFrame({"ef": rng.normal(size=n)}, index=IDX)
    if kind == "mixed":
        exog["xi"] = extra_column("int", n, seed + 1)
        exog["xc"] = extra_column("category_int", n, seed + 2)
        exog["xb"] = extra_column("bool", n, seed + 3)
        exog["xn"] = extra_column("Int64_na", n, seed + 4)
    else:
        exog["x"] = extra_column(kind, n, seed + 1)
    return exog


def exog_per_series_dtypes():
    """
    Dict of exog with an `int` column `x`, whose predicted rows have another
    dtype in each series (fit rejects different dtypes between series, predict
    combines them).
    """
    exog = {name: exog_of("int", seed=11 + i) for i, name in enumerate(NAMES)}
    for name, dtype in zip(NAMES, ["int64", "float64", "Int64", "category"]):
        exog[name] = exog[name].astype({"x": dtype})
    return exog


def long_of(frames):
    return pd.concat(frames, names=["series_id", "datetime"])


def predict_exog(exog, fmt):
    """Exog of the predicted steps (rows T to T + STEPS) in the format `fmt`."""
    if isinstance(exog, dict):
        rows = {k: v.iloc[T:T + STEPS].copy() for k, v in exog.items()}
        if fmt == "dict_own":
            return rows
        if fmt == "long":
            return long_of(rows)
        raise ValueError(fmt)
    rows = exog.iloc[T:T + STEPS].copy()
    if fmt in ("wide", "wide_stored"):
        return rows
    if fmt == "dict_shared":
        return {k: rows for k in NAMES}
    if fmt == "dict_own":
        return {k: rows.copy() for k in NAMES}
    if fmt == "dict_extra":
        return {k: exog.iloc[T:T + STEPS + 4].copy() for k in NAMES}
    if fmt == "dict_reordered":
        return {k: rows[rows.columns[::-1]].copy() for k in NAMES}
    if fmt == "dict_partial":
        return {k: rows.copy() for k in NAMES[1:]}
    if fmt == "long":
        return long_of({k: rows for k in NAMES})
    raise ValueError(fmt)


def copy_of(obj):
    if isinstance(obj, dict):
        return {k: v.copy() for k, v in obj.items()}
    return obj.copy()


def unchanged(obj, before):
    try:
        if isinstance(obj, dict):
            assert list(obj) == list(before)
            for k in obj:
                pd.testing.assert_frame_equal(obj[k], before[k])
        else:
            pd.testing.assert_frame_equal(obj, before)
        return True
    except AssertionError:
        return False


def record(name, fn, exog):
    assert name not in RESULTS, name
    exog_before = None if exog is None else copy_of(exog)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out, err = fn(), None
        except Exception as e:  # noqa: BLE001
            out, err = None, (type(e).__name__, str(e))
    RESULTS[name] = {
        "out": {
            "result": out,
            "exog_unchanged": None if exog is None else unchanged(exog, exog_before),
        },
        "err": err,
        # The file a warning points to: the caller (this script) or skforecast.
        "warn": [
            (x.category.__name__, str(x.message),
             "skforecast" if x.filename.startswith(SKFORECAST_ROOT) else os.path.basename(x.filename))
            for x in w
        ],
    }
    return err


def series_as(fmt):
    if fmt == "wide":
        return SERIES
    if fmt == "dict":
        return {k: SERIES[k].copy() for k in NAMES}
    if fmt == "long":
        return long_of({k: SERIES[k].to_frame("value") for k in NAMES})
    raise ValueError(fmt)


def exog_as(exog, fmt):
    if fmt == "wide":
        return exog
    if fmt == "dict":
        return {k: exog.copy() for k in NAMES}
    if fmt == "long":
        return long_of({k: exog for k in NAMES})
    raise ValueError(fmt)


def run_kind(kind, exog):
    per_series = isinstance(exog, dict)
    fit_exog = (
        {k: v.iloc[:T].astype({"x": "int64"}) for k, v in exog.items()}
        if per_series else exog.iloc[:T]
    )
    for fname, make in FORECASTERS.items():
        key = f"{fname}|{kind}"
        forecaster = make()

        def fit():
            forecaster.fit(series=SERIES.iloc[:T], exog=fit_exog)
            return forecaster.exog_dtypes_in_, forecaster.exog_dtypes_out_

        err = record(f"fit|{key}", fit, fit_exog)
        if err is not None:
            continue
        for fmt in (["dict_own", "long"] if per_series else PREDICT_FORMATS):
            ex = predict_exog(exog, fmt)
            last_window = None if fmt == "wide_stored" else SERIES.iloc[T - LAGS:T]
            record(
                f"predict|{key}|exog={fmt}",
                lambda: forecaster.predict(steps=STEPS, last_window=last_window, exog=ex),
                ex,
            )
        # The per series dtypes are only in the predicted rows: no backtesting.
        configs = [] if per_series else [
            ("wide", False), ("dict", False), ("long", False), ("wide", True),
        ]
        for fmt, refit in configs:
            series_bt, exog_bt = series_as(fmt), exog_as(exog, fmt)
            cv = TimeSeriesFold(steps=2, initial_train_size=T, refit=refit)
            record(
                f"backtesting|{key}|fmt={fmt}|refit={refit}",
                lambda: backtesting_forecaster_multiseries(
                    forecaster=make(), series=series_bt, exog=exog_bt, cv=cv,
                    metric="mean_absolute_error", n_jobs=1, verbose=False,
                    show_progress=False,
                ),
                exog_bt,
            )


if __name__ == "__main__":
    for kind in KINDS:
        if kind.endswith("_pyarrow") and not HAS_PYARROW:
            continue
        run_kind(kind, exog_of(kind))
    run_kind("dtypes_per_series", exog_per_series_dtypes())

    n_err = sum(r["err"] is not None for r in RESULTS.values())
    n_modified = sum(r["out"]["exog_unchanged"] is False for r in RESULTS.values())
    with open(sys.argv[1], "wb") as fh:
        pickle.dump(RESULTS, fh)
    storage = pd.Series(["a"]).dtype.storage if pd.__version__ >= "3" else "object"
    print(f"pandas {pd.__version__} ({storage}) | {os.path.dirname(SKFORECAST_ROOT)[-24:]} | "
          f"{len(RESULTS)} cases, {n_err} errors, {n_modified} exog modified")
