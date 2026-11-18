"""
Identity battery of the labels of a dict exog: column names that are not text
(int, float, NaN, bool mixed with int or text, None, `numpy.str_`,
`numpy.int64`, timezone-aware dates, tuples in a MultiIndex), in columns
Index of several kinds (default, object, category, `string` with and without
`pyarrow`, `str` with `pyarrow`), with the same names in fit and predict and
with names that are only equal (1 and 1.0, True and 1, `numpy.str_` and `str`,
the same instant in two time zones). It also passes exog whose index is
named or has no frequency. It covers the check of change 4 (`reindex_like`
keeps the labels of the user's object only when both axes are `identical`)
and the column check of `check_predict_input`.

Every case fits a ForecasterRecursiveMultiSeries (plain, with categorical
features, with a StandardScaler and with a ColumnTransformer) and records
`predict` and `create_predict_X` with the exog of each series aligned with
the predictions, and whether the exog passed by the user was modified.
Compare two trees with identity_compare_ms.py.

The set of names printed by some warnings is sorted here: with NaN names its
order depends on memory addresses, not on the code.

Usage: python identity_labels.py <out.pkl>
Env: NO_PYARROW=1 blocks pyarrow (the `pyarrow` Index kinds are then skipped).
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
import re
import warnings

import numpy as np
import pandas as pd
import skforecast
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursiveMultiSeries

SKFORECAST_ROOT = os.path.dirname(skforecast.__file__)
RESULTS = {}

try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ModuleNotFoundError:
    HAS_PYARROW = False

warnings.simplefilter("ignore")
NAMES = ["s0", "s1", "s2"]
T, STEPS = 40, 3
IDX = pd.date_range("2021-01-01", periods=T + STEPS, freq="D")
_rng = np.random.default_rng(5)
SERIES = {
    k: pd.Series(_rng.normal(size=T).cumsum() + 50, index=IDX[:T], name=k)
    for k in NAMES
}
VALUES = _rng.normal(size=(T + STEPS, 3))
CATEGORIES = np.arange(T + STEPS) % 4

ts = pd.Timestamp
LABELS = {
    "str": ["a", "b", "c"],
    "int": [1, 2, 3],
    "float": [1.0, 2.0, 3.0],
    "float_nan": [1.0, np.nan, 3.0],
    "bool_int": [True, 2, 3],
    "obj_int_str": [1, "b", "c"],
    "obj_bool_str": [True, "b", "c"],
    "obj_none": ["a", None, "c"],
    "np_str": [np.str_("a"), np.str_("b"), np.str_("c")],
    "np_int": [np.int64(1), np.int64(2), np.int64(3)],
    "tz_utc": [ts("2020-01-01", tz="UTC"), ts("2020-01-02", tz="UTC"),
               ts("2020-01-03", tz="UTC")],
    "tz_madrid": [ts("2020-01-01 01:00", tz="Europe/Madrid"),
                  ts("2020-01-02 01:00", tz="Europe/Madrid"),
                  ts("2020-01-03 01:00", tz="Europe/Madrid")],
    "tuples": [("x", 1), ("x", 2), ("y", 1)],
}
INDEX_KINDS = {
    "plain": lambda labels: pd.Index(labels),
    "object": lambda labels: pd.Index(labels, dtype=object),
    "category": lambda labels: pd.CategoricalIndex(labels),
    "string_python": lambda labels: pd.Index(labels, dtype="string[python]"),
}
if HAS_PYARROW:
    INDEX_KINDS["string_pyarrow"] = lambda labels: pd.Index(labels, dtype="string[pyarrow]")
    INDEX_KINDS["str_pyarrow"] = lambda labels: pd.Index(
        labels, dtype=pd.StringDtype("pyarrow", na_value=np.nan))
INDEX_KINDS["multi"] = lambda labels: pd.MultiIndex.from_tuples(labels)

# (labels in fit, labels in predict)
PAIRS = [(k, k) for k in LABELS] + [
    ("int", "float"), ("float", "int"), ("int", "bool_int"), ("bool_int", "int"),
    ("obj_int_str", "obj_bool_str"), ("obj_bool_str", "obj_int_str"),
    ("str", "np_str"), ("np_str", "str"), ("int", "np_int"),
    ("tz_utc", "tz_madrid"), ("tz_madrid", "tz_utc"),
    ("str", "obj_none"),
]
ESTIMATORS = ("plain", "cat", "scaler", "coltrans")
# Index of the exog of each series at predict time.
EXOG_INDEX = {
    "aligned": lambda index: index,
    "named": lambda index: index.rename("date"),
    "no_freq": lambda index: pd.DatetimeIndex(list(index)),
}


def make_exog(labels, kind, n_rows, categorical):
    columns = INDEX_KINDS[kind](labels)
    exog = pd.DataFrame(VALUES[:n_rows].copy(), index=IDX[:n_rows])
    if categorical:
        exog[1] = pd.Categorical(CATEGORIES[:n_rows])
    exog.columns = columns
    return exog


def sort_sets(message):
    return re.sub(
        r"\{([^{}]*)\}",
        lambda m: "{" + ", ".join(sorted(m.group(1).split(", "))) + "}",
        message,
    )


def unchanged(exog, before):
    try:
        for k in exog:
            pd.testing.assert_frame_equal(exog[k], before[k])
        return True
    except AssertionError:
        return False


def record(name, fn, exog):
    assert name not in RESULTS, name
    exog_before = None if exog is None else {k: v.copy() for k, v in exog.items()}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out, err = fn(), None
        except Exception as e:  # noqa: BLE001
            out, err = None, (type(e).__name__, sort_sets(str(e)))
    RESULTS[name] = {
        "out": {
            "result": out,
            "exog_unchanged": None if exog is None else unchanged(exog, exog_before),
        },
        "err": err,
        "warn": [
            (x.category.__name__, sort_sets(str(x.message)),
             "skforecast" if x.filename.startswith(SKFORECAST_ROOT) else os.path.basename(x.filename))
            for x in w
        ],
    }
    return err


def make_forecaster(est, exog_fit):
    kwargs = {}
    if est == "cat":
        kwargs["categorical_features"] = "auto"
    if est == "scaler":
        kwargs["transformer_exog"] = StandardScaler()
    if est == "coltrans":
        kwargs["transformer_exog"] = make_column_transformer(
            (StandardScaler(), [exog_fit.columns[0]]), remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    return ForecasterRecursiveMultiSeries(Ridge(), lags=3, **kwargs)


def run_case(fit_lab, fit_kind, pred_lab, pred_kind, est, exog_index):
    key = f"{fit_lab}/{fit_kind}->{pred_lab}/{pred_kind}|idx={exog_index}"
    categorical = est == "cat"
    try:
        exog_fit = make_exog(LABELS[fit_lab], fit_kind, T, categorical)
        exog_full = make_exog(LABELS[pred_lab], pred_kind, T + STEPS, categorical)
    except Exception:  # noqa: BLE001
        return  # the Index kind does not accept these labels
    forecaster = make_forecaster(est, exog_fit)
    fit_exog = {k: exog_fit for k in NAMES}
    err = record(f"fit|{est}|{key}", lambda: forecaster.fit(series=SERIES, exog=fit_exog), fit_exog)
    if err is not None:
        return
    rows = exog_full.iloc[T:]
    rows = rows.set_axis(EXOG_INDEX[exog_index](rows.index))
    exog = {k: rows.copy() for k in NAMES}
    record(f"predict|{est}|{key}", lambda: forecaster.predict(steps=STEPS, exog=exog), exog)
    record(
        f"create_predict_X|{est}|{key}",
        lambda: forecaster.create_predict_X(steps=STEPS, exog=exog),
        exog,
    )


if __name__ == "__main__":
    for fit_lab, pred_lab in PAIRS:
        for fit_kind in INDEX_KINDS:
            for pred_kind in INDEX_KINDS:
                if (fit_kind == "multi") != (pred_kind == "multi"):
                    continue
                if (fit_kind == "multi") != (fit_lab == "tuples"):
                    continue
                for est in ESTIMATORS:
                    run_case(fit_lab, fit_kind, pred_lab, pred_kind, est, "aligned")
    for labels in ("str", "int"):
        for exog_index in ("named", "no_freq"):
            for kind in ("plain", "object", "category"):
                for est in ESTIMATORS:
                    run_case(labels, kind, labels, kind, est, exog_index)

    n_err = sum(r["err"] is not None for r in RESULTS.values())
    n_modified = sum(r["out"]["exog_unchanged"] is False for r in RESULTS.values())
    with open(sys.argv[1], "wb") as fh:
        pickle.dump(RESULTS, fh)
    storage = pd.Series(["a"]).dtype.storage if pd.__version__ >= "3" else "object"
    print(f"pandas {pd.__version__} ({storage}) | {os.path.dirname(SKFORECAST_ROOT)[-24:]} | "
          f"{len(RESULTS)} cases, {n_err} errors, {n_modified} exog modified")
