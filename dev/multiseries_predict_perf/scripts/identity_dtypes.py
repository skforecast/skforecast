"""
Identity battery of the last window with columns that are not float:
category (numbers and text), string, default text (object in pandas 2, str in
pandas 3), object, bool (one column and all), boolean and Int64 with NA, int,
datetime and mixed. Every window goes through `predict` (inputs checked) and
`_create_predict_inputs` (inputs not checked, as in backtesting), with
`window_size` rows and more, and with the levels taken from its columns (the
selection without reordering of change 2) and given as a subset or in another
order (the selection by position). Each case also records whether the window
passed by the user was modified. Compare two trees with identity_compare_ms.py.

Usage: python identity_dtypes.py <out.pkl>
Env: NO_PYARROW=1 blocks pyarrow.
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursiveMultiSeries

SKFORECAST_ROOT = os.path.dirname(skforecast.__file__)
RESULTS = {}

KINDS = [
    "float", "category_num", "category_str", "string_num", "str_default",
    "object_float", "bool", "all_bool", "boolean_na", "Int64_na", "int",
    "mixed_all", "datetime",
]
LEVELS = {"none": None, "subset": ["b", "c"], "reordered": ["d", "c", "b", "a"]}

rng = np.random.default_rng(7)
series = pd.DataFrame(
    rng.normal(size=(60, 4)).cumsum(axis=0) + 50,
    index=pd.date_range("2020-01-01", periods=60, freq="D"),
    columns=["a", "b", "c", "d"],
)


def window(kind, rows):
    """Last `rows` rows of the series, with column `b` (or more) of another dtype."""
    lw = series.iloc[-rows:].copy()
    b = lw["b"]
    if kind == "float":
        pass
    elif kind == "category_num":
        lw["b"] = b.round(1).astype("category")
    elif kind == "category_str":
        lw["b"] = b.round(1).astype(str).astype("category")
    elif kind == "string_num":
        lw["b"] = b.round(1).astype(str).astype("string")
    elif kind == "str_default":
        lw["b"] = b.round(1).astype(str)
    elif kind == "object_float":
        lw["b"] = b.astype(object)
    elif kind == "bool":
        lw["b"] = b > b.mean()
    elif kind == "all_bool":
        lw = lw > lw.mean()
    elif kind == "boolean_na":
        lw["b"] = (b > b.mean()).astype("boolean")
        lw.iloc[0, 1] = pd.NA
    elif kind == "Int64_na":
        lw["b"] = b.round().astype("Int64")
        lw.iloc[0, 1] = pd.NA
    elif kind == "int":
        lw["b"] = b.round().astype("int64")
    elif kind == "mixed_all":
        lw["a"] = lw["a"].round(1).astype("category")
        lw["b"] = b > b.mean()
        lw["c"] = lw["c"].round(1).astype(str).astype("string")
    elif kind == "datetime":
        lw["b"] = pd.date_range("2021-01-01", periods=len(lw), freq="D")
    else:
        raise ValueError(kind)
    return lw


def record(name, fn, lw):
    assert name not in RESULTS, name
    lw_before = lw.copy()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out, err = fn(), None
        except Exception as e:  # noqa: BLE001
            out, err = None, (type(e).__name__, str(e))
    try:
        pd.testing.assert_frame_equal(lw, lw_before)
        window_unchanged = True
    except AssertionError:
        window_unchanged = False
    RESULTS[name] = {
        "out": {"result": out, "window_unchanged": window_unchanged},
        "err": err,
        # The file a warning points to: the caller (this script) or skforecast.
        "warn": [
            (x.category.__name__, str(x.message),
             "skforecast" if x.filename.startswith(SKFORECAST_ROOT) else os.path.basename(x.filename))
            for x in w
        ],
    }


if __name__ == "__main__":
    forecasters = {
        "plain": ForecasterRecursiveMultiSeries(Ridge(), lags=5),
        "transf_diff": ForecasterRecursiveMultiSeries(
            Ridge(), lags=5, transformer_series=StandardScaler(), differentiation=1
        ),
    }
    for forecaster in forecasters.values():
        forecaster.fit(series=series)

    for fname, forecaster in forecasters.items():
        for kind in KINDS:
            for rows in (5, 9):
                for lname, levels in LEVELS.items():
                    key = f"{fname}|{kind}|rows={rows}|levels={lname}"
                    lw = window(kind, rows)
                    record(
                        f"predict|{key}",
                        lambda: forecaster.predict(3, levels=levels, last_window=lw),
                        lw,
                    )
                    lw = window(kind, rows)
                    # The differentiators (last item) are compared by identity.
                    record(
                        f"inputs_nocheck|{key}",
                        lambda: forecaster._create_predict_inputs(
                            3, levels=levels, last_window=lw, check_inputs=False
                        )[:5],
                        lw,
                    )

    n_err = sum(r["err"] is not None for r in RESULTS.values())
    n_modified = sum(not r["out"]["window_unchanged"] for r in RESULTS.values())
    with open(sys.argv[1], "wb") as fh:
        pickle.dump(RESULTS, fh)
    storage = pd.Series(["a"]).dtype.storage if pd.__version__ >= "3" else "object"
    print(f"pandas {pd.__version__} ({storage}) | {os.path.dirname(SKFORECAST_ROOT)[-24:]} | "
          f"{len(RESULTS)} cases, {n_err} errors, {n_modified} windows modified")
