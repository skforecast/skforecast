"""
Change 3 edge cases (check of the missing values of `last_window` in
`check_predict_input`) through the public API of
ForecasterRecursiveMultiSeries (`predict`, `predict_interval`, backtesting),
ForecasterDirectMultiVariate (also with `lags` in a dict with a series
without lags) and ForecasterRnn (`predict`), for the tree on PYTHONPATH.
Outcome of every call: result (values, index, columns, dtypes) or exception
(type, text), warnings (category, text, file) and whether the user's window
changed, pickled.

Series names: str, int, [0, 1], booleans, numpy.str_, None, NaN, mixed, float
and tuples. Windows: stored and given, columns in another order, a missing
value in each column (also in another order), more rows than the window size
with a missing value in a row that is not used, an extra column (with and
without missing values), a missing column, a duplicated column (with and
without missing values), object, numpy.str_, category, float, boolean and
named column indexes. Levels: None, all, another order, one, a subset, with a
duplicate, as an array and with Python-equal substitutes (bool for int,
numpy.str_ for str). Rnn needs keras (torch backend).

Usage: [NO_PYARROW=1] PYTHONHASHSEED=0 python edges3.py <out.pkl> [--no-rnn]
       python edges3.py --compare <a.pkl> <b.pkl>
PYTHONHASHSEED: some warnings print a set of names, whose order depends on it.
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

os.environ.setdefault("KERAS_BACKEND", "torch")

import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd


def frame_key(obj):
    if isinstance(obj, pd.DataFrame):
        return ("df", obj.to_numpy(dtype=object).tolist() if obj.size else [],
                repr(obj.index), obj.index.dtype.name, repr(obj.columns),
                obj.columns.dtype.name, [str(d) for d in obj.dtypes])
    if isinstance(obj, pd.Series):
        return ("s", obj.to_numpy(dtype=object).tolist(), repr(obj.index), str(obj.dtype))
    return ("other", repr(obj))


def run(fn, inputs=()):
    snapshots = [x.copy(deep=True) for x in inputs]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = ("ok", frame_key(fn()))
        except Exception as e:  # noqa: BLE001
            res = ("err", type(e).__name__, str(e))
    ws = tuple((x.category.__name__, str(x.message), os.path.basename(x.filename))
               for x in w if not issubclass(x.category, DeprecationWarning))
    changed = tuple(not (a.equals(b) and a.columns.equals(b.columns))
                    for a, b in zip(inputs, snapshots))
    return res, ws, changed


NAME_SETS = {
    "str": ["s0", "s1", "s2", "s3"],
    "int": [0, 1, 2, 3],
    "int01": [0, 1],
    "bool": [True, False],
    "npstr": [np.str_("s0"), np.str_("s1"), np.str_("s2")],
    "none": ["s0", None, "s2"],
    "nan": ["s0", np.nan, "s2"],
    "mixed": ["s0", 1, 2.5],
    "float": [0.0, 1.0, 2.5],
    "tuples": [("a", 1), ("b", 2), ("c", 3)],
}
N = 80
IDX = pd.date_range("2020-01-01", periods=N, freq="D")


def make_series(names):
    rng = np.random.default_rng(1)
    values = rng.normal(size=(N, len(names))).cumsum(axis=0) + 50
    if all(isinstance(x, tuple) for x in names):
        columns = pd.MultiIndex.from_tuples(names)
    else:
        columns = pd.Index(names)
    return pd.DataFrame(values, index=IDX, columns=columns)


def window_variants(series, window_size, extra_name):
    lw = series.iloc[-window_size:]
    names = series.columns.to_list()
    out = {"stored": None, "same": lw.copy()}
    # `iloc` to reverse: a boolean column index used with [] selects rows
    out["reversed"] = lw.iloc[:, ::-1].copy()
    for j in range(len(names)):
        x = lw.copy()
        x.iloc[-1, j] = np.nan
        out[f"nan_col{j}"] = x
        out[f"nan_col{j}_reversed"] = x.iloc[:, ::-1].copy()
    x = series.iloc[-(window_size + 5):].copy()
    x.iloc[0, 0] = np.nan
    out["longer_nan_early"] = x
    x = series.iloc[-(window_size + 5):].copy()
    x.iloc[-1, -1] = np.nan
    out["longer_nan_last"] = x
    x = lw.copy()
    x[extra_name] = np.nan
    out["extra_nan_col"] = x
    x = lw.copy()
    x[extra_name] = 1.0
    out["extra_col"] = x
    out["missing_col"] = lw.iloc[:, :-1].copy()
    out["dup_col"] = pd.concat([lw, lw.iloc[:, [0]]], axis=1)
    x = pd.concat([lw, lw.iloc[:, [0]]], axis=1)
    x.iloc[-1, -1] = np.nan
    out["dup_col_nan"] = x
    x = lw.copy()
    x.columns = pd.Index(list(x.columns), dtype=object)
    out["object_cols"] = x
    x = lw.copy()
    x.iloc[-1, 0] = np.nan
    x.columns = pd.Index(list(x.columns), dtype=object)
    out["object_cols_nan"] = x
    if all(isinstance(c, str) for c in names):
        x = lw.copy()
        x.columns = pd.Index([np.str_(c) for c in names], dtype=object)
        x.iloc[-1, -1] = np.nan
        out["npstr_cols_nan"] = x
        x = lw.copy()
        x.columns = pd.CategoricalIndex(names)
        x.iloc[-1, 0] = np.nan
        out["category_cols_nan"] = x
    if all(isinstance(c, (int, np.integer)) and not isinstance(c, bool) for c in names):
        x = lw.copy()
        x.columns = pd.Index([float(c) for c in names])
        x.iloc[-1, 0] = np.nan
        out["float_cols_nan"] = x
        x = lw.copy()
        if len(names) == 2:
            x.columns = pd.Index([bool(c) for c in names], dtype=object)
        x.iloc[-1, 0] = np.nan
        out["bool_cols_nan"] = x
    x = lw.copy()
    x.columns = x.columns.set_names(["cols"] * x.columns.nlevels)
    x.iloc[-1, 0] = np.nan
    out["named_cols_nan"] = x
    return out


def level_variants(names):
    out = {"None": None, "all": list(names), "reversed": list(names)[::-1],
           "first": [names[0]], "dup": [names[0]] + list(names[:-1]),
           "array": np.array(names, dtype=object)}
    if isinstance(names[-1], str):
        out["last_str"] = names[-1]
    if len(names) >= 2:
        out["all_but_last"] = list(names[:-1])
    subs = []
    for x in names:
        if isinstance(x, (bool, np.bool_)):
            subs.append(int(x))
        elif isinstance(x, (int, np.integer)):
            subs.append(bool(x) if x in (0, 1) else float(x))
        elif isinstance(x, str):
            subs.append(np.str_(x))
        else:
            subs.append(x)
    out["substitutes"] = subs
    return out


def extra_name_for(names):
    return "extra" if any(isinstance(c, str) for c in names) else 99


def multiseries_cases(results):
    from sklearn.linear_model import Ridge
    from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    for sname, names in NAME_SETS.items():
        series = make_series(names)
        for encoding in ("ordinal", "onehot", None):
            f = ForecasterRecursiveMultiSeries(Ridge(), lags=3, encoding=encoding)
            key = ("ms", sname, encoding)
            fit = run(lambda: f.fit(series=series) or pd.DataFrame())
            results[key + ("fit",)] = fit
            if fit[0][0] != "ok":
                continue
            for wname, lw in window_variants(series, 3, extra_name_for(names)).items():
                inputs = [lw] if lw is not None else []
                for lname, lv in level_variants(names).items():
                    results[key + ("predict", wname, lname)] = run(
                        lambda: f.predict(steps=2, levels=lv, last_window=lw,
                                          suppress_warnings=False), inputs)
                    if encoding == "ordinal" and lname in ("None", "reversed", "first"):
                        results[key + ("interval", wname, lname)] = run(
                            lambda: f.predict_interval(steps=2, levels=lv, last_window=lw,
                                                       n_boot=5, interval=[0.1, 0.9]), inputs)
            if encoding == "ordinal":
                series_nan = series.copy()
                series_nan.iloc[60, 0] = np.nan
                for lname, lv in (("None", None), ("first", [names[0]])):
                    for data_name, data in (("clean", series), ("nan", series_nan)):
                        cv = TimeSeriesFold(steps=2, initial_train_size=60, refit=False)
                        results[key + ("backtesting", data_name, lname)] = run(
                            lambda: backtesting_forecaster_multiseries(
                                forecaster=f, series=data, cv=cv, levels=lv,
                                metric="mean_absolute_error", show_progress=False,
                                suppress_warnings=False)[1])


def multivariate_cases(results):
    from sklearn.linear_model import Ridge
    from skforecast.direct import ForecasterDirectMultiVariate
    from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
    for sname, names in NAME_SETS.items():
        series = make_series(names)
        for lags_kind in ("int", "dict_none"):
            lags = 3
            if lags_kind == "dict_none":
                lags = {n: 3 for n in names}
                lags[names[-1]] = None
            key = ("mv", sname, lags_kind)
            try:
                f = ForecasterDirectMultiVariate(Ridge(), level=names[0], lags=lags, steps=2)
            except Exception as e:  # noqa: BLE001
                results[key + ("init",)] = (("err", type(e).__name__, str(e)), (), ())
                continue
            fit = run(lambda: f.fit(series=series) or pd.DataFrame())
            results[key + ("fit",)] = fit
            if fit[0][0] != "ok":
                continue
            for wname, lw in window_variants(series, 3, extra_name_for(names)).items():
                inputs = [lw] if lw is not None else []
                results[key + ("predict", wname)] = run(
                    lambda: f.predict(steps=2, last_window=lw, suppress_warnings=False), inputs)
                if lags_kind == "int":
                    results[key + ("interval", wname)] = run(
                        lambda: f.predict_interval(steps=2, last_window=lw, n_boot=5,
                                                   interval=[0.1, 0.9]), inputs)
            series_nan = series.copy()
            series_nan.iloc[60, -1] = np.nan
            for data_name, data in (("clean", series), ("nan", series_nan)):
                cv = TimeSeriesFold(steps=2, initial_train_size=60, refit=False)
                results[key + ("backtesting", data_name)] = run(
                    lambda: backtesting_forecaster_multiseries(
                        forecaster=f, series=data, cv=cv, metric="mean_absolute_error",
                        show_progress=False, suppress_warnings=False)[1])


def rnn_cases(results):
    import keras
    from skforecast.deep_learning import ForecasterRnn
    from skforecast.deep_learning.utils import create_and_compile_model
    for sname in ("str", "int", "npstr", "nan", "mixed"):
        names = NAME_SETS[sname]
        series = make_series(names)
        for levels_kind in ("all", "first"):
            levels = list(names) if levels_kind == "all" else [names[0]]
            key = ("rnn", sname, levels_kind)
            keras.utils.set_random_seed(123)
            try:
                model = create_and_compile_model(series=series, levels=levels, lags=3,
                                                 steps=2, recurrent_units=4, dense_units=4)
                f = ForecasterRnn(estimator=model, levels=levels, lags=3,
                                  fit_kwargs={"epochs": 1, "verbose": 0})
            except Exception as e:  # noqa: BLE001
                results[key + ("init",)] = (("err", type(e).__name__, str(e)), (), ())
                continue
            fit = run(lambda: f.fit(series=series) or pd.DataFrame())
            results[key + ("fit",)] = fit
            if fit[0][0] != "ok":
                continue
            for wname, lw in window_variants(series, 3, extra_name_for(names)).items():
                inputs = [lw] if lw is not None else []
                for lname, lv in (("None", None), ("first", [names[0]]),
                                  ("reversed", list(levels)[::-1])):
                    results[key + ("predict", wname, lname)] = run(
                        lambda: f.predict(steps=2, levels=lv, last_window=lw,
                                          suppress_warnings=False), inputs)


def compare(a_path, b_path):
    with open(a_path, "rb") as fh:
        a = pickle.load(fh)
    with open(b_path, "rb") as fh:
        b = pickle.load(fh)
    keys = sorted(set(a) | set(b), key=repr)
    n_diff = 0
    for k in keys:
        # repr: NaN values compare equal, -0.0 and 0.0 do not
        if repr(a.get(k)) != repr(b.get(k)):
            n_diff += 1
            if n_diff <= 20:
                print("DIFF", k)
                print("   a:", repr(a.get(k))[:400])
                print("   b:", repr(b.get(k))[:400])
    print(f"{len(keys)} cases, {n_diff} differences; by forecaster: "
          f"{dict(Counter(k[0] for k in keys))}")
    print("outcomes a:", dict(Counter(v[0][0] for v in a.values())),
          "with warnings:", sum(1 for v in a.values() if v[1]),
          "window changed:", sum(1 for v in a.values() if any(v[2])))


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
        sys.exit()
    import skforecast
    print(skforecast.__file__, pd.__version__)
    results = {}
    multiseries_cases(results)
    multivariate_cases(results)
    if "--no-rnn" not in sys.argv:
        rnn_cases(results)
    with open(sys.argv[1], "wb") as fh:
        pickle.dump(results, fh)
    print(len(results), "cases")
