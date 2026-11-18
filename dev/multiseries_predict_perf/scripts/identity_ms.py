"""
Identity battery of the multiple series forecasters: outputs, warnings and
errors of fit, predict (and the probabilistic methods) and backtesting of
ForecasterRecursiveMultiSeries and ForecasterDirectMultiVariate for the tree on
PYTHONPATH. Compare two trees with identity_compare_ms.py. It extends
identity_trims.py (3 series of equal length) with more series, series of
different length, unknown levels, the four encodings, input formats (wide,
dict, long), names that are not text and levels given in several ways.

Usage: python identity_ms.py <out.pkl> [regex of case names]
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
import re
import warnings

import numpy as np
import pandas as pd
import skforecast
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import (
    TimeSeriesFold, backtesting_forecaster_multiseries,
)
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries

N = 200
T = 160  # end of the training data in the predict cases
PATTERN = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
SKFORECAST_ROOT = os.path.dirname(skforecast.__file__)
RESULTS = {}


def record(name, fn):
    assert name not in RESULTS, name
    if PATTERN and not PATTERN.search(name):
        return
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out, err = fn(), None
        except Exception as e:  # noqa: BLE001
            out, err = None, (type(e).__name__, str(e))
    RESULTS[name] = {
        "out": out, "err": err,
        # The file a warning points to: the caller (this script) or skforecast.
        "warn": [
            (x.category.__name__, str(x.message),
             "skforecast" if x.filename.startswith(SKFORECAST_ROOT) else os.path.basename(x.filename))
            for x in w
        ],
    }


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def idx_of(kind, n=N):
    if kind == "dt":
        return pd.date_range("2020-01-01", periods=n, freq="h")
    if kind == "tz_D":  # local calendar days, crosses the DST change of October
        return pd.date_range("2019-06-01", periods=n, freq="D", tz="Europe/Madrid")
    if kind == "utc_D":  # UTC-anchored days shown in a DST timezone
        return pd.date_range("2019-06-01", periods=n, freq="D", tz="UTC").tz_convert(
            "Europe/Madrid"
        )
    if kind == "MS":
        return pd.date_range("1990-01-01", periods=n, freq="MS")
    if kind == "range":
        return pd.RangeIndex(n)
    raise ValueError(kind)


def names_of(n_series, kind):
    if kind == "str":
        return [f"series_{i:03d}" for i in range(n_series)]
    if kind == "short":
        return [f"s{i}" for i in range(n_series)]
    if kind == "int":
        return list(range(100, 100 + n_series))
    if kind == "unsorted":  # not in alphabetical order
        return [f"id_{(i * 7) % n_series:03d}" for i in range(n_series)]
    raise ValueError(kind)


def wide_series(n_series, ikind, lengths, names_kind, n=N):
    """
    Wide DataFrame. With `ragged`, series start and end at different dates
    (NaN before the start and after the end): one series in four ends before
    the end of the training data and one in five starts late.
    """
    rng = np.random.default_rng(3)
    idx = idx_of(ikind, n)
    values = np.cumsum(rng.normal(size=(n, n_series)), axis=0) + 50
    if lengths == "ragged":
        for j in range(n_series):
            if j % 5 == 1:
                values[: 10 + 3 * (j % 7), j] = np.nan
            if j % 4 == 2:
                values[T - 5 - (j % 3):, j] = np.nan
            if j % 6 == 5:
                values[n - 12:, j] = np.nan
    return pd.DataFrame(values, index=idx, columns=names_of(n_series, names_kind))


def exog_wide(ekind, idx):
    rng = np.random.default_rng(7)
    n = len(idx)
    if ekind is None:
        return None
    if ekind == "num":
        return pd.DataFrame(rng.normal(size=(n, 3)), index=idx, columns=["e1", "e2", "e3"])
    if ekind == "series":
        return pd.Series(rng.normal(size=n), index=idx, name="e1")
    if ekind == "mixed":
        return pd.DataFrame({
            "ef": rng.normal(size=n),
            "ei": rng.integers(0, 10, size=n),
            "eb": rng.integers(0, 2, size=n).astype(bool),
            "ecat": pd.Categorical(np.array(["a", "b", "c", "d"])[rng.integers(0, 4, size=n)]),
        }, index=idx)
    raise ValueError(ekind)


def long_of(frames):
    return pd.concat(frames, names=["series_id", "datetime"])


def series_as(series, fmt):
    if fmt == "wide":
        return series
    trimmed = {
        k: series[k].loc[series[k].first_valid_index():series[k].last_valid_index()]
        for k in series.columns
    }
    if fmt == "dict":
        return trimmed
    if fmt == "long":
        # The dates level keeps the frequency of the wide index; the rows
        # without value (before the start and after the end of a series) are
        # dropped, so every series has its own span.
        index = pd.MultiIndex.from_product(
            [list(series.columns), series.index], names=["series_id", "datetime"]
        )
        long = pd.DataFrame({"value": series.to_numpy().T.ravel()}, index=index)
        return long.dropna()
    raise ValueError(fmt)


def exog_as(exog, names, fmt, rows=slice(None)):
    """
    `exog` (wide, shared by all series) in the format `fmt`. With `dict_partial`
    one series has no key, another one is `None`, another one lacks a column
    and another one has fewer rows.
    """
    if exog is None:
        return None
    exog = exog.iloc[rows]
    if fmt == "wide":
        return exog
    frame = exog.to_frame() if isinstance(exog, pd.Series) else exog
    if fmt == "dict":
        return {k: exog for k in names}
    if fmt == "dict_partial":
        out = {k: exog for k in names}
        if len(names) > 3:
            out.pop(names[1])
            out[names[2]] = None
            out[names[3]] = frame.iloc[:, :-1] if frame.shape[1] > 1 else frame
        out[names[0]] = exog.iloc[:max(len(exog) - 1, 1)]
        return out
    if fmt == "long":
        return long_of({k: frame for k in names})
    if fmt == "long_series":
        return long_of({k: frame for k in names}).iloc[:, 0]
    raise ValueError(fmt)


# ----------------------------------------------------------------------------
# Forecasters
# ----------------------------------------------------------------------------
def make_ms(encoding="ordinal", cfg="base", lgbm=False, names=None):
    kw = {}
    if cfg == "tr_series":
        kw["transformer_series"] = StandardScaler()
    elif cfg == "tr_series_dict":
        kw["transformer_series"] = {
            **{k: (StandardScaler() if i % 2 else MinMaxScaler()) for i, k in enumerate(names[:-1])},
            "_unknown_level": StandardScaler(),
        }
    elif cfg == "tr_exog":
        kw["transformer_exog"] = StandardScaler()
    elif cfg == "diff":
        kw["differentiation"] = 1
    elif cfg == "diff_dict":
        kw["differentiation"] = {
            **{k: (1 if i % 2 else None) for i, k in enumerate(names)}, "_unknown_level": 1
        }
    elif cfg == "wf":
        kw["window_features"] = RollingFeatures(stats=["mean", "std"], window_sizes=7)
    elif cfg == "calendar":
        kw["calendar_features"] = CalendarFeatures(features=["month", "day_of_week"])
    elif cfg == "dropna":
        kw["dropna_from_series"] = True
    elif cfg != "base":
        raise ValueError(cfg)
    est = LGBMRegressor(n_estimators=5, verbose=-1, random_state=1) if lgbm else Ridge()
    return ForecasterRecursiveMultiSeries(estimator=est, lags=5, encoding=encoding, **kw)


def fitted_attributes(f):
    names = [
        "series_names_in_", "X_train_series_names_in_", "X_train_features_names_out_",
        "X_train_exog_names_out_", "exog_names_in_", "exog_in_", "exog_type_in_",
        "exog_dtypes_in_", "exog_dtypes_out_", "encoding_mapping_", "training_range_",
        "index_type_", "index_freq_", "last_window_", "window_size", "is_fitted",
        "in_sample_residuals_",
    ]
    return {name: getattr(f, name, "<no attribute>") for name in names}


# ----------------------------------------------------------------------------
# Predict cases
# ----------------------------------------------------------------------------
def predict_variants(f, series, exog, names, has_exog, steps, group, mode="mini"):
    """
    Predict cases of a forecaster `f` fitted with the rows `[:T]` of `series`
    and `exog` (wide data). `mode="full"` crosses the last windows with the
    ways of giving `levels`, the exog variants and the methods; `mode="mini"`
    is one pass over each of them.
    """
    ws = f.window_size
    unknown = "unknown_level" if isinstance(names[0], str) else 999
    lw_exact = series.iloc[T - ws:T]
    lw_unknown = lw_exact.copy()
    lw_unknown[unknown] = lw_exact.iloc[:, 0].to_numpy() * 1.1
    lw_nan = lw_exact.copy()
    lw_nan.iloc[-1, 0] = np.nan
    last_windows = {
        "stored": None,
        "exact": lw_exact,
        "longer": series.iloc[T - 3 * ws:T],
        "all": series.iloc[:T],
        "reordered": lw_exact.iloc[:, ::-1],
        "subset": lw_exact.iloc[:, : max(len(names) // 2, 1)],
        "unknown": lw_unknown,
        "nan": lw_nan,
        "named_index": lw_exact.rename_axis("date"),
        "named_columns": lw_exact.rename_axis("id", axis=1),
    }
    some = [names[0], names[-1]] if len(names) > 1 else [names[0]]
    levels_variants = {
        "None": None,
        "str": names[0],
        "list": some,
        "all_reversed": names[::-1],
        "index": pd.Index(some),
        "ndarray": np.array(some),
        "tuple": tuple(some),
        "unknown": [unknown],
        "known+unknown": [names[0], unknown],
        "empty": [],
        "duplicated": [names[0], names[0]],
    }

    def exog_variant(kind):
        if not has_exog:
            return {"none": None, "unexpected": exog_wide("num", series.index).iloc[T:T + steps]}[kind]
        full = exog.iloc[T:T + steps + 6]
        ex = full.iloc[:steps]
        if kind == "exact":
            return ex
        if kind == "none":
            return None
        if kind == "long_rows":
            return full
        if kind == "nan_in":
            ex = ex.copy()
            if isinstance(ex, pd.Series):
                ex.iloc[0] = np.nan
            else:
                ex.iloc[0, 0] = np.nan
            return ex
        if kind == "shifted":
            return full.iloc[1:steps + 1]
        if kind == "short":
            return full.iloc[:max(steps - 1, 0)]
        if kind == "reordered":
            return ex.iloc[:, ::-1] if isinstance(ex, pd.DataFrame) else ex
        if kind == "missing_col":
            return ex.iloc[:, 1:] if isinstance(ex, pd.DataFrame) else ex.rename("zz")
        if kind == "extra_col":
            return ex.assign(zz=1.0) if isinstance(ex, pd.DataFrame) else ex
        if kind in ("dict", "dict_partial", "long", "long_series"):
            return exog_as(ex, names, kind)
        if kind == "dict_unknown":  # exog also for a level not seen in training
            return {**{k: ex for k in names}, unknown: ex}
        if kind == "dict_long_rows":
            return {k: full for k in names}
        if kind == "dict_own":  # every level with its own object and values
            return {k: ex * (1 + i / 10) if isinstance(ex, pd.DataFrame) and ex.dtypes.eq(float).all()
                    else ex.copy() for i, k in enumerate(names)}
        raise ValueError(kind)

    default_exog = "exact" if has_exog else "none"
    is_datetime = isinstance(series.index, pd.DatetimeIndex)

    def run(tag, lw, levels, ek, method="predict", **extra):
        def fn():
            kw = dict(steps=steps, last_window=last_windows[lw],
                      levels=levels_variants[levels], exog=exog_variant(ek), **extra)
            if method == "predict":
                return f.predict(**kw)
            if method == "create_predict_X":
                return f.create_predict_X(**kw)
            if method == "interval_conformal":
                return f.predict_interval(method="conformal", interval=0.8, **kw)
            if method == "interval_boot":
                return f.predict_interval(
                    method="bootstrapping", n_boot=8, interval=[0.1, 0.9], **kw
                )
            if method == "bootstrapping":
                return f.predict_bootstrapping(n_boot=5, **kw)
            if method == "quantiles":
                return f.predict_quantiles(n_boot=8, quantiles=[0.1, 0.5, 0.9], **kw)
            if method == "dist":
                return f.predict_dist(distribution=stats.norm, n_boot=8, **kw)
            raise ValueError(method)

        record(f"P|{group}|steps:{steps}|{tag}", fn)

    if mode == "mini":
        for lw in ["stored", "exact", "longer"]:
            for levels in ["None", "list"]:
                run(f"lw:{lw} levels:{levels}", lw, levels, default_exog)
        run("lw:unknown levels:None", "unknown", "None", default_exog)
        run("lw:unknown levels:known+unknown", "unknown", "known+unknown", default_exog)
        run("lw:reordered levels:all_reversed", "reordered", "all_reversed", default_exog)
        if has_exog:
            ek_list = ["dict", "dict_partial", "dict_own", "shifted", "nan_in"]
            if is_datetime:
                ek_list.append("long")
            for ek in ek_list:
                run(f"exog:{ek} lw:stored", "stored", "None", ek)
                run(f"exog:{ek} lw:exact", "exact", "list", ek)
        for method in ["create_predict_X", "interval_conformal", "bootstrapping"]:
            run(f"{method} lw:exact", "exact", "None", default_exog, method=method)
        run("no_check lw:exact", "exact", "None", default_exog, check_inputs=False)
        return

    # 1. Levels x last window. The ways of giving `levels` that only raise an
    # error (tuple, empty, duplicated) are run with two windows.
    only_error = ("tuple", "empty", "duplicated")
    for lw in last_windows:
        for levels in levels_variants:
            if levels in only_error and lw not in ("stored", "exact"):
                continue
            run(f"lw:{lw} levels:{levels}", lw, levels, default_exog)
    # 2. Exog variants.
    if has_exog:
        ek_list = ["none", "long_rows", "nan_in", "shifted", "short", "reordered", "missing_col",
                   "extra_col", "dict", "dict_partial", "dict_unknown", "dict_long_rows",
                   "dict_own"]
        if is_datetime:
            ek_list += ["long", "long_series"]
    else:
        ek_list = ["unexpected"]
    for ek in ek_list:
        for lw, levels in [("stored", "None"), ("exact", "list"), ("unknown", "known+unknown"),
                           ("longer", "None")]:
            run(f"exog:{ek} lw:{lw} levels:{levels}", lw, levels, ek)
    # 3. Other methods.
    for method in ["create_predict_X", "interval_conformal", "interval_boot", "bootstrapping",
                   "quantiles", "dist"]:
        for lw, levels in [("stored", "None"), ("exact", "list"), ("unknown", "None")]:
            for ek in ([default_exog, "dict"] if has_exog else [default_exog]):
                run(f"{method} lw:{lw} levels:{levels} exog:{ek}", lw, levels, ek, method=method)
    for lw in ["stored", "exact", "longer"]:
        for ek in ([default_exog, "dict", "nan_in"] if has_exog else [default_exog]):
            run(f"no_check lw:{lw} exog:{ek}", lw, "None", ek, check_inputs=False)
            run(f"suppress lw:{lw} exog:{ek}", lw, "known+unknown", ek, suppress_warnings=True)


DATA_CFGS = {
    # tag: (n_series, index, lengths, names, format of series in fit,
    #       exog formats in fit run with the reduced set of predict cases)
    "n3": (3, "dt", "equal", "short", "wide", [None, "wide"]),
    "n20": (20, "dt", "equal", "str", "wide", ["dict_partial", "long"]),
    "n20_dict": (20, "dt", "equal", "str", "dict", ["dict"]),
    "n60_dict": (60, "dt", "equal", "str", "dict", [None, "dict"]),
    "n20_ragged_wide": (20, "dt", "ragged", "str", "wide", [None, "wide"]),
    "n20_ragged_dict": (20, "dt", "ragged", "str", "dict", [None, "dict_partial"]),
    "n12_ragged_long": (12, "dt", "ragged", "str", "long", ["long"]),
    "n8_long": (8, "dt", "equal", "unsorted", "long", [None, "long"]),
    "n8_tzD": (8, "tz_D", "equal", "str", "wide", ["wide"]),
    "n8_utcD": (8, "utc_D", "equal", "str", "dict", ["dict"]),
    "n8_range": (8, "range", "equal", "str", "dict", [None, "dict"]),
    "n8_MS": (8, "MS", "ragged", "str", "wide", ["wide"]),
    "n8_int": (8, "dt", "equal", "int", "wide", [None, "dict"]),
    "n8_unsorted": (8, "dt", "equal", "unsorted", "wide", ["long"]),
    "n1": (1, "dt", "equal", "str", "wide", ["wide"]),
}
# Groups run with the full set of predict cases (default encoding).
FULL_GROUPS = [("n20", None), ("n20", "wide"), ("n20", "dict"), ("n20_ragged_dict", "dict")]
ENCODINGS = ["ordinal", "ordinal_category", "onehot", None]


def data_of(tag):
    n_series, ikind, lengths, names_kind, fmt, efmts = DATA_CFGS[tag]
    series = wide_series(n_series, ikind, lengths, names_kind)
    return series, list(series.columns), fmt, efmts


def fit_and_predict(group, f, series, names, fmt, exog, efmt, steps_list, mode):
    fit_kw = dict(
        series=series_as(series.iloc[:T], fmt),
        exog=None if efmt is None else exog_as(exog, names, efmt, slice(0, T)),
        store_in_sample_residuals=True,
    )
    record(f"F|{group}|fit", lambda: (f.fit(**fit_kw), fitted_attributes(f))[1])
    if f.is_fitted:
        for steps in steps_list:
            predict_variants(f, series, exog, names, efmt is not None, steps, group, mode)


def predict_cases():
    # 1. Full set of cases: 20 series, default encoding, no exog and exog in
    # wide and dict format, and series of different length.
    for tag, efmt in FULL_GROUPS:
        series, names, fmt, _ = data_of(tag)
        exog = exog_wide("num", series.index)
        fit_and_predict(f"ms|{tag}|enc:ordinal|exog_fit:{efmt}", make_ms(), series, names, fmt,
                        exog, efmt, [1, 5], "full")
    # 2. Every data set (number of series, index, names, input format) with
    # the reduced set of cases.
    for tag in DATA_CFGS:
        series, names, fmt, efmts = data_of(tag)
        exog = exog_wide("num", series.index)
        for efmt in efmts:
            fit_and_predict(f"ms|{tag}|enc:ordinal|exog_fit:{efmt}", make_ms(), series, names,
                            fmt, exog, efmt, [3], "mini")
    # 3. The other encodings, with and without exog, equal and different lengths.
    for tag, efmt in [("n20", None), ("n20", "dict"), ("n20_ragged_dict", None),
                      ("n20_ragged_dict", "dict"), ("n8_long", "long")]:
        series, names, fmt, _ = data_of(tag)
        exog = exog_wide("num", series.index)
        for encoding in ENCODINGS[1:]:
            fit_and_predict(f"ms|{tag}|enc:{encoding}|exog_fit:{efmt}", make_ms(encoding), series,
                            names, fmt, exog, efmt, [1, 3], "mini")
    # 4. Forecaster configurations and exog of several dtypes.
    for tag in ["n20", "n20_ragged_dict"]:
        series, names, fmt, _ = data_of(tag)
        for cfg in ["tr_series", "tr_series_dict", "tr_exog", "diff", "diff_dict", "wf",
                    "calendar", "dropna"]:
            for ekind in (["num", "series"] if cfg == "tr_exog" else [None, "num"]):
                exog = exog_wide(ekind, series.index)
                fit_and_predict(f"ms|{tag}|cfg:{cfg}|exog:{ekind}", make_ms(cfg=cfg, names=names),
                                series, names, fmt, exog, None if ekind is None else "wide",
                                [3], "mini")
        for encoding in ENCODINGS:
            exog = exog_wide("mixed", series.index)
            for efmt in ["wide", "dict"]:
                fit_and_predict(f"ms|{tag}|lgbm|enc:{encoding}|exog:mixed_{efmt}",
                                make_ms(encoding, lgbm=True), series, names, fmt, exog, efmt,
                                [3], "mini")


# ----------------------------------------------------------------------------
# Training matrices
# ----------------------------------------------------------------------------
def train_matrix_cases():
    for tag in DATA_CFGS:
        series, names, fmt, efmts = data_of(tag)
        exog = exog_wide("mixed", series.index)
        for efmt in efmts:
            for encoding in ENCODINGS:
                f = make_ms(encoding)
                kw = dict(series=series_as(series, fmt),
                          exog=None if efmt is None else exog_as(exog, names, efmt))
                record(f"X|ms|{tag}|enc:{encoding}|exog_fit:{efmt}|cfg:base",
                       lambda: f.create_train_X_y(**kw))
            for cfg in ["diff", "wf", "dropna"]:
                f = make_ms(cfg=cfg, names=names)
                kw = dict(series=series_as(series, fmt),
                          exog=None if efmt is None else exog_as(exog, names, efmt))
                record(f"X|ms|{tag}|enc:ordinal|exog_fit:{efmt}|cfg:{cfg}",
                       lambda: f.create_train_X_y(**kw))


# ----------------------------------------------------------------------------
# Backtesting cases
# ----------------------------------------------------------------------------
CVS = {
    "plain": {},
    "steps1": {"steps": 1},
    "steps7": {"steps": 7},
    "refit": {"refit": True},
    "refit2": {"refit": 2},
    "gap": {"gap": 2},
    "fixed": {"refit": True, "fixed_train_size": True},
    "incomplete": {"allow_incomplete_fold": False, "steps": 7},
    "skip": {"skip_folds": 2},
    "stride": {"fold_stride": 1},
    "one_fold": {"initial_train_size": N - 3},
    "early": {"initial_train_size": 20},  # ragged series: levels not in the first train sets
}
BACKTESTING_FULL = ("n20", "n20_ragged_dict", "n8_long")


def backtesting_cases():
    def run(name, forecaster, series, exog, cv_kw, **bt_kw):
        def fn():
            cv = TimeSeriesFold(**{"steps": 3, "initial_train_size": N - 40, **cv_kw})
            kw = {"cv": cv, "exog": exog, "metric": "mean_absolute_error", "n_jobs": 1,
                  "verbose": False, "show_progress": False, **bt_kw}
            return backtesting_forecaster_multiseries(forecaster=forecaster(), series=series, **kw)

        record(f"B|{name}", fn)

    for tag in DATA_CFGS:
        series, names, fmt, efmts = data_of(tag)
        exog = exog_wide("num", series.index)
        some = [names[0], names[-1]]
        is_full = tag in BACKTESTING_FULL
        if is_full:
            efmts = [None, "wide", "dict"] + (["long"] if tag == "n8_long" else ["dict_partial"])
        s = series_as(series, fmt)
        for efmt in efmts:
            ex = None if efmt is None else exog_as(exog, names, efmt)
            base = f"ms|{tag}|exog:{efmt}"
            # Folds: all the kinds with the main data sets, the three that
            # differ most (no refit, 1 step, refit) with the rest.
            for cv_name, cv_kw in CVS.items():
                if is_full or cv_name in ("plain", "steps1", "refit"):
                    run(f"{base}|cv:{cv_name}", make_ms, s, ex, cv_kw)
            if not is_full or efmt not in (None, "dict"):
                continue
            # The same data in the other input formats.
            for other in ["wide", "dict", "long"]:
                if other != fmt:
                    run(f"{base}|series_fmt:{other}", make_ms, series_as(series, other), ex, {})
            for encoding in ENCODINGS[1:]:
                run(f"{base}|enc:{encoding}", lambda e=encoding: make_ms(e), s, ex, {})
                run(f"{base}|enc:{encoding}|refit", lambda e=encoding: make_ms(e), s, ex,
                    {"refit": True})
            run(f"{base}|levels:str", make_ms, s, ex, {}, levels=names[0])
            run(f"{base}|levels:list", make_ms, s, ex, {}, levels=some)
            run(f"{base}|levels:reversed", make_ms, s, ex, {}, levels=names[::-1])
            run(f"{base}|levels:unknown", make_ms, s, ex, {}, levels=[names[0], "zz"])
            run(f"{base}|no_aggregated", make_ms, s, ex, {}, add_aggregated_metric=False)
            run(f"{base}|metrics", make_ms, s, ex, {},
                metric=["mean_absolute_error", "mean_absolute_scaled_error"])
            run(f"{base}|predictors", make_ms, s, ex, {}, return_predictors=True)
            run(f"{base}|conformal", make_ms, s, ex, {}, interval=0.8, interval_method="conformal")
            run(f"{base}|boot", make_ms, s, ex, {}, interval=[0.1, 0.9],
                interval_method="bootstrapping", n_boot=8)
            run(f"{base}|boot_refit", make_ms, s, ex, {"refit": True}, interval=[0.1, 0.9],
                interval_method="bootstrapping", n_boot=8, use_binned_residuals=False)
            run(f"{base}|suppress", make_ms, s, ex, {}, suppress_warnings=True)
            for cfg in ["tr_series", "diff", "wf", "calendar", "dropna"]:
                run(f"{base}|cfg:{cfg}", lambda c=cfg: make_ms(cfg=c, names=names), s, ex, {})
            run(f"{base}|lgbm", lambda: make_ms(lgbm=True), s, ex, {})
            run(f"{base}|lgbm|refit", lambda: make_ms(lgbm=True), s, ex, {"refit": True})

            # Forecaster already fitted, no initial training.
            def fitted():
                f = make_ms()
                f.fit(series=series_as(series.iloc[:N - 60], fmt),
                      exog=None if efmt is None else exog_as(exog, names, efmt, slice(0, N - 60)))
                return f

            run(f"{base}|externally_fitted", fitted, s, ex, {"initial_train_size": None})
        if tag in ("n20", "n20_ragged_dict"):
            run(f"ms|{tag}|n_jobs:2", make_ms, s, exog_as(exog, names, "dict"), {}, n_jobs=2)
            run(f"ms|{tag}|n_jobs:2|refit", make_ms, s, exog, {"refit": True}, n_jobs=2)
            mixed = exog_wide("mixed", series.index)
            for efmt in ["wide", "dict", "dict_partial"]:
                run(f"ms|{tag}|lgbm|mixed:{efmt}", lambda: make_ms(lgbm=True), s,
                    exog_as(mixed, names, efmt), {})

    # ForecasterDirectMultiVariate with more series than identity_trims.py.
    for n_series, ikind in [(10, "dt"), (10, "range"), (6, "tz_D")]:
        series = wide_series(n_series, ikind, "equal", "str")
        names = list(series.columns)
        for ekind in [None, "num", "mixed"]:
            exog = exog_wide(ekind, series.index)
            for cv_name in ["plain", "steps1", "refit", "gap"]:
                def make_mv(steps=3 + CVS[cv_name].get("gap", 0), ekind=ekind):
                    est = (LGBMRegressor(n_estimators=5, verbose=-1, random_state=1)
                           if ekind == "mixed" else Ridge())
                    return ForecasterDirectMultiVariate(
                        estimator=est, level=names[1], lags=5, steps=steps
                    )

                run(f"mv|n{n_series}|idx:{ikind}|exog:{ekind}|cv:{cv_name}", make_mv, series,
                    exog, CVS[cv_name])


def direct_mv_predict_cases():
    for n_series, ikind, ekinds in [(10, "dt", [None, "num", "mixed"]), (10, "range", ["num"]),
                                    (6, "tz_D", [None, "num"]), (6, "utc_D", ["num"])]:
        series = wide_series(n_series, ikind, "equal", "str")
        names = list(series.columns)
        for ekind in ekinds:
            exog = exog_wide(ekind, series.index)
            lags_dict = {**{k: 4 for k in names}, names[0]: 3, names[1]: [1, 5], names[2]: None}
            for lags in ([5, lags_dict] if (ikind, ekind) == ("dt", "num") else [5]):
                est = (LGBMRegressor(n_estimators=5, verbose=-1, random_state=1)
                       if ekind == "mixed" else Ridge())
                f = ForecasterDirectMultiVariate(estimator=est, level=names[1], lags=lags, steps=4)
                group = f"mv|n{n_series}|idx:{ikind}|exog:{ekind}|lags:{'dict' if isinstance(lags, dict) else lags}"
                fit_kw = dict(series=series.iloc[:T], exog=None if exog is None else exog.iloc[:T],
                              store_in_sample_residuals=True)
                record(f"F|{group}|fit", lambda: (f.fit(**fit_kw), fitted_attributes(f))[1])
                if not f.is_fitted:
                    continue
                ws = f.window_size
                lw_exact = series.iloc[T - ws:T]
                lw_nan = lw_exact.copy()
                lw_nan.iloc[-1, :] = np.nan
                last_windows = {
                    "stored": None, "exact": lw_exact, "longer": series.iloc[T - 3 * ws:T],
                    "reordered": lw_exact.iloc[:, ::-1],
                    "extra": lw_exact.assign(zz=1.0),
                    "missing": lw_exact.iloc[:, 1:],
                    "nan": lw_nan,
                }
                for lw_name, lw in last_windows.items():
                    for steps in [None, [1, 4]]:
                        n_steps = 4
                        full = None if exog is None else exog.iloc[T:T + n_steps + 3]
                        exogs = {"exact": None if full is None else full.iloc[:n_steps]}
                        if full is not None and lw_name in ("stored", "exact"):
                            exogs["long_rows"] = full
                            exogs["shifted"] = full.iloc[1:n_steps + 1]
                            exogs["reordered"] = full.iloc[:n_steps, ::-1]
                        for ex_name, ex in exogs.items():
                            name = f"P|{group}|lw:{lw_name} steps:{steps} exog:{ex_name}"
                            record(name, lambda: f.predict(steps=steps, last_window=lw, exog=ex))
                            if ex_name != "exact":
                                continue
                            record(name + " X", lambda: f.create_predict_X(
                                steps=steps, last_window=lw, exog=ex))
                            if lw_name in ("stored", "exact"):
                                record(name + " conformal", lambda: f.predict_interval(
                                    steps=steps, last_window=lw, exog=ex, method="conformal",
                                    interval=0.8))
                                record(name + " boot", lambda: f.predict_interval(
                                    steps=steps, last_window=lw, exog=ex, method="bootstrapping",
                                    interval=[0.1, 0.9], n_boot=8))
                                record(name + " levels", lambda: f.predict(
                                    steps=steps, last_window=lw, exog=ex, levels=names[1]))
                                record(name + " no_check", lambda: f.predict(
                                    steps=steps, last_window=lw, exog=ex, check_inputs=False))


if __name__ == "__main__":
    import time

    t_start = time.perf_counter()
    predict_cases()
    direct_mv_predict_cases()
    train_matrix_cases()
    backtesting_cases()
    n_err = sum(r["err"] is not None for r in RESULTS.values())
    n_warn = sum(bool(r["warn"]) for r in RESULTS.values())
    with open(sys.argv[1], "wb") as fh:
        pickle.dump(RESULTS, fh)
    storage = pd.Series(["a"]).dtype.storage if pd.__version__ >= "3" else "object"
    print(f"pandas {pd.__version__} ({storage}) | {os.path.dirname(SKFORECAST_ROOT)[-24:]} | "
          f"{len(RESULTS)} cases, {n_err} errors, {n_warn} with warnings, "
          f"{time.perf_counter() - t_start:.0f} s")
