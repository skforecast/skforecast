"""
Timing of ForecasterRecursiveMultiSeries: predict, backtesting (without refit,
and with refit in @adverse), fit and create_train_X_y, by number of series,
exog and input format.

Usage: python bench_ms.py <out.json> [@core | @adverse | @scale | regex of case names]
       python bench_ms.py --list
Env: see common.py. Inputs are built before the timed region.
"""
import common  # noqa: F401  (first: blocks pyarrow, pins the process)

import json
import re
import sys
import time

import pandas as pd

from common import (
    LAGS, TimeSeriesFold, backtesting_forecaster_multiseries, env_info,
    exog_for_predict, make_forecaster, make_wide, to_format,
)
from skforecast.recursive import ForecasterRecursiveMultiSeries

T = 400        # training rows of the predict and backtesting cases
T_FIT = 500    # training rows of the fit and create_train_X_y cases


def reps_of(n_series):
    return 5 if n_series <= 100 else 3


# ----------------------------------------------------------------------------
# predict
# ----------------------------------------------------------------------------
def predict_case(n_series, exog_kind, steps=1, est="linear", lw="given",
                 encoding="ordinal", n_levels=None, extra_rows=0, early=0,
                 index_name=None, exog_dtypes="float", missing=0, nofreq=False):
    """
    `n_levels`: predict only the first `n_levels` series (the window keeps all
    the columns). `extra_rows`: exog with more rows than steps. `early`: exog
    that starts `early` steps before the first predicted date and still has
    all the predicted dates (it has to be aligned by date). `index_name`: name
    of the exog index (the index of the predictions has none, so the exog is
    aligned). `exog_dtypes`: see `make_wide`. `missing`: the last `missing`
    series have no exog in the dict (their exog features are NaN). `nofreq`:
    exog index without `freq` (as read from a file), same dates.
    """
    n_calls = {3: 200, 20: 200, 100: 100}.get(n_series, 20)
    if est == "lgbm":
        n_calls = min(n_calls, 50)
    series, exog = make_wide(
        n_series, T + n_calls + steps + extra_rows, exog_dtypes=exog_dtypes
    )
    if index_name is not None:
        exog = exog.rename_axis(index_name)
    if nofreq:
        exog = exog.set_axis(pd.DatetimeIndex(exog.index, freq=None))
        assert exog.index.freq is None
    names = list(series.columns)
    levels = None if n_levels is None else names[:n_levels]
    f = make_forecaster(est, encoding)
    f.fit(series=series.iloc[:T], exog=None if exog_kind == "none" else exog.iloc[:T])
    if lw == "given":
        inputs = [
            (series.iloc[T + i - LAGS: T + i],
             exog_for_predict(
                 exog.iloc[T + i - early: T + i + steps + extra_rows], names, exog_kind
             ))
            for i in range(n_calls)
        ]
        if missing:
            inputs = [
                (lw_, {k: v for k, v in ex_.items() if k not in names[-missing:]})
                for lw_, ex_ in inputs
            ]
    else:  # window stored in fit: every call predicts the same steps
        ex = exog_for_predict(exog.iloc[T: T + steps], names, exog_kind)
        inputs = [(None, ex)] * n_calls

    def run():
        t0 = time.perf_counter()
        for last_window, ex in inputs:
            f.predict(steps=steps, levels=levels, last_window=last_window, exog=ex)
        return (time.perf_counter() - t0) / n_calls * 1e6

    run()
    return {"us_per_call": min(run() for _ in range(reps_of(n_series)))}


# ----------------------------------------------------------------------------
# backtesting (without refit, or with refit in every fold)
# ----------------------------------------------------------------------------
_fit_seconds = []
_fit_orig = ForecasterRecursiveMultiSeries.fit


def _fit_timed(self, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        return _fit_orig(self, *args, **kwargs)
    finally:
        _fit_seconds.append(time.perf_counter() - t0)


def backtesting_case(n_series, fmt, with_exog, steps=1, est="linear", refit=False,
                     exog_dtypes="float", index_name=None, nofreq=False):
    n_folds = {1: 100, 24: 10}[steps] if n_series <= 100 else {1: 20, 24: 5}[steps]
    if refit:
        n_folds = 20
    series, exog = make_wide(n_series, T + n_folds * steps, exog_dtypes=exog_dtypes)
    if index_name is not None:
        exog = exog.rename_axis(index_name)
    if nofreq:
        exog = exog.set_axis(pd.DatetimeIndex(exog.index, freq=None))
        assert exog.index.freq is None
    series, exog = to_format(series, exog if with_exog else None, fmt)
    cv = TimeSeriesFold(steps=steps, initial_train_size=T, refit=refit)
    kw = dict(
        forecaster=make_forecaster(est), series=series, exog=exog, cv=cv,
        metric="mean_absolute_error", n_jobs=1, verbose=False, show_progress=False,
        suppress_warnings=True,
    )

    def run():
        _fit_seconds.clear()
        ForecasterRecursiveMultiSeries.fit = _fit_timed
        try:
            t0 = time.perf_counter()
            backtesting_forecaster_multiseries(**kw)
            total = time.perf_counter() - t0
        finally:
            ForecasterRecursiveMultiSeries.fit = _fit_orig
        assert len(_fit_seconds) == (n_folds if refit else 1)
        return total, sum(_fit_seconds)

    run()
    runs = [run() for _ in range(reps_of(n_series))]
    return {
        "ms_total": min(t for t, _ in runs) * 1e3,
        "ms_fit": min(fit for _, fit in runs) * 1e3,
        # Everything but the initial fit, per fold.
        "ms_per_fold_no_fit": min(t - fit for t, fit in runs) * 1e3 / n_folds,
        "n_folds": n_folds,
    }


# ----------------------------------------------------------------------------
# fit and create_train_X_y
# ----------------------------------------------------------------------------
def fit_case(n_series, fmt, with_exog, method):
    series, exog = make_wide(n_series, T_FIT)
    series, exog = to_format(series, exog if with_exog else None, fmt)
    f = make_forecaster("linear")

    def run():
        t0 = time.perf_counter()
        if method == "fit":
            f.fit(series=series, exog=exog, suppress_warnings=True)
        else:
            f.create_train_X_y(series=series, exog=exog, suppress_warnings=True)
        return (time.perf_counter() - t0) * 1e3

    run()
    return {"ms": min(run() for _ in range(reps_of(n_series)))}


# ----------------------------------------------------------------------------
CASES = {}
for n in (3, 20, 100, 600, 1000):
    for ek in ("none", "wide", "dict", "dictown", "long"):
        CASES[f"predict|n={n}|exog={ek}|steps=1"] = (
            lambda n=n, ek=ek: predict_case(n, ek)
        )
    for ek in ("none", "wide"):
        CASES[f"predict|n={n}|exog={ek}|steps=1|lw=stored"] = (
            lambda n=n, ek=ek: predict_case(n, ek, lw="stored")
        )
for n in (20, 100):
    for ek in ("none", "wide", "dict"):
        CASES[f"predict|n={n}|exog={ek}|steps=24"] = (
            lambda n=n, ek=ek: predict_case(n, ek, steps=24)
        )
    for ek in ("none", "wide"):
        for steps in (1, 24):
            CASES[f"predict|n={n}|exog={ek}|steps={steps}|lgbm"] = (
                lambda n=n, ek=ek, steps=steps: predict_case(n, ek, steps=steps, est="lgbm")
            )
    for enc in ("onehot", "none"):
        CASES[f"predict|n={n}|exog=wide|steps=1|enc={enc}"] = (
            lambda n=n, enc=enc: predict_case(
                n, "wide", encoding=None if enc == "none" else enc
            )
        )
for n in (3, 20, 100, 1000):
    for fmt in ("wide", "dict", "long"):
        for we in (False, True):
            ex = fmt if we else "none"
            CASES[f"backtesting|n={n}|fmt={fmt}|exog={ex}|steps=1"] = (
                lambda n=n, fmt=fmt, we=we: backtesting_case(n, fmt, we)
            )
# Backtesting with refit (20 folds): the training set of every fold is new.
for n in (20, 100):
    for fmt, ex in (("wide", "none"), ("dict", "dict")):
        CASES[f"backtesting|n={n}|fmt={fmt}|exog={ex}|steps=1|refit"] = (
            lambda n=n, fmt=fmt, we=ex != "none": backtesting_case(
                n, fmt, we, refit=True
            )
        )
for n in (20, 100):
    for fmt in ("wide", "dict"):
        for we in (False, True):
            ex = fmt if we else "none"
            CASES[f"backtesting|n={n}|fmt={fmt}|exog={ex}|steps=24"] = (
                lambda n=n, fmt=fmt, we=we: backtesting_case(n, fmt, we, steps=24)
            )
    CASES[f"backtesting|n={n}|fmt=wide|exog=wide|steps=24|lgbm"] = (
        lambda n=n: backtesting_case(n, "wide", True, steps=24, est="lgbm")
    )
for n in (20, 100, 1000):
    for fmt in ("wide", "dict", "long"):
        for we in (False, True):
            ex = fmt if we else "none"
            for method in ("fit", "create_train_X_y"):
                CASES[f"{method}|n={n}|fmt={fmt}|exog={ex}"] = (
                    lambda n=n, fmt=fmt, we=we, method=method: fit_case(n, fmt, we, method)
                )

# Cases where a fast path does not apply and only its check is paid: they must
# not get slower.
for n in (20, 100, 1000):
    CASES[f"predict|n={n}|exog=none|steps=1|levels=5"] = (
        lambda n=n: predict_case(n, "none", n_levels=5)
    )
for n in (20, 100):
    CASES[f"predict|n={n}|exog=wide|steps=1|extra=10"] = (
        lambda n=n: predict_case(n, "wide", extra_rows=10)
    )
    CASES[f"predict|n={n}|exog=dictown|steps=1|extra=10"] = (
        lambda n=n: predict_case(n, "dictown", extra_rows=10)
    )
    CASES[f"predict|n={n}|exog=dictown|steps=1|early=1"] = (
        lambda n=n: predict_case(n, "dictown", early=1)
    )
    CASES[f"predict|n={n}|exog=dictown|steps=1|named"] = (
        lambda n=n: predict_case(n, "dictown", index_name="date")
    )
for fmt, ex in (("wide", "none"), ("wide", "wide"), ("dict", "dict")):
    CASES[f"backtesting|n=600|fmt={fmt}|exog={ex}|steps=1"] = (
        lambda fmt=fmt, ex=ex: backtesting_case(600, fmt, ex != "none")
    )

# Change 7: dict exog without the exog of one series (NaN features, so LightGBM).
for n in (20, 100):
    CASES[f"predict|n={n}|exog=dictown|steps=1|missing=1|lgbm"] = (
        lambda n=n: predict_case(n, "dictown", est="lgbm", missing=1)
    )

# Change 4b: exog with the dates of the predictions but an index that is not
# `identical` (named, or without `freq`), and the adverse case with other dates
# and a named index (it now compares the dates before `reindex`).
for n in (20, 100):
    CASES[f"predict|n={n}|exog=dictown|steps=1|nofreq"] = (
        lambda n=n: predict_case(n, "dictown", nofreq=True)
    )
    CASES[f"predict|n={n}|exog=dictown|steps=1|early=1|named"] = (
        lambda n=n: predict_case(n, "dictown", early=1, index_name="date")
    )
    for fmt in ("wide", "dict"):
        CASES[f"backtesting|n={n}|fmt={fmt}|exog={fmt}|steps=1|named"] = (
            lambda n=n, fmt=fmt: backtesting_case(n, fmt, True, index_name="date")
        )
        CASES[f"backtesting|n={n}|fmt={fmt}|exog={fmt}|steps=1|nofreq"] = (
            lambda n=n, fmt=fmt: backtesting_case(n, fmt, True, nofreq=True)
        )

# Exog with an `int` and a `category` column (calendar features), not only float.
for n in (20, 100, 600, 1000):
    for ek in ("wide", "dictown", "long"):
        CASES[f"predict|n={n}|exog={ek}|steps=1|exogdt=mixed"] = (
            lambda n=n, ek=ek: predict_case(n, ek, exog_dtypes="mixed")
        )
for n in (20, 100, 600):
    for fmt in ("wide", "dict"):
        CASES[f"backtesting|n={n}|fmt={fmt}|exog={fmt}|steps=1|exogdt=mixed"] = (
            lambda n=n, fmt=fmt: backtesting_case(n, fmt, True, exog_dtypes="mixed")
        )

# Sets of cases: `@core` for every change, `@adverse` to show that nothing gets
# slower, `@scale` (one round) for many series.
SETS = {
    "core": [
        f"predict|n={n}|exog={ek}|steps=1"
        for n in (20, 100) for ek in ("none", "wide", "dict", "dictown", "long")
    ] + [
        f"predict|n={n}|exog=wide|steps=1|lw=stored" for n in (20, 100)
    ] + [
        "predict|n=20|exog=wide|steps=24|lgbm",
    ] + [
        f"backtesting|n={n}|fmt={fmt}|exog={ex}|steps=1"
        for n in (20, 100) for fmt, ex in (("wide", "none"), ("wide", "wide"), ("dict", "dict"))
    ] + [
        "backtesting|n=20|fmt=wide|exog=wide|steps=24|lgbm",
        "fit|n=100|fmt=wide|exog=wide", "fit|n=100|fmt=long|exog=long",
        "create_train_X_y|n=100|fmt=wide|exog=wide",
        "create_train_X_y|n=100|fmt=long|exog=long",
    ] + [
        f"predict|n={n}|exog={ek}|steps=1|exogdt=mixed"
        for n in (20, 100) for ek in ("wide", "dictown", "long")
    ] + [
        f"backtesting|n={n}|fmt={fmt}|exog={fmt}|steps=1|exogdt=mixed"
        for n in (20, 100) for fmt in ("wide", "dict")
    ],
    "adverse": [
        "predict|n=100|exog=none|steps=1|levels=5", "predict|n=1000|exog=none|steps=1|levels=5",
    ] + [
        f"predict|n={n}|exog={ek}|steps=1|{tag}"
        for n in (20, 100)
        for ek, tag in (
            ("wide", "extra=10"), ("dictown", "extra=10"), ("dictown", "early=1"),
            ("dictown", "named"),
        )
    ] + [
        f"backtesting|n={n}|fmt={fmt}|exog={ex}|steps=1|refit"
        for n in (20, 100) for fmt, ex in (("wide", "none"), ("dict", "dict"))
    ],
    "scale": [
        f"predict|n={n}|exog={ek}|steps=1"
        for n in (600, 1000) for ek in ("none", "wide", "dictown", "long")
    ] + [
        f"backtesting|n=600|fmt={fmt}|exog={ex}|steps=1"
        for fmt, ex in (("wide", "none"), ("wide", "wide"), ("dict", "dict"))
    ] + [
        f"predict|n={n}|exog={ek}|steps=1|exogdt=mixed"
        for n in (600, 1000) for ek in ("wide", "dictown", "long")
    ] + [
        f"backtesting|n=600|fmt={fmt}|exog={fmt}|steps=1|exogdt=mixed"
        for fmt in ("wide", "dict")
    ],
}
for _names in SETS.values():
    assert all(name in CASES for name in _names), [n for n in _names if n not in CASES]

if __name__ == "__main__":
    if "--list" in sys.argv:
        print("\n".join(CASES))
        sys.exit()
    out = sys.argv[1]
    selection = sys.argv[2] if len(sys.argv) > 2 else None
    selected = SETS[selection[1:]] if selection and selection.startswith("@") else None
    pattern = re.compile(selection) if selection and selected is None else None
    res = {}
    t_start = time.perf_counter()
    for name, fn in CASES.items():
        if selected is not None and name not in selected:
            continue
        if pattern and not pattern.search(name):
            continue
        res[name] = fn()
        with open(out, "w") as fh:
            json.dump({"res": res, **env_info()}, fh)
    info = env_info()
    print(info["pandas"], info["storage"], "pyarrow" if info["pyarrow_loaded"] else "no-pyarrow",
          info["tree"][-30:], f"{len(res)} cases in {time.perf_counter() - t_start:.0f} s")
