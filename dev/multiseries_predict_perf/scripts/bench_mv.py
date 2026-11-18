"""
Timing of ForecasterDirectMultiVariate predict and backtesting without refit,
by number of series, to see if it loses with pandas 3 as
ForecasterRecursiveMultiSeries does.

Usage: python bench_mv.py <out.json> [regex of case names]
"""
import common  # noqa: F401

import json
import re
import sys
import time

from common import LAGS, TimeSeriesFold, backtesting_forecaster_multiseries, env_info, make_wide
from sklearn.linear_model import LinearRegression
from skforecast.direct import ForecasterDirectMultiVariate

T = 400


def make(series, steps):
    return ForecasterDirectMultiVariate(
        LinearRegression(), level=series.columns[0], lags=LAGS, steps=steps
    )


def predict_case(n_series, with_exog, steps=1, lw="given"):
    n_calls = 200
    series, exog = make_wide(n_series, T + n_calls + steps)
    f = make(series, steps)
    f.fit(series=series.iloc[:T], exog=exog.iloc[:T] if with_exog else None)
    if lw == "given":
        inputs = [
            (series.iloc[T + i - LAGS: T + i],
             exog.iloc[T + i: T + i + steps] if with_exog else None)
            for i in range(n_calls)
        ]
    else:
        inputs = [(None, exog.iloc[T: T + steps] if with_exog else None)] * n_calls

    def run():
        t0 = time.perf_counter()
        for last_window, ex in inputs:
            f.predict(last_window=last_window, exog=ex)
        return (time.perf_counter() - t0) / n_calls * 1e6

    run()
    return {"us_per_call": min(run() for _ in range(5))}


def backtesting_case(n_series, with_exog, steps=1):
    n_folds = 100
    series, exog = make_wide(n_series, T + n_folds * steps)
    cv = TimeSeriesFold(steps=steps, initial_train_size=T, refit=False)
    kw = dict(
        forecaster=make(series, steps), series=series, exog=exog if with_exog else None,
        cv=cv, metric="mean_absolute_error", n_jobs=1, verbose=False, show_progress=False,
        suppress_warnings=True,
    )
    fit_seconds = []
    fit_orig = ForecasterDirectMultiVariate.fit

    def fit_timed(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fit_orig(self, *args, **kwargs)
        finally:
            fit_seconds.append(time.perf_counter() - t0)

    def run():
        fit_seconds.clear()
        ForecasterDirectMultiVariate.fit = fit_timed
        try:
            t0 = time.perf_counter()
            backtesting_forecaster_multiseries(**kw)
            total = time.perf_counter() - t0
        finally:
            ForecasterDirectMultiVariate.fit = fit_orig
        return total, sum(fit_seconds)

    run()
    runs = [run() for _ in range(3)]
    return {"ms_per_fold_no_fit": min(t - fit for t, fit in runs) * 1e3 / n_folds,
            "n_folds": n_folds}


CASES = {}
for n in (3, 20, 100):
    for we in (False, True):
        ex = "wide" if we else "none"
        CASES[f"predict|n={n}|exog={ex}|steps=1|mv"] = lambda n=n, we=we: predict_case(n, we)
        CASES[f"predict|n={n}|exog={ex}|steps=1|lw=stored|mv"] = (
            lambda n=n, we=we: predict_case(n, we, lw="stored")
        )
    CASES[f"backtesting|n={n}|fmt=wide|exog=wide|steps=1|mv"] = (
        lambda n=n: backtesting_case(n, True)
    )

if __name__ == "__main__":
    out = sys.argv[1]
    pattern = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
    res = {}
    for name, fn in CASES.items():
        if pattern and not pattern.search(name):
            continue
        res[name] = fn()
        with open(out, "w") as fh:
            json.dump({"res": res, **env_info()}, fh)
    info = env_info()
    print(info["pandas"], info["storage"], "pyarrow" if info["pyarrow_loaded"] else "no-pyarrow",
          {k: round(next(iter(v.values()))) for k, v in res.items()})
