"""
Review of T1, points 4, 6 and 7: cost of the inf check, acceptance criteria and
design alternatives.

python dev/preprocessing_speedups/review_t1_bench.py micro
python dev/preprocessing_speedups/review_t1_bench.py macro --size M --est ridge
python dev/preprocessing_speedups/review_t1_bench.py macro --size M --est lgbm
python dev/preprocessing_speedups/review_t1_bench.py backtesting --size M --est ridge
python dev/preprocessing_speedups/review_t1_bench.py others

Run it with nothing else running on the machine. The candidates are those of
`review_t1_common.py`: PLAN (T1 as written), PROTO (the prototype) and PROPOSED.
"""

from __future__ import annotations

import argparse
import gc
import math
import time
import timeit
import warnings

import numpy as np
import pandas as pd
import sklearn
from prototypes import fast_transform_numpy
from review_t1_common import (
    ORIG_TRANSFORM_NUMPY,
    Counted,
    env_line,
    patched,
    plan_applies,
    plan_literal,
    proposed,
    proposed_applies,
    proposed_isinf_first,
    prototype_conditions,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def per_call(fn, min_time=0.2):
    """Best time per call, in microseconds."""
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    number = max(1, int(number * min_time / 0.2))
    return min(timer.repeat(repeat=7, number=number)) / number * 1e6


# ----------------------------------------------------------------------- micro


def current_assume_finite(array, transformer, fit=False, inverse_transform=False):
    """Alternative A: the current path with scikit-learn's finite check disabled."""
    with sklearn.config_context(assume_finite=True, skip_parameter_validation=True):
        return ORIG_TRANSFORM_NUMPY(array=array, transformer=transformer, fit=fit,
                                    inverse_transform=inverse_transform)


def micro():
    rng = np.random.default_rng(0)
    sc = StandardScaler().fit((50 + rng.normal(0, 5, 2000)).reshape(-1, 1))
    sc_names = StandardScaler().fit(pd.DataFrame({"y": 50 + rng.normal(0, 5, 2000)}))
    matrix = 50 + rng.normal(0, 5, (168, 300))

    def nan(a, frac=0.1):
        a = a.copy()
        a.ravel()[rng.random(a.size) < frac] = np.nan
        return a

    arrays = {
        "1D n=48": 50 + rng.normal(0, 5, 48),
        "1D n=168": 50 + rng.normal(0, 5, 168),
        "1D n=168, column of a matrix": matrix[:, 7],
        "1D n=48 with NaN": nan(50 + rng.normal(0, 5, 48)),
        "1D n=168 with NaN": nan(50 + rng.normal(0, 5, 168)),
        "2D (24, 3)": rng.normal(0, 2, (24, 3)),
        "2D (168, 250)": rng.normal(0, 2, (168, 250)),
        "2D (168, 250) with NaN": nan(rng.normal(0, 2, (168, 250))),
    }
    funcs = {
        "current": ORIG_TRANSFORM_NUMPY,
        "PLAN": plan_literal,
        "PROTO": fast_transform_numpy,
        "PROPOSED": proposed,
        "PROPOSED-B": proposed_isinf_first,
        "alt A": current_assume_finite,
    }
    print("\n===== Cost per call of transform_numpy, in microseconds (best of 7)")
    print("alt A: the current path inside sklearn.config_context(assume_finite=True, "
          "skip_parameter_validation=True)")
    print(f"{'input':30s} {'direction':9s} " + " ".join(f"{k:>12s}" for k in funcs))
    for name, arr in arrays.items():
        for inverse in ([True] if arr.ndim == 2 else [False, True]):
            cells = []
            for fn in funcs.values():
                cells.append(per_call(lambda: fn(arr, sc, fit=False, inverse_transform=inverse)))
            print(f"{name:30s} {'inverse' if inverse else 'transform':9s} "
                  + " ".join(f"{c:12.2f}" for c in cells))
    print("\nScaler fitted with feature names (ForecasterRecursive, ForecasterDirect, "
          "ForecasterStats, ForecasterRnn), 1D n=48:")
    arr = arrays["1D n=48"]
    for inverse in (False, True):
        cells = [per_call(lambda: fn(arr, sc_names, fit=False, inverse_transform=inverse))
                 for fn in funcs.values()]
        print(f"{'':30s} {'inverse' if inverse else 'transform':9s} "
              + " ".join(f"{c:12.2f}" for c in cells))

    print("\n===== Cost of the inf check alone, in microseconds (best of 7)")
    checks = {
        "decided: not isfinite(out.sum()) and isinf(array).any()":
            lambda a, o: (not math.isfinite(o.sum())) and np.isinf(a).any(),
        "decided + np.errstate(over='ignore')": None,
        "isinf(array).any() before computing": lambda a, o: np.isinf(a).any(),
        "previous plan: isfinite(out[~isnan(out)]).all()":
            lambda a, o: np.isfinite(o[~np.isnan(o)]).all(),
    }

    def with_errstate(a, o):
        with np.errstate(over="ignore"):
            total = o.sum()
        return (not math.isfinite(total)) and np.isinf(a).any()

    checks["decided + np.errstate(over='ignore')"] = with_errstate
    print(f"{'input':32s} " + " | ".join(checks))
    for name, arr in arrays.items():
        out = proposed(arr, sc, inverse_transform=True)
        cells = [per_call(lambda: fn(arr, out)) for fn in checks.values()]
        print(f"{name:32s} " + " | ".join(f"{c:8.2f}" for c in cells))


# ----------------------------------------------------------------------- macro


def interleaved(workflows, variants, rounds):
    """
    Time each (workflow, variant) with the variants interleaved in every round,
    so that the drift of the machine affects all of them. Returns the median,
    the minimum and the quartiles of the ratio current / variant paired by
    round (the first round is a warm-up).
    """
    times = {(w, v): [] for w in workflows for v in variants}
    for _ in range(rounds + 1):
        for w, fn in workflows.items():
            for v, patch in variants.items():
                gc.collect()
                with patched(patch):
                    t0 = time.perf_counter()
                    fn()
                    times[(w, v)].append(time.perf_counter() - t0)
    res = {}
    for (w, v), t in times.items():
        t = np.array(t[1:])
        ratios = np.array(times[(w, "current")][1:]) / t
        res[(w, v)] = (float(np.median(t)), float(t.min()),
                       tuple(float(q) for q in np.percentile(ratios, [25, 50, 75])))
    return res


class Timed:
    """Time spent inside `transform_numpy` and number of calls."""

    def __init__(self, func):
        self.func, self.total, self.calls = func, 0.0, 0

    def __call__(self, *args, **kw):
        t0 = time.perf_counter()
        try:
            return self.func(*args, **kw)
        finally:
            self.total += time.perf_counter() - t0
            self.calls += 1


def report(res, workflows, variants, width=26):
    """Median in ms, ratio of the medians and paired ratio (median and quartiles)."""
    print("cell: median ms | ratio of medians | paired ratio per round: median [q25, q75] | min ms")
    for w in workflows:
        print(w)
        base = res[(w, "current")][0]
        for v in variants:
            med, mn, (q25, q50, q75) = res[(w, v)]
            print(f"   {v:9s} {med * 1e3:9.2f} ms | x{base / med:.2f} | "
                  f"x{q50:.2f} [{q25:.2f}, {q75:.2f}] | min {mn * 1e3:.2f} ms")


VARIANTS = {"current": ORIG_TRANSFORM_NUMPY, "PLAN": plan_literal,
            "PROTO": fast_transform_numpy, "PROPOSED": proposed}


def macro(size, est, rounds):
    from data import SIZES, env_report, make_data, make_forecaster, split_train_future

    print(env_report())
    s, e = make_data(**SIZES[size])
    s_tr, e_tr, e_fut = split_train_future(s, e, 168)
    f = make_forecaster(est)
    f.fit(s_tr, exog=e_tr, store_in_sample_residuals=True)
    print(f"\nForecasterRecursiveMultiSeries, size {size} ({len(s_tr)} series), {est}")

    workflows = {
        "predict_168": lambda: f.predict(steps=168, exog=e_fut),
        "predict_24": lambda: f.predict(steps=24, exog=e_fut),
        "interval_conformal_24": lambda: f.predict_interval(steps=24, exog=e_fut, method="conformal"),
        "interval_bootstrapping_24": lambda: f.predict_interval(
            steps=24, exog=e_fut, method="bootstrapping", n_boot=100),
    }
    print("\n-- Time inside transform_numpy (upper bound of what any shortcut can save)")
    for w, fn in workflows.items():
        fn()
        for label, func in (("current", ORIG_TRANSFORM_NUMPY), ("PROPOSED", proposed)):
            best = None
            for _ in range(3):
                timed = Timed(func)
                gc.collect()
                with patched(timed):
                    t0 = time.perf_counter()
                    fn()
                    total = time.perf_counter() - t0
                if best is None or total < best[0]:
                    best = (total, timed.total, timed.calls)
            print(f"{w:26s} {label:9s} total {best[0] * 1e3:8.1f} ms, inside transform_numpy "
                  f"{best[1] * 1e3:7.2f} ms ({best[1] / best[0]:5.1%}) in {best[2]} calls "
                  f"({best[1] / best[2] * 1e6:.1f} us per call)")

    print("\n-- Calls in which each shortcut applies (predict_168)")
    for label, func, applies in (("PLAN", plan_literal, plan_applies),
                                 ("PROTO", fast_transform_numpy, prototype_conditions),
                                 ("PROPOSED", proposed, proposed_applies)):
        counted = Counted(func, applies)
        with patched(counted):
            workflows["predict_168"]()
        print(f"{label:9s} {counted.hits} of {counted.calls} calls")

    print(f"\n-- End to end, variants interleaved, {rounds} rounds")
    res = interleaved(workflows, VARIANTS, rounds)
    report(res, workflows, VARIANTS)


class TimedSplit:
    """Time inside `transform_numpy`, split by whether the shortcut would apply."""

    def __init__(self):
        self.eligible = [0.0, 0]
        self.rest = [0.0, 0]

    def __call__(self, array, transformer, fit=False, inverse_transform=False,
                 force_single_column=False):
        bucket = (self.eligible if transformer is not None and proposed_applies(
            array, transformer, fit, inverse_transform) else self.rest)
        t0 = time.perf_counter()
        try:
            return ORIG_TRANSFORM_NUMPY(
                array=array, transformer=transformer, fit=fit,
                inverse_transform=inverse_transform, force_single_column=force_single_column)
        finally:
            bucket[0] += time.perf_counter() - t0
            bucket[1] += 1


def backtesting(size, est):
    """Share of the backtesting spent in calls that the shortcut can replace."""
    from data import SIZES, env_report, make_data, make_forecaster

    from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries

    print(env_report())
    s, e = make_data(**SIZES[size])
    n_span = max(len(v) for v in s.values())
    print(f"\nbacktesting_forecaster_multiseries, size {size}, {est} (same call as bench.py)")
    for refit in (False, True):
        timed = TimedSplit()
        cv = TimeSeriesFold(steps=24, initial_train_size=n_span - 168, refit=refit, verbose=False)
        gc.collect()
        with patched(timed):
            t0 = time.perf_counter()
            backtesting_forecaster_multiseries(
                make_forecaster(est), series=s, exog=e, cv=cv, metric="mean_absolute_error",
                n_jobs=1, show_progress=False, suppress_warnings=True)
            total = time.perf_counter() - t0
        el, rest = timed.eligible, timed.rest
        print(f"refit={refit!s:5s} total {total:6.2f} s | calls the shortcut can replace: "
              f"{el[1]} calls, {el[0]:.3f} s ({el[0] / total:.1%} of the total, upper bound of "
              f"the gain x{total / (total - el[0]):.3f}) | other calls (fit=True, None): "
              f"{rest[1]} calls, {rest[0]:.3f} s")


def others(rounds):
    """Forecasters whose scaler is fitted with a pandas object (feature names)."""
    from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
    from skforecast.model_selection import (
        TimeSeriesFold,
        backtesting_forecaster,
        backtesting_forecaster_multiseries,
    )
    from skforecast.recursive import ForecasterRecursive

    print(env_line())
    rng = np.random.default_rng(1)
    n = 1500
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    y = pd.Series(50 + rng.normal(0, 1, n).cumsum(), index=idx, name="y")
    series = pd.DataFrame({f"s{i}": 10 * i + rng.normal(0, 1, n).cumsum() for i in range(3)}, index=idx)

    fr = ForecasterRecursive(Ridge(), lags=24, transformer_y=StandardScaler())
    fr.fit(y)
    fd = ForecasterDirect(Ridge(), lags=24, steps=5, transformer_y=StandardScaler())
    fd.fit(y)
    fm = ForecasterDirectMultiVariate(Ridge(), level="s0", lags=24, steps=5,
                                      transformer_series=StandardScaler())
    fm.fit(series)
    cv = TimeSeriesFold(steps=1, initial_train_size=500, refit=False, verbose=False)
    cv5 = TimeSeriesFold(steps=5, initial_train_size=500, refit=False, verbose=False)

    def bt(forecaster, cv_):
        return lambda: backtesting_forecaster(
            forecaster, y=y, cv=cv_, metric="mean_absolute_error", show_progress=False, n_jobs=1)

    workflows = {
        "Recursive predict(24)": lambda: fr.predict(steps=24),
        "Recursive backtesting 1000 folds": bt(fr, cv),
        "Direct predict(5)": lambda: fd.predict(),
        "Direct backtesting 200 folds": bt(fd, cv5),
        "DirectMultiVariate backtesting 200 folds": lambda: backtesting_forecaster_multiseries(
            fm, series=series, cv=cv5, metric="mean_absolute_error", show_progress=False,
            n_jobs=1, suppress_warnings=True),
    }
    try:
        from skforecast.recursive import ForecasterStats
        from skforecast.stats import Arima

        fs = ForecasterStats(estimator=Arima(order=(1, 0, 0)), transformer_y=StandardScaler())
        fs.fit(y)
        workflows["Stats predict(24)"] = lambda: fs.predict(steps=24)
    except Exception as exc:
        print(f"[skip] ForecasterStats: {exc}")

    print("\n-- Calls in which each shortcut applies")
    for w, fn in workflows.items():
        row = []
        for label, func, applies in (("PLAN", plan_literal, plan_applies),
                                     ("PROTO", fast_transform_numpy, prototype_conditions),
                                     ("PROPOSED", proposed, proposed_applies)):
            counted = Counted(func, applies)
            with patched(counted):
                fn()
            row.append(f"{label} {counted.hits}/{counted.calls}")
        print(f"{w:42s} " + ", ".join(row))

    print(f"\n-- End to end, variants interleaved, {rounds} rounds")
    res = interleaved(workflows, VARIANTS, rounds)
    report(res, workflows, VARIANTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["micro", "macro", "others", "backtesting"])
    ap.add_argument("--size", default="M")
    ap.add_argument("--est", default="ridge")
    ap.add_argument("--rounds", type=int, default=7)
    args = ap.parse_args()
    if args.mode == "micro":
        print(env_line())
        micro()
    elif args.mode == "macro":
        macro(args.size, args.est, args.rounds)
    elif args.mode == "backtesting":
        backtesting(args.size, args.est)
    else:
        others(args.rounds)


if __name__ == "__main__":
    main()
