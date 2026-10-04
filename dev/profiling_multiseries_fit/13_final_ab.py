"""
Final A/B of the `ForecasterRecursiveMultiSeries.fit()` optimizations, measured on the
real code: the module before the changes (`--base`, default `082aa0966`) against the
working tree, in the same process.

The "before" module is read with `git show <base>:<module>` and executed in its own
namespace with `__package__ = "skforecast.recursive"`, so it uses the current
`skforecast` utilities. This is exact only if nothing else used by `fit` changed between
`<base>` and the working tree; the script checks it with `git diff` and stops otherwise.

For every case:

1. Identity: both forecasters are fitted on the same data and compared (`X_train`,
   `y_train`, predictions with future exog, `X_train_series_names_in_`,
   `binner_intervals_`). The only accepted difference is the dtype of the one-hot and
   calendar columns (int64 before, float64 after; same values).
2. `_create_train_X_y` and `fit()` timed with `common.ab_interleaved` (warm-up, then
   before/after alternated, order swapped every round). Median and minimum are reported:
   on this machine `fit()` ratios with the same code on both sides range from 0.88 to
   1.14, so `fit()` cannot resolve effects under about 5%; component timings are stable.
3. Peak memory of `create_train_X_y` with `tracemalloc`.

Run from the repository root (about 15 minutes):

    $env:PYTHONIOENCODING = "utf-8"
    C:\\Users\\Joaquin\\miniconda3\\envs\\skforecast_24_py13\\python.exe dev\\profiling_multiseries_fit\\13_final_ab.py

Results: printed as a Markdown table and saved to `results/final_ab.json`.
"""
import argparse
import gc
import subprocess
import tracemalloc

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from common import (
    REPO_ROOT, ab_interleaved, future_exog, make_data, print_env, save_json,
)
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries as After

MODULE = "skforecast/recursive/_forecaster_recursive_multiseries.py"
STEPS = 10

# name, n_series, exog scenario, forecaster options, time _create_train_X_y, fit reps
CASES = [
    ("A (no exog)", 500, "none", {}, True, 5),
    ("B (10 float exog)", 500, "numeric", {}, True, 5),
    ("C (5 float + 5 category exog)", 500, "mixed", {}, True, 5),
    ("A, encoding=None", 500, "none", {"encoding": None}, True, 5),
    ("A, calendar_features (default, 20 columns)", 500, "none", {"calendar": True}, True, 5),
    ("A, encoding='onehot', 300 series", 300, "none", {"encoding": "onehot"}, True, 3),
    ("A, series_weights (250 of 500 series)", 500, "none", {"weights": True}, False, 3),
    ("A, LightGBM 100 trees", 500, "none", {"n_estimators": 100}, False, 5),
    ("A, Ridge", 500, "none", {"ridge": True}, False, 5),
]


def load_before(base: str):
    """Class of the module at `base`, run against the current utilities."""
    changed = subprocess.run(
        ["git", "diff", "--name-only", base, "--", "skforecast",
         f":!{MODULE}", ":!skforecast/feature_selection", ":!**/tests/**"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    if changed:
        raise SystemExit(
            f"Files used by fit() changed since {base}, the baseline would not be exact: "
            f"{changed}"
        )
    source = subprocess.run(
        ["git", "show", f"{base}:{MODULE}"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    namespace = {
        "__name__": "skforecast.recursive._before", "__package__": "skforecast.recursive"
    }
    exec(compile(source, f"{base}:{MODULE}", "exec"), namespace)
    return namespace["ForecasterRecursiveMultiSeries"]


def build(cls, names, n_estimators=25, ridge=False, calendar=False, weights=False,
          encoding="ordinal"):
    estimator = (
        Ridge()
        if ridge
        else LGBMRegressor(n_estimators=n_estimators, random_state=123, verbose=-1, n_jobs=4)
    )
    return cls(
        estimator=estimator,
        lags=24,
        window_features=RollingFeatures(stats=["mean", "std"], window_sizes=[7, 28]),
        encoding=encoding,
        calendar_features=CalendarFeatures() if calendar else None,
        series_weights={k: 2.0 for k in names[::2]} if weights else None,
    )


def assert_identical(fa, fb, series, exog):
    """Fitted state and predictions are identical; one-hot and calendar columns may
    change from int64 to float64 (same values)."""
    Xa, ya = fa.create_train_X_y(series=series, exog=exog)
    Xb, yb = fb.create_train_X_y(series=series, exog=exog)
    retyped = [
        c for c in Xa.columns
        if Xa[c].dtype != Xb[c].dtype
        and pd.api.types.is_integer_dtype(Xa[c].dtype) and Xb[c].dtype == np.float64
    ]
    pd.testing.assert_frame_equal(Xa.astype({c: float for c in retyped}), Xb, check_exact=True)
    pd.testing.assert_series_equal(ya, yb, check_exact=True)
    ex = future_exog(exog, STEPS)
    pd.testing.assert_frame_equal(
        fa.predict(steps=STEPS, exog=ex), fb.predict(steps=STEPS, exog=ex), check_exact=True
    )
    assert fa.X_train_series_names_in_ == fb.X_train_series_names_in_
    assert fa.binner_intervals_ == fb.binner_intervals_
    return retyped


def peak_mb(fn) -> float:
    gc.collect()
    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 2**20


def ratio(before: dict, after: dict) -> str:
    return f"{after['median'] / before['median']:.2f} / {after['min'] / before['min']:.2f}"


def seconds(t: dict) -> str:
    return f"{t['median']:.2f} / {t['min']:.2f} s"


def run_case(Before, name, n_series, scenario, options, time_ctx, reps_fit, reps_ctx):
    data = make_data(n_series=n_series, n_obs=2000, exog=scenario)
    series, exog, names = data["series_dict"], data["exog_dict"], data["names"]
    result = {"case": name, "n_series": n_series, "exog": scenario, "options": options}

    fa, fb = build(Before, names, **options), build(After, names, **options)
    fa.fit(series=series, exog=exog)
    fb.fit(series=series, exog=exog)
    result["retyped_columns"] = len(assert_identical(fa, fb, series, exog))
    fa = fb = None
    gc.collect()

    if time_ctx:
        fa, fb = build(Before, names, **options), build(After, names, **options)
        ab = ab_interleaved(
            lambda: fa._create_train_X_y(series=series, exog=exog),
            lambda: fb._create_train_X_y(series=series, exog=exog),
            reps=reps_ctx, label_a="before", label_b="after",
        )
        result["create_train_X_y"] = ab
        result["peak_mb"] = {
            "before": peak_mb(lambda: fa.create_train_X_y(series=series, exog=exog)),
            "after": peak_mb(lambda: fb.create_train_X_y(series=series, exog=exog)),
        }
        fa = fb = None
        gc.collect()

    fa, fb = build(Before, names, **options), build(After, names, **options)
    result["fit"] = ab_interleaved(
        lambda: fa.fit(series=series, exog=exog),
        lambda: fb.fit(series=series, exog=exog),
        reps=reps_fit, label_a="before", label_b="after",
    )
    return result


def markdown(results: list) -> str:
    lines = [
        "| Case | `create_train_X_y` before | after | after / before | `fit()` before "
        "| after | after / before | Peak `create_train_X_y` |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ctx = r.get("create_train_X_y")
        fit = r["fit"]
        peak = r.get("peak_mb")
        if ctx:
            ctx_cols = (
                f"{seconds(ctx['before'])} | {seconds(ctx['after'])} "
                f"| {ratio(ctx['before'], ctx['after'])}"
            )
            peak_col = f"{peak['before']:.0f} -> {peak['after']:.0f} MB"
        else:
            ctx_cols, peak_col = "= | = | ", ""
        lines.append(
            f"| {r['case']} | {ctx_cols} "
            f"| {seconds(fit['before'])} | {seconds(fit['after'])} "
            f"| {ratio(fit['before'], fit['after'])} | {peak_col} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="082aa0966")
    ap.add_argument("--reps", type=int, default=7, help="reps of _create_train_X_y")
    ap.add_argument("--cases", nargs="*", type=int, help="indexes of CASES to run")
    args = ap.parse_args()

    env = print_env()
    Before = load_before(args.base)
    print(f"before: {args.base}:{MODULE}\nafter: working tree\n")
    selected = args.cases if args.cases else range(len(CASES))
    results = []
    for i in selected:
        name, n_series, scenario, options, time_ctx, reps_fit = CASES[i]
        print(f"[{i}] {name} ...", flush=True)
        r = run_case(Before, name, n_series, scenario, options, time_ctx, reps_fit, args.reps)
        results.append(r)
        print(markdown([r]).splitlines()[-1], flush=True)
    print("\n" + markdown(results))
    path = save_json({"env": env, "base": args.base, "results": results}, "final_ab.json")
    print(f"\nsaved {path}")
