"""
PR 2c (BUG-search-short-series, section 15): warnings of the joblib worker
processes in the backtesting and the hyperparameter searches.

How many warnings (rich panels of skforecast and plain warnings of other
libraries) does a user see with `n_jobs='auto'` and with `n_jobs=1`, and how
many of them come from the worker processes of joblib?

Each case runs in a clean Python process (`--case NAME --n-jobs auto|1`),
inside its own `warnings.catch_warnings()` with the default filters of Python.
The warnings that reach the main process are counted by wrapping
`warnings.showwarning`; the warnings shown in total are counted from the
output of the process (stdout and stderr, which the workers of loky inherit).
Workers = total - main.

Usage:
    python PR-2c_joblib_warnings.py                 # every case, sequentially
    python PR-2c_joblib_warnings.py --only S1 B2    # some cases
    python PR-2c_joblib_warnings.py --case B1 --n-jobs auto   # one run, raw output
    python PR-2c_joblib_warnings.py --bench 0.27.x > /dev/null  # cost, to stderr

Run it from the root of the repository. To measure a prototype, put its
worktree first: `PYTHONPATH=<worktree> python <this script>`.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

MARKER = "@@MAIN_WARNINGS@@"


# --------------------------------------------------------------------------- #
# Cases (run inside the child process)
# --------------------------------------------------------------------------- #
def build_cases(n_jobs):
    import numpy as np
    import pandas as pd
    import optuna
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from skforecast.recursive import (
        ForecasterRecursive, ForecasterRecursiveMultiSeries, ForecasterStats
    )
    from skforecast.stats import Arima
    from skforecast.model_selection import (
        TimeSeriesFold, OneStepAheadFold,
        backtesting_forecaster, backtesting_forecaster_multiseries, backtesting_stats,
        grid_search_forecaster, random_search_forecaster, bayesian_search_forecaster,
        grid_search_forecaster_multiseries, random_search_forecaster_multiseries,
        bayesian_search_forecaster_multiseries,
    )
    from skforecast.exceptions import MissingValuesWarning

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Data of notebook 08: two stores and a new product (29 observations in the
    # first training set).
    rng = np.random.default_rng(42)
    index = pd.date_range("2023-01-01", periods=365, freq="D")

    def sales(n, level):
        t = np.arange(n)
        return (
            level + 10 * np.sin(2 * np.pi * t / 7)
            + 0.3 * rng.normal(0, 3, n).cumsum() + rng.normal(0, 2, n)
        )

    series = {
        "store_1": pd.Series(sales(365, 100), index=index),
        "store_2": pd.Series(sales(365, 150), index=index),
        "new_product": pd.Series(sales(120, 50), index=index[-120:]),
    }
    stores = {k: v for k, v in series.items() if k != "new_product"}
    exog = pd.DataFrame(
        {"promo": rng.integers(0, 2, len(index)).astype(float)}, index=index
    )
    exog_gap = exog.copy()
    exog_gap.loc["2023-10-20", "promo"] = np.nan        # gap inside the test set

    # QW-13: gaps in `y` inside the training set of every fold
    y_gaps = series["store_1"].copy()
    y_gaps.loc[["2023-03-10", "2023-06-15", "2023-08-01"]] = np.nan
    stores_gaps = {"store_1": y_gaps, "store_2": series["store_2"]}

    cv = TimeSeriesFold(steps=14, initial_train_size="2023-09-30", refit=False)
    cv_refit = TimeSeriesFold(steps=14, initial_train_size="2023-09-30", refit=True)
    cv_osa = OneStepAheadFold(initial_train_size="2023-09-30")

    def hgbr():
        return HistGradientBoostingRegressor(max_iter=20, random_state=0)

    def f_single():
        return ForecasterRecursive(hgbr(), lags=7)

    def f_multi():
        return ForecasterRecursiveMultiSeries(hgbr(), lags=7)

    def f_multi_ridge():
        return ForecasterRecursiveMultiSeries(
            Ridge(), lags=7, transformer_series=StandardScaler()
        )

    common = dict(metric="mean_absolute_error", show_progress=False, n_jobs=n_jobs)
    search = dict(return_best=False, **common)

    def space_single(trial):
        return {
            "lags": trial.suggest_categorical("lags", [7, 14]),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
        }

    def space_multi(trial):
        return {
            "lags": trial.suggest_categorical("lags", [7, 35]),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
        }

    def space_ridge(trial):
        return {
            "lags": trial.suggest_categorical("lags", [7, 14]),
            "alpha": trial.suggest_float("alpha", 0.01, 10, log=True),
        }

    grid = {"learning_rate": [0.1, 0.2]}

    def user_filter_case():
        # The user silences MissingValuesWarning in their own code
        import warnings
        warnings.simplefilter("ignore", category=MissingValuesWarning)
        backtesting_forecaster_multiseries(
            f_multi(), series=stores, exog=exog_gap, cv=cv, **common
        )

    cases = {
        # ---------------- backtesting ----------------
        "B1": ("backtesting_forecaster, HGBR, gap in exog (test), refit=False",
               lambda: backtesting_forecaster(
                   f_single(), y=series["store_1"], exog=exog_gap, cv=cv, **common)),
        "B2": ("backtesting_forecaster, HGBR, gaps in y (QW-13), refit=True",
               lambda: backtesting_forecaster(
                   f_single(), y=y_gaps, cv=cv_refit, **common)),
        "B3": ("backtesting_forecaster, MLP max_iter=5 (ConvergenceWarning), refit=True",
               lambda: backtesting_forecaster(
                   ForecasterRecursive(MLPRegressor(max_iter=5, random_state=0), lags=7),
                   y=series["store_1"], cv=cv_refit, **common)),
        "B4": ("backtesting_multiseries, HGBR, short series (#1415) + gap in exog, refit=False",
               lambda: backtesting_forecaster_multiseries(
                   ForecasterRecursiveMultiSeries(hgbr(), lags=35),
                   series=series, exog=exog_gap, cv=cv, **common)),
        "B5": ("backtesting_multiseries, HGBR, gaps in y (QW-13), refit=True",
               lambda: backtesting_forecaster_multiseries(
                   f_multi(), series=stores_gaps, cv=cv_refit, **common)),
        "B6": ("backtesting_multiseries, HGBR, gap in exog, user filter ignore MissingValuesWarning",
               user_filter_case),
        "B7": ("backtesting_multiseries, HGBR, gap in exog, suppress_warnings=True",
               lambda: backtesting_forecaster_multiseries(
                   f_multi(), series=stores, exog=exog_gap, cv=cv,
                   suppress_warnings=True, **common)),
        "B8": ("backtesting_stats, Arima(1,0,0), refit=True",
               lambda: backtesting_stats(
                   ForecasterStats(Arima(order=(1, 0, 0))), y=series["store_1"],
                   cv=TimeSeriesFold(steps=14, initial_train_size="2023-11-30", refit=True),
                   **common)),
        # ---------------- single series searches ----------------
        "S1": ("grid single TSF, HGBR, gap in exog, 2 lags x 2 lr",
               lambda: grid_search_forecaster(
                   f_single(), y=series["store_1"], exog=exog_gap, cv=cv,
                   param_grid=grid, lags_grid=[7, 14], **search)),
        "S2": ("grid single TSF, HGBR, gaps in y (QW-13), refit=True, 2 lags x 2 lr",
               lambda: grid_search_forecaster(
                   f_single(), y=y_gaps, cv=cv_refit,
                   param_grid=grid, lags_grid=[7, 14], **search)),
        "S3": ("random single TSF, HGBR, gap in exog, 4 iters",
               lambda: random_search_forecaster(
                   f_single(), y=series["store_1"], exog=exog_gap, cv=cv,
                   param_distributions={"learning_rate": [0.05, 0.1, 0.2, 0.3]},
                   lags_grid=[7, 14], n_iter=2, random_state=1, **search)),
        "S4": ("bayesian single TSF, HGBR, gap in exog, 6 trials",
               lambda: bayesian_search_forecaster(
                   f_single(), y=series["store_1"], exog=exog_gap, cv=cv,
                   search_space=space_single, n_trials=6, random_state=123, **search)),
        "S5": ("grid single OSA, HGBR, gap in exog, 2 lags x 2 lr",
               lambda: grid_search_forecaster(
                   f_single(), y=series["store_1"], exog=exog_gap, cv=cv_osa,
                   param_grid=grid, lags_grid=[7, 14], **search)),
        "S6": ("bayesian single OSA, HGBR, gaps in y (QW-13), 6 trials",
               lambda: bayesian_search_forecaster(
                   f_single(), y=y_gaps, cv=cv_osa,
                   search_space=space_single, n_trials=6, random_state=123, **search)),
        # ---------------- multi-series searches ----------------
        "M1": ("grid multi TSF, HGBR, short series + gap in exog, lags [7, 35] x 2 lr",
               lambda: grid_search_forecaster_multiseries(
                   f_multi(), series=series, exog=exog_gap, cv=cv,
                   param_grid=grid, lags_grid=[7, 35], **search)),
        "M2": ("random multi TSF, HGBR, short series + gap in exog, 4 iters",
               lambda: random_search_forecaster_multiseries(
                   f_multi(), series=series, exog=exog_gap, cv=cv,
                   param_distributions={"learning_rate": [0.05, 0.1, 0.2, 0.3]},
                   lags_grid=[7, 35], n_iter=2, random_state=1, **search)),
        "M3": ("bayesian multi TSF, HGBR, short series + gap in exog, 6 trials",
               lambda: bayesian_search_forecaster_multiseries(
                   f_multi(), series=series, exog=exog_gap, cv=cv,
                   search_space=space_multi, n_trials=6, random_state=123, **search)),
        "M4": ("grid multi TSF, HGBR, gaps in y (QW-13), refit=True, 2 lags x 2 lr",
               lambda: grid_search_forecaster_multiseries(
                   f_multi(), series=stores_gaps, cv=cv_refit,
                   param_grid=grid, lags_grid=[7, 14], **search)),
        "M5": ("grid multi OSA, HGBR, short series + gap in exog, lags [7, 35] x 2 lr",
               lambda: grid_search_forecaster_multiseries(
                   f_multi(), series=series, exog=exog_gap, cv=cv_osa,
                   param_grid=grid, lags_grid=[7, 35], **search)),
        "M6": ("bayesian multi OSA, HGBR, short series + gap in exog, 6 trials",
               lambda: bayesian_search_forecaster_multiseries(
                   f_multi(), series=series, exog=exog_gap, cv=cv_osa,
                   search_space=space_multi, n_trials=6, random_state=123, **search)),
        # ---------------- scenario 3 of notebook 08 ----------------
        "N3": ("notebook 08 sc. 3: bayesian multi TSF, Ridge + gap in exog, 10 trials",
               lambda: bayesian_search_forecaster_multiseries(
                   f_multi_ridge(), series=stores, exog=exog_gap, cv=cv,
                   search_space=space_ridge, n_trials=10, random_state=123, **search)),
    }

    return cases


def run_case(name, n_jobs):
    import warnings
    import skforecast  # noqa: F401 (installs the rich handler before wrapping it)
    import skforecast.exceptions  # noqa: F401

    n_jobs = n_jobs if n_jobs == "auto" else int(n_jobs)
    _, fn = build_cases(n_jobs)[name]
    main = Counter()
    error = ""
    with warnings.catch_warnings():
        original = warnings.showwarning

        def counting_showwarning(message, category, *args, **kwargs):
            main[category.__name__] += 1
            original(message, category, *args, **kwargs)

        warnings.showwarning = counting_showwarning
        start = time.perf_counter()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {str(e)[:150]}"
        elapsed = time.perf_counter() - start
    sys.stdout.flush()
    sys.stderr.flush()
    print(f"\n{MARKER}" + json.dumps(
        {"main": dict(main), "error": error, "seconds": round(elapsed, 2)}
    ), flush=True)


# --------------------------------------------------------------------------- #
# Driver (parent process)
# --------------------------------------------------------------------------- #
PANEL_TITLE = re.compile(r"╭─+ (\w+) ─")
PLAIN_WARNING = re.compile(r":\d+: (\w*(?:Warning|Error)): ", re.MULTILINE)


def parse_output(output):
    marker_line = [line for line in output.splitlines() if line.startswith(MARKER)]
    info = json.loads(marker_line[-1][len(MARKER):]) if marker_line else {
        "main": {}, "error": "no marker (crash?)", "seconds": None}
    shown = Counter(PANEL_TITLE.findall(output)) + Counter(PLAIN_WARNING.findall(output))
    main = Counter(info["main"])
    workers = shown - main
    return {
        "shown": dict(shown), "main": dict(main), "workers": dict(workers),
        "n_shown": sum(shown.values()), "n_main": sum(main.values()),
        "n_workers": sum(workers.values()), "replacement_chars": output.count("�"),
        "error": info["error"], "seconds": info["seconds"],
    }


def fmt(counter):
    return ", ".join(f"{k} x{v}" for k, v in sorted(counter.items())) or "-"


def main_driver(only):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    import skforecast
    print(f"skforecast from {os.path.dirname(skforecast.__file__)}\n", flush=True)
    names = only or list(build_cases("auto").keys())
    descriptions = {k: v[0] for k, v in build_cases("auto").items()}
    rows = []
    for name in names:
        for n_jobs in ["auto", "1"]:
            proc = subprocess.run(
                [sys.executable, __file__, "--case", name, "--n-jobs", n_jobs],
                capture_output=True, text=True, env=env, timeout=900,
            )
            r = parse_output(proc.stdout + proc.stderr)
            r.update(case=name, n_jobs=n_jobs)
            rows.append(r)
            print(
                f"{name:3} n_jobs={n_jobs:4} shown={r['n_shown']:3} main={r['n_main']:3} "
                f"workers={r['n_workers']:3} ({r['seconds']} s) | {descriptions[name]}\n"
                f"      main: {fmt(r['main'])}\n      workers: {fmt(r['workers'])}"
                + (f"\n      ERROR: {r['error']}" if r["error"] else ""),
                flush=True,
            )

    print("\n| Case | Description | `auto` shown (workers) | `1` shown |")
    print("|---|---|---|---|")
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], {})[r["n_jobs"]] = r
    for name, r in by_case.items():
        print(f"| {name} | {descriptions[name]} | {r['auto']['n_shown']} "
              f"({r['auto']['n_workers']}) | {r['1']['n_shown']} |")


# --------------------------------------------------------------------------- #
# Cost of the change: backtesting with many folds
# --------------------------------------------------------------------------- #
def bench(label, n_repeats=11):
    """
    `backtesting_forecaster` with 500 folds (refit=False, HGBR, n_jobs='auto'),
    without warnings and with 2 warnings per fold (a NaN in `exog` in the test
    set of every fold), default filters. Results go to stderr; run it with the
    stdout to /dev/null, so the panels are not printed to the terminal.
    Compare two versions with PYTHONPATH (the repository and the worktree of
    the prototype), alternating rounds.
    """
    import warnings
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from skforecast.recursive import ForecasterRecursive
    from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster

    rng = np.random.default_rng(0)
    n = 3000
    index = pd.date_range("2015-01-01", periods=n, freq="D")
    y = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=index, name="y")
    exog = pd.DataFrame({"x": rng.normal(size=n)}, index=index)
    exog_nan = exog.copy()
    exog_nan.iloc[2000::2, 0] = np.nan
    cv = TimeSeriesFold(steps=2, initial_train_size=2000, refit=False)

    def run(ex):
        forecaster = ForecasterRecursive(
            HistGradientBoostingRegressor(max_iter=20, random_state=0), lags=7
        )
        start = time.perf_counter()
        with warnings.catch_warnings():
            backtesting_forecaster(
                forecaster, y=y, exog=ex, cv=cv, metric="mean_absolute_error",
                n_jobs="auto", show_progress=False,
            )
        return time.perf_counter() - start

    run(exog)                                   # start the workers
    for case, ex in [("no warnings", exog), ("2 warnings per fold", exog_nan)]:
        times = [run(ex) for _ in range(n_repeats)]
        print(f"{label:9} | {case:20} | median {np.median(times):.3f} s | "
              f"min {min(times):.3f} s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--n-jobs", default="auto")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--bench", metavar="LABEL")
    args = parser.parse_args()
    if args.bench:
        bench(args.bench)
    elif args.case:
        run_case(args.case, args.n_jobs)
    else:
        main_driver(args.only)
