"""
Reproduce the failure of backtesting_stats with Sarimax and n_jobs > 1 seen
in block F (size L: BrokenProcessPool "A task has failed to un-serialize").

Runs the same backtesting with n_jobs=1 and n_jobs=4 for the S and L sizes
of block F and prints the full chained traceback of any error, including
the remote traceback of the worker.

Usage: python exp_sarimax_parallel.py
"""

from __future__ import annotations

import sys
import traceback
import warnings

warnings.filterwarnings("ignore")


def main() -> int:
    from cases import STATS, _cv, make_stats_estimator, single_data
    from skforecast.model_selection import backtesting_stats
    from skforecast.recursive import ForecasterStats

    for size in ["S", "L"]:
        cfg = STATS[size]
        y, _ = single_data(cfg["n"], freq="D", period_a=cfg["m"], period_b=4 * cfg["m"])
        for n_jobs in [1, 4]:
            forecaster = ForecasterStats(
                estimator=make_stats_estimator("sarimax", cfg["m"])
            )
            try:
                backtesting_stats(
                    forecaster=forecaster,
                    y=y,
                    cv=_cv(cfg["steps"], cfg["initial"], True),
                    metric="mean_absolute_error",
                    n_jobs=n_jobs,
                    show_progress=False,
                    suppress_warnings=True,
                )
                print(f"size={size} n_jobs={n_jobs}: ok", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"size={size} n_jobs={n_jobs}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                traceback.print_exception(exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
