"""
Cost of refitting an already fitted ForecasterDirect / ForecasterDirectMultiVariate
with n_jobs > 1.

`fit` sends `forecaster=self` to every step task of `Parallel`, and the
fitted `estimators_` of the previous fit are not cleared before, so every
task pickles all the fitted estimators. This script times a first fit and
later refits of the same forecaster, with the current code ("current") and
with `estimators_` emptied before the refit ("cleared", the proposed fix,
emulated from outside the package).

Usage: python exp_direct_refit.py
Output: results/exp_direct_refit.csv
"""

from __future__ import annotations

import pickle
import sys
import time

from common import RESULTS_DIR, CsvLog, check_repo_skforecast

FIELDS = [
    "forecaster",
    "estimator",
    "n_jobs",
    "variant",
    "call",
    "seconds",
    "pickle_mb_before_fit",
]


def main() -> int:
    check_repo_skforecast()
    from cases import DIRECT, DMV, make_estimator, multi_data, single_data
    from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate

    log = CsvLog(RESULTS_DIR / "exp_direct_refit.csv", FIELDS)
    cfg = DIRECT["L"]
    y, exog = single_data(cfg["n"])
    y, exog = y.iloc[: cfg["initial"]], exog.iloc[: cfg["initial"]]
    cfg_m = DMV["L"]
    series, exog_m = multi_data(cfg_m["n_series"], cfg_m["n"])
    series, exog_m = series.iloc[: cfg_m["initial"]], exog_m.iloc[: cfg_m["initial"]]

    for kind in ["direct", "dmv"]:
        for est in ["lgbm_1", "hgb", "ridge"]:
            for n_jobs in [1, 21]:
                for variant in ["current", "cleared"]:
                    if kind == "direct":
                        f = ForecasterDirect(
                            estimator=make_estimator(est),
                            steps=cfg["steps"],
                            lags=cfg["lags"],
                            n_jobs=n_jobs,
                        )
                        fit = lambda: f.fit(y=y, exog=exog)  # noqa: E731
                    else:
                        f = ForecasterDirectMultiVariate(
                            estimator=make_estimator(est),
                            level=series.columns[0],
                            steps=cfg_m["steps"],
                            lags=cfg_m["lags"],
                            n_jobs=n_jobs,
                        )
                        fit = lambda: f.fit(series=series, exog=exog_m)  # noqa: E731
                    # call 0 warms the executor, call 1 is the first fit timed
                    # with workers alive, calls 2+ are refits.
                    for call in range(6):
                        if variant == "cleared" and call >= 2:
                            f.estimators_ = {}
                        size = len(pickle.dumps(f)) / 1e6
                        t0 = time.perf_counter()
                        fit()
                        seconds = time.perf_counter() - t0
                        if call == 0:
                            # Reset to an unfitted forecaster for call 1.
                            f.estimators_ = {}
                        log.write(
                            {
                                "forecaster": kind,
                                "estimator": est,
                                "n_jobs": n_jobs,
                                "variant": variant,
                                "call": call,
                                "seconds": seconds,
                                "pickle_mb_before_fit": round(size, 2),
                            }
                        )
                        print(
                            f"{kind} {est} n_jobs={n_jobs} {variant} call={call}: "
                            f"{seconds:.2f} s (pickle {size:.1f} MB)",
                            flush=True,
                        )
    log.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
