"""
Current multi-series backtesting against `proto_bt_blocks.backtesting_blocks`,
with or without refit, with the metrics of today and with the metrics in numpy.
Prints the stages of both and the peak memory of each (measured in a child
process each, so the peaks do not mix).

Usage: python time_bt_blocks.py N_SERIES EXOG(none|float|mixed) EST(ridge|lgbm) N_FOLDS STEPS REFIT(0|1|k) [N_ROWS]
Run it alone on the machine (OMP_NUM_THREADS=1).
"""
import _env  # noqa: F401
import multiprocessing as mp
import resource
import sys
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from bench_data import make_data

warnings.simplefilter("ignore")
n_series = int(sys.argv[1]); exog_kind = sys.argv[2]; est = sys.argv[3]
n_folds = int(sys.argv[4]); steps = int(sys.argv[5]); refit = int(sys.argv[6])
n_rows = int(sys.argv[7]) if len(sys.argv) > 7 else 1500
refit = refit if refit > 1 else bool(refit)
REPS = 1 if refit else 2


def setup():
    from skforecast.model_selection import TimeSeriesFold
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    series, exog = make_data(n_series, exog_kind, n_rows)
    estimator = Ridge() if est == "ridge" else LGBMRegressor(
        n_estimators=50, verbose=-1, n_jobs=1, random_state=1
    )
    forecaster = ForecasterRecursiveMultiSeries(estimator, lags=24, encoding="ordinal")
    cv = TimeSeriesFold(
        steps=steps, initial_train_size=n_rows - n_folds * steps, refit=refit,
        fixed_train_size=False, verbose=False,
    )
    return dict(forecaster=forecaster, series=series, exog=exog, cv=cv,
                metric="mean_absolute_error")


def run_current(queue):
    import skforecast.model_selection._validation as val
    import skforecast.recursive._forecaster_recursive_multiseries as frm
    from skforecast.model_selection import backtesting_forecaster_multiseries
    from skforecast.recursive import ForecasterRecursiveMultiSeries as C
    kwargs = setup()
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    best = np.inf
    for _ in range(REPS):
        t = time.perf_counter()
        out = backtesting_forecaster_multiseries(**kwargs, n_jobs=1, show_progress=False)
        best = min(best, time.perf_counter() - t)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    acc = {}

    def timed(name, fn, gen=False):
        def wrapper(*a, **k):
            t = time.perf_counter()
            res = fn(*a, **k)
            if gen:
                res = iter(list(res))
            acc[name] = acc.get(name, 0.0) + time.perf_counter() - t
            return res
        return wrapper

    val.check_preprocess_series = timed("preprocess", val.check_preprocess_series)
    val.check_preprocess_exog_multiseries = timed("preprocess", val.check_preprocess_exog_multiseries)
    val._extract_data_folds_multiseries = timed("extract folds", val._extract_data_folds_multiseries, True)
    val._calculate_metrics_backtesting_multiseries = timed("metrics", val._calculate_metrics_backtesting_multiseries)
    C.fit = timed("fit", C.fit)
    C.predict = timed("predict", C.predict)
    C._recursive_predict = timed("  of which _recursive_predict", C._recursive_predict)
    t = time.perf_counter()
    backtesting_forecaster_multiseries(**kwargs, n_jobs=1, show_progress=False)
    acc["total (instrumented)"] = time.perf_counter() - t
    queue.put((best, acc, out, (peak - base) / 1024, peak / 1024))


def run_proto(queue, fast_metrics):
    from proto_bt_blocks import backtesting_blocks
    kwargs = setup()
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    best_t = None
    for _ in range(REPS):
        tm = {}
        out = backtesting_blocks(**kwargs, timings=tm, fast_metrics=fast_metrics)
        if best_t is None or tm["total"] < best_t["total"]:
            best_t = tm
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    queue.put((best_t["total"], best_t, out, (peak - base) / 1024, peak / 1024))


def in_child(target, *args):
    queue = mp.Queue()
    p = mp.Process(target=target, args=(queue, *args))
    p.start()
    res = queue.get()
    p.join()
    return res


if __name__ == "__main__":
    mp.set_start_method("spawn")
    print(f"n_series={n_series} rows={n_rows} exog={exog_kind} est={est} folds={n_folds} steps={steps} refit={refit}")
    t_cur, acc, out_cur, mem_cur, peak_cur = in_child(run_current)
    print(f"CURRENT: {t_cur * 1e3:9.1f} ms (best of {REPS}); memory over the data +{mem_cur:.0f} MB (peak {peak_cur:.0f} MB)")
    for k, v in acc.items():
        print(f"   {k:<32}{v * 1e3:10.1f} ms")
    for fast in (False, True):
        t_new, tm, out_new, mem_new, peak_new = in_child(run_proto, fast)
        label = "numpy metrics" if fast else "metrics of today"
        same = []
        for a, b in zip(out_cur, out_new):
            try:
                pd.testing.assert_frame_equal(a, b, check_exact=True)
                same.append("identical")
            except AssertionError:
                num = a.select_dtypes("number").columns
                diff = np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float))) if a.shape == b.shape else np.nan
                same.append(f"max abs diff {diff:.2g}")
        print(f"PROTOTYPE ({label}): {t_new * 1e3:9.1f} ms -> {t_cur / t_new:.2f}x the whole call; "
              f"metrics {same[0]}, predictions {same[1]}; memory +{mem_new:.0f} MB (peak {peak_new:.0f} MB)")
        for k in ("preprocess", "matrix", "exog_align", "extract_train", "fit", "prepare_model", "folds", "output", "metrics", "rest"):
            print(f"   {k:<32}{tm[k] * 1e3:10.1f} ms")
    print(_env.versions())
