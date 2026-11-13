"""
Does sending the fitted forecaster once, already serialized in the main
thread, remove the cost per fold of running the predictions of a backtesting
without refit in parallel, and the slowdown of the fits after a parallel run?

Each (estimator, mode) runs in a new process. Modes of the prediction of 100
folds of 24 steps with a fitted forecaster:
  seq           in the main process
  obj           joblib, the forecaster object in every task (as skforecast does)
  bytes         joblib, the forecaster pickled once in the main thread, bytes in
                every task, unpickled in every task
  bytes_cached  the same, unpickled once per worker
  array_cached  the same, with the bytes as a numpy array (joblib shares arrays
                over 1 MB with the workers through a memory map)
"""
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "bench_n_jobs.py"
MODES = ["seq", "obj", "bytes", "bytes_cached", "array_cached"]


def child(est, mode):
    import warnings
    warnings.filterwarnings("ignore")
    import joblib
    import numpy as np
    from joblib import Parallel, delayed
    sys.path.insert(0, str(HERE))
    import proto_helpers as helpers

    spec = importlib.util.spec_from_file_location("bench", BENCH)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    data = bench.load_bike().loc[bench.START["S"]:]
    n_train = len(data.loc[:bench.END_TRAIN])
    y, exog = data["users"], data.drop(columns="users")

    def fit():
        forecaster = bench.make_recursive(est)
        start = time.perf_counter()
        forecaster.fit(y=y.iloc[:n_train], exog=exog.iloc[:n_train])
        return time.perf_counter() - start, forecaster

    fit_before, forecaster = fit()
    steps, window = 24, forecaster.window_size
    folds = [
        (y.iloc[i - window:i], exog.iloc[i:i + steps])
        for i in range(n_train, n_train + 100 * steps, steps)
    ]
    start = time.perf_counter()
    blob = pickle.dumps(forecaster, protocol=pickle.HIGHEST_PROTOCOL)
    dumps_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pickle.loads(blob)
    loads_seconds = time.perf_counter() - start
    n_jobs = max(1, joblib.cpu_count() - 1)

    def run(some_folds):
        if mode == "seq":
            return [
                helpers.predict_obj(forecaster, lw, ex, steps) for lw, ex in some_folds
            ]
        if mode == "obj":
            return Parallel(n_jobs=n_jobs)(
                delayed(helpers.predict_obj)(forecaster, lw, ex, steps)
                for lw, ex in some_folds
            )
        # The forecaster is serialized here, in the main thread, once per call.
        data_ = pickle.dumps(forecaster, protocol=pickle.HIGHEST_PROTOCOL)
        if mode == "bytes":
            return Parallel(n_jobs=n_jobs)(
                delayed(helpers.predict_bytes)(data_, lw, ex, steps)
                for lw, ex in some_folds
            )
        token = f"{os.getpid()}-{time.perf_counter_ns()}"
        if mode == "array_cached":
            data_ = np.frombuffer(data_, dtype=np.uint8)
        return Parallel(n_jobs=n_jobs)(
            delayed(helpers.predict_cached)(token, data_, lw, ex, steps)
            for lw, ex in some_folds
        )

    run(folds[:2 * n_jobs])          # start the workers and import the libraries
    times, results = [], None
    for _ in range(2):
        start = time.perf_counter()
        results = run(folds)
        times.append(time.perf_counter() - start)
    fit_after, _ = fit()
    print("RESULT " + json.dumps({
        "est": est, "mode": mode, "n_jobs": n_jobs, "times": times,
        "fit_before": fit_before, "fit_after": fit_after,
        "dumps_ms": dumps_seconds * 1e3, "loads_ms": loads_seconds * 1e3,
        "mb": len(blob) / 1e6, "checksum": float(np.concatenate(results).sum()),
    }))


def main():
    estimators = sys.argv[1:] or ["hgb", "lgbm_1t", "lgbm", "xgb", "catboost", "rf"]
    out = []
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    for est in estimators:
        for mode in MODES:
            p = subprocess.run([sys.executable, __file__, "child", est, mode],
                               capture_output=True, text=True, env=env, timeout=1500,
                               start_new_session=True)
            lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if not lines:
                print(f"FAILED {est} {mode}: {p.stderr[-800:]}", flush=True)
                continue
            r = json.loads(lines[-1][7:])
            out.append(r)
            print(f"{est:<9} {mode:<13} n_jobs={r['n_jobs']}"
                  f" predictions {min(r['times']):6.2f} s"
                  f" | fit before {r['fit_before']:5.2f} after {r['fit_after']:5.2f}"
                  f" ({r['fit_after'] / r['fit_before']:4.2f}x)"
                  f" | dumps {r['dumps_ms']:6.1f} ms loads {r['loads_ms']:6.1f} ms"
                  f" {r['mb']:5.1f} MB"
                  f" | checksum {r['checksum']:.3f}", flush=True)
    (HERE / "proto_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
