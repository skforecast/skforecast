"""
Prototype test with real cases (`real_data.py`): the whole backtesting call of
today against `proto_bt_blocks.backtesting_blocks` (with the metrics of today
and with the metrics in numpy), with the same outputs. For the single series
case (no prototype), where the time goes and what P5 (no validation per fold,
`predict(check_inputs=False)`) would save.

Model time = `fit` of the forecaster + the predict function of the estimator
(`_build_predict_function`), measured with wrappers in every run. The rest is what a "validate and prepare once"
implementation can remove.

Usage: REAL_DATA_DIR=... python time_real_cases.py CASE [REPS]
Run it alone on the machine (OMP_NUM_THREADS=1, as `run3.sh` does).
"""
import _env  # noqa: F401  (first: it can block pyarrow)
import sys
import time
import warnings

import numpy as np
import pandas as pd

from real_data import CASES

case_name = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
case = CASES[case_name]()
make = case["forecaster"]
est_cls = type(make().estimator)
ACC = {"fit": 0.0, "est_predict": 0.0}


def _wrap(cls, attr, key):
    orig = getattr(cls, attr)

    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            ACC[key] += time.perf_counter() - t
    setattr(cls, attr, wrapper)


_wrap(type(make()), "fit", "fit")


def _wrap_predict_builder(module):
    # NOTE: The forecasters predict through `_build_predict_function` (a fast
    # path for LightGBM, XGBoost, trees...), not always `estimator.predict`.
    orig = module._build_predict_function

    def builder(*args, **kwargs):
        fn = orig(*args, **kwargs)

        def timed(X):
            t = time.perf_counter()
            try:
                return fn(X)
            finally:
                ACC["est_predict"] += time.perf_counter() - t
        return timed
    module._build_predict_function = builder


import skforecast.recursive._forecaster_recursive as _fr  # noqa: E402
import skforecast.recursive._forecaster_recursive_multiseries as _frm  # noqa: E402

_wrap_predict_builder(_fr)
_wrap_predict_builder(_frm)


def run(fn):
    """Best of REPS by whole call; returns (seconds, model seconds, result, warnings)."""
    best = None
    for _ in range(REPS):
        ACC.update(fit=0.0, est_predict=0.0)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            t = time.perf_counter()
            res = fn()
            total = time.perf_counter() - t
        if best is None or total < best[0]:
            best = (total, ACC["fit"], ACC["est_predict"], res,
                    [(w.category.__name__, str(w.message)) for w in rec])
    return best


def line(label, r, ref=None):
    total, fit, pred = r[:3]
    rest = total - fit - pred
    speed = f"  x{ref[0] / total:4.2f} whole call, x{(ref[0] - ref[1] - ref[2]) / rest:5.1f} rest" if ref else ""
    print(f"  {label:<34}{total:8.2f} s | fit {fit:7.2f} s | estimator.predict {pred:6.2f} s "
          f"| rest {rest:6.2f} s ({rest / total * 100:4.1f} %){speed}")


if case["kind"] == "ms":
    from proto_bt_blocks import backtesting_blocks
    from skforecast.model_selection import backtesting_forecaster_multiseries

    series, exog, cv, kw = case["series"], case["exog"], case["cv"], case["kwargs"]
    n_series = len(series) if isinstance(series, dict) else series.shape[1]
    n_folds = len(cv.split(X=series, as_pandas=False)) if cv.window_size else None
    cur = run(lambda: backtesting_forecaster_multiseries(
        forecaster=make(), series=series, exog=exog, cv=cv, n_jobs=1,
        show_progress=False, **kw))
    timings = {}
    new = run(lambda: backtesting_blocks(
        forecaster=make(), series=series, exog=exog, cv=cv, timings=timings, **kw))
    fast = run(lambda: backtesting_blocks(
        forecaster=make(), series=series, exog=exog, cv=cv, fast_metrics=True, **kw))
    same = []
    for a, b in zip(cur[3], new[3]):
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            same.append("identical")
        except AssertionError as e:
            same.append("DIFFERENT " + str(e).splitlines()[0][:70])
    num = cur[3][0].select_dtypes("number").columns
    diff = np.nanmax(np.abs(cur[3][0][num].to_numpy(float) - fast[3][0][num].to_numpy(float)))
    preds = cur[3][1]
    print(f"{case_name}: {n_series} series, {preds.index.nunique()} predicted dates, "
          f"{len(preds)} predictions, estimator {est_cls.__name__} (n_jobs={make().estimator.get_params().get('n_jobs')}), refit={cv.refit}, reps={REPS}")
    line("today", cur)
    line("prototype", new, cur)
    line("prototype + numpy metrics", fast, cur)
    print(f"  outputs: metrics {same[0]}, predictions {same[1]}; warnings same list: "
          f"{cur[4] == new[4]} ({len(cur[4])} today); numpy metrics max abs diff {diff:.3g}")
    stages = ", ".join(f"{k} {v:.2f}" for k, v in timings.items() if k != "total")
    print(f"  prototype stages (s, last run): {stages}")
else:
    import skforecast.recursive._forecaster_recursive as fr
    from skforecast.model_selection import backtesting_forecaster

    y, exog, cv, kw = case["y"], case["exog"], case["cv"], case["kwargs"]
    call = lambda: backtesting_forecaster(
        forecaster=make(), y=y, exog=exog, cv=cv, n_jobs=1, show_progress=False, **kw)
    cur = run(call)
    orig_predict = fr.ForecasterRecursive.predict

    def predict_no_check(self, *a, **k):
        k["check_inputs"] = False
        return orig_predict(self, *a, **k)
    fr.ForecasterRecursive.predict = predict_no_check
    p5 = run(call)
    fr.ForecasterRecursive.predict = orig_predict
    try:
        pd.testing.assert_frame_equal(cur[3][1], p5[3][1], check_exact=True)
        same = "identical"
    except AssertionError as e:
        same = "DIFFERENT " + str(e).splitlines()[0][:70]
    n_folds = len(cur[3][1]) / cv.steps
    print(f"{case_name}: single series, {len(y)} rows, {exog.shape[1]} exog, "
          f"about {n_folds:.0f} folds of {cv.steps} steps, estimator {est_cls.__name__} (n_jobs={make().estimator.get_params().get('n_jobs')}), reps={REPS}")
    line("today", cur)
    line("P5 (check_inputs=False)", p5, cur)
    print(f"  predictions {same}")
print(" ", _env.versions())
