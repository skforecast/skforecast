"""
Benchmark of the preprocessing speedups (current vs P1 encoder, P2 scaler
shortcut and both), end-to-end per workflow. Reports the median and the
spread (max - min) of the repetitions, and the first call separately.

python bench.py --size M --est ridge --reps 5
python bench.py --size M --est lgbm --reps 3 --no-backtesting
python bench.py --size M --est ridge --text-cats    # categorical exog as str

Run it with nothing else running on the machine (OpenMP estimators degrade a
lot under CPU contention). Writes results/bench_<size>_<est>[_text].json
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from data import SIZES, env_report, make_data, make_forecaster, split_train_future  # noqa: E402
from prototypes import use_fast_encoder, use_fast_scaler  # noqa: E402

from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries  # noqa: E402

warnings.filterwarnings("ignore")

VARIANTS = ("current", "P1", "P2", "P1+P2")


@contextlib.contextmanager
def variant_ctx(variant, forecaster):
    with contextlib.ExitStack() as stack:
        if variant in ("P1", "P1+P2"):
            stack.enter_context(use_fast_encoder(forecaster))
        if variant in ("P2", "P1+P2"):
            stack.enter_context(use_fast_scaler())
        yield


def measure(fn, reps):
    times, out = [], None
    for r in range(reps + 1):
        gc.collect()
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    rest = times[1:]
    return {"first": times[0], "median": statistics.median(rest),
            "spread": max(rest) - min(rest), "times": rest}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S")
    ap.add_argument("--est", default="ridge")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-backtesting", action="store_true")
    ap.add_argument("--text-cats", action="store_true",
                    help="categorical exog as str (P1 falls back to scikit-learn)")
    args = ap.parse_args()

    s, e = make_data(**SIZES[args.size])
    if args.text_cats:
        e = {k: v.astype({c: str for c in ("cat_common", "cat_store", "cat_promo")})
             for k, v in e.items()}
    s_tr, e_tr, e_fut = split_train_future(s, e, 168)
    res = {"config": vars(args), "env": env_report(),
           "n_rows": sum(len(v) for v in s_tr.values()), "runs": {}}

    def record(name, variant, stats):
        res["runs"].setdefault(name, {})[variant] = stats
        base = res["runs"][name].get("current", stats)["median"]
        print(f"[{args.size} {args.est}] {name:22s} {variant:6s} median {stats['median']:.3f}s "
              f"spread {stats['spread']:.3f}s first {stats['first']:.3f}s "
              f"x{base / stats['median']:.2f}", flush=True)

    # Build of X_train / y_train (the part of fit that P1 changes)
    for v in VARIANTS:
        def build():
            f = make_forecaster(args.est)
            with variant_ctx(v, f):
                f._create_train_X_y(s_tr, exog=e_tr)
        record("build_X_train", v, measure(build, args.reps)[0])

    # fit (new forecaster each time) and the predict side on a fitted one
    for v in VARIANTS:
        def fit():
            f = make_forecaster(args.est)
            with variant_ctx(v, f):
                f.fit(s_tr, exog=e_tr, store_in_sample_residuals=True)
            return f
        stats, f = measure(fit, max(1, args.reps - 1))
        record("fit", v, stats)
        # NOTE: The encoder fitted in `fit` is kept; only the scaler shortcut
        # is applied here (a new encoder would not be fitted).
        scaler_ctx = use_fast_scaler() if v in ("P2", "P1+P2") else contextlib.nullcontext()
        with scaler_ctx:
            for steps in (24, 168):
                record(f"predict_{steps}", v, measure(
                    lambda: f.predict(steps=steps, exog=e_fut), args.reps)[0])
            record("interval_conformal_24", v, measure(
                lambda: f.predict_interval(steps=24, exog=e_fut, method="conformal"),
                args.reps)[0])

    if not args.no_backtesting:
        n_span = max(len(v) for v in s.values())
        for refit in (False, True):
            preds = {}
            for v in VARIANTS:
                def bt():
                    f = make_forecaster(args.est)
                    cv = TimeSeriesFold(steps=24, initial_train_size=n_span - 168,
                                        refit=refit, verbose=False)
                    with variant_ctx(v, f):
                        return backtesting_forecaster_multiseries(
                            f, series=s, exog=e, cv=cv, metric="mean_absolute_error",
                            n_jobs=1, show_progress=False, suppress_warnings=True)[1]
                stats, p = measure(bt, 1)
                preds[v] = p["pred"].to_numpy()
                record(f"backtesting_refit={refit}", v, stats)
            for v in VARIANTS[1:]:
                np.testing.assert_array_equal(preds["current"], preds[v])

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    name = f"bench_{args.size}_{args.est}{'_text' if args.text_cats else ''}.json"
    (out / name).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
