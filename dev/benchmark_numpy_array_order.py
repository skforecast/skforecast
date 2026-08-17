"""
Benchmark: validate the `order='F'` optimization for the bootstrapping input
matrix `X` in `ForecasterRecursive._recursive_predict_bootstrapping`.

Context
-------
`dev/NUMPY-ARRAY-ORDER-AUDIT.md` recommends changing

    X = np.full((n_boot, n_features), fill_value=np.nan, dtype=float)         # order='C'
to
    X = np.full((n_boot, n_features), fill_value=np.nan, order='F', dtype=float)

at `skforecast/recursive/_forecaster_recursive.py:1816`. `X` is filled with
full-height column-block writes (`X[:, a:b] = ...`), which are contiguous under
`order='F'`, and then handed to a `predict_fn(X)` callable every step.

This script reproduces that exact fill-then-predict loop, toggling *only* the
memory layout of `X`, and measures the whole pipeline (allocate -> fill -> predict,
repeated `steps` times) against real fitted estimators. It uses the library's own
`_build_predict_function`, because that is the actual code path and its fast
routes (raw booster predict, `inplace_predict`, per-tree `tree_.predict`, the
CatBoost/linear `predict` fallbacks) react to memory order very differently than
a plain `estimator.predict`.

What it checks
--------------
1. Correctness: predictions must be bit-identical for `order='C'`, `order='F'`,
   and `order='F'` + `ascontiguousarray` before predict (layout never changes values).
2. Performance: mean/std wall time per pipeline over many reps after warmup, plus
   the speedup of `'F'` vs the `'C'` baseline, and the `ascontiguousarray` control
   (which should land back near the `'C'` baseline when the win comes from an
   avoided internal ingestion copy).

Usage
-----
    python dev/benchmark_numpy_array_order.py                 # default: audit-scale
    python dev/benchmark_numpy_array_order.py --tier realistic
    python dev/benchmark_numpy_array_order.py --n-boot 2000 --n-features 200 --steps 24
    python dev/benchmark_numpy_array_order.py --n-reps 300

Environment note: run inside the project conda env (e.g. `skforecast_24_py13`).
"""

import argparse
import platform
from time import perf_counter

import numpy as np
import sklearn
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from skforecast.utils.utils import _build_predict_function


def build_estimators(tier, n_features, seed):
    """
    Fit every available estimator once on synthetic regression data.

    Returns a list of (name, fitted_estimator) tuples. Estimators whose backend
    library is not installed are silently skipped.
    """
    X_train, y_train = make_regression(
        n_samples=2000, n_features=n_features, noise=0.1, random_state=seed
    )

    n_tree = 10 if tier == "cheap" else 100
    n_boost = 10 if tier == "cheap" else 200

    estimators = []

    # sklearn linear model -> _build_predict_function uses np.dot(X, coef)
    estimators.append(("sklearn Ridge", Ridge().fit(X_train, y_train)))

    # sklearn RandomForest -> per-tree tree_.predict on an astype(float32) copy
    estimators.append((
        f"RandomForest(n={n_tree})",
        RandomForestRegressor(n_estimators=n_tree, random_state=seed).fit(X_train, y_train),
    ))

    try:
        from lightgbm import LGBMRegressor
        est = LGBMRegressor(n_estimators=n_boost, random_state=seed, verbose=-1)
        estimators.append((f"LightGBM(n={n_boost})", est.fit(X_train, y_train)))
    except ImportError:
        print("  (LightGBM not installed, skipping)")

    try:
        from xgboost import XGBRegressor
        est = XGBRegressor(n_estimators=n_boost, random_state=seed, tree_method="hist")
        estimators.append((f"XGBoost(n={n_boost})", est.fit(X_train, y_train)))
    except ImportError:
        print("  (XGBoost not installed, skipping)")

    try:
        from catboost import CatBoostRegressor
        est = CatBoostRegressor(
            n_estimators=n_boost, random_state=seed, allow_writing_files=False,
            logging_level="Silent",
        )
        estimators.append((f"CatBoost(n={n_boost})", est.fit(X_train, y_train)))
    except ImportError:
        print("  (CatBoost not installed, skipping)")

    return estimators


def run_pipeline(predict_fn, n_lags, n_exog, n_boot, steps, last_window_init,
                 exog_values, order, ascontig=False):
    """
    Faithful reproduction of `_recursive_predict_bootstrapping`'s fill-then-predict
    loop, parameterized by the memory `order` of the input matrix `X`.

    `X` is filled with two full-height column-block writes per step (lags block
    and exog block), then passed to `predict_fn`. Predictions feed back into
    `last_window`, exactly as in the real recursive loop.
    """
    n_features = n_lags + n_exog
    X = np.full((n_boot, n_features), fill_value=np.nan, order=order, dtype=float)
    predictions = np.full((steps, n_boot), fill_value=np.nan, dtype=float)

    # last_window expanded to 2D: (n_lags + steps, n_boot), one column per trajectory
    last_window = np.tile(last_window_init[:, np.newaxis], (1, n_boot)).astype(float)
    last_window = np.vstack([last_window, np.full((steps, n_boot), np.nan)])

    exog_start = n_lags
    exog_end = n_lags + n_exog

    for i in range(steps):
        remaining = steps - i

        # Contiguous-lags branch (matches self.lags_are_contiguous is True)
        X[:, :n_lags] = last_window[-(remaining + n_lags): -remaining, :][::-1].T

        if n_exog:
            X[:, exog_start:exog_end] = exog_values[i]

        X_pred = np.ascontiguousarray(X) if ascontig else X
        pred = predict_fn(X_pred)

        # CatBoost takes a zero-copy view of an F-contiguous array and marks it
        # read-only; the next step's column write would then fail. The real
        # multi-series loop guards against this the same way
        # (_forecaster_recursive_multiseries.py:2856-2858). The single-series
        # loop must adopt the same guard for the order='F' change to be safe.
        if not X.flags.writeable:
            X.flags.writeable = True

        predictions[i, :] = pred
        last_window[-remaining, :] = pred

    return predictions


def time_pipeline(fn, n_warmup, n_reps):
    """Return (mean_ms, std_ms) over `n_reps` timed reps after `n_warmup` warmups."""
    for _ in range(n_warmup):
        fn()
    times = np.empty(n_reps, dtype=float)
    for r in range(n_reps):
        t0 = perf_counter()
        fn()
        times[r] = (perf_counter() - t0) * 1000.0
    return float(times.mean()), float(times.std())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["cheap", "realistic", "both"], default="both")
    parser.add_argument("--n-boot", type=int, default=250)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-exog", type=int, default=5)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    n_exog = min(args.n_exog, args.n_features)
    n_lags = args.n_features - n_exog

    rng = np.random.default_rng(args.seed)
    last_window_init = rng.standard_normal(n_lags)
    exog_values = rng.standard_normal((args.steps, n_exog)) if n_exog else None

    print("=" * 92)
    print("NumPy array-order benchmark: order='C' (baseline) vs order='F' for bootstrapping X")
    print("=" * 92)
    print(f"Platform      : {platform.platform()}")
    print(f"Python        : {platform.python_version()}   NumPy: {np.__version__}   "
          f"scikit-learn: {sklearn.__version__}")
    print(f"Matrix X      : (n_boot={args.n_boot}, n_features={args.n_features}) "
          f"[n_lags={n_lags}, n_exog={n_exog}]")
    print(f"Loop          : steps={args.steps}   reps={args.n_reps} (warmup={args.n_warmup})")
    print()

    tiers = ["cheap", "realistic"] if args.tier == "both" else [args.tier]

    header = (
        f"{'Estimator':<24}{'C (ms)':>14}{'F (ms)':>14}"
        f"{'Speedup':>11}{'F+ascontig (ms)':>18}{'max|diff|':>13}"
    )

    for tier in tiers:
        print(f"--- Tier: {tier} " + "-" * (92 - len(f"--- Tier: {tier} ")))
        estimators = build_estimators(tier, args.n_features, args.seed)
        print(header)
        print("-" * 92)

        for name, estimator in estimators:
            predict_fn = _build_predict_function(estimator)

            common = dict(
                predict_fn=predict_fn, n_lags=n_lags, n_exog=n_exog,
                n_boot=args.n_boot, steps=args.steps,
                last_window_init=last_window_init, exog_values=exog_values,
            )

            # Correctness: outputs must match across layouts. Bit-identical for
            # tree models; the linear path (np.dot) reorders BLAS summation, so
            # C vs F can differ at ULP scale. Report the max abs diff instead of
            # requiring exact equality.
            pred_c = run_pipeline(order="C", **common)
            pred_f = run_pipeline(order="F", **common)
            pred_fa = run_pipeline(order="F", ascontig=True, **common)
            max_diff = max(
                np.nanmax(np.abs(pred_c - pred_f)),
                np.nanmax(np.abs(pred_c - pred_fa)),
            )

            mean_c, std_c = time_pipeline(lambda: run_pipeline(order="C", **common),
                                          args.n_warmup, args.n_reps)
            mean_f, std_f = time_pipeline(lambda: run_pipeline(order="F", **common),
                                          args.n_warmup, args.n_reps)
            mean_fa, _ = time_pipeline(
                lambda: run_pipeline(order="F", ascontig=True, **common),
                args.n_warmup, args.n_reps,
            )

            speedup = (mean_c - mean_f) / mean_c * 100.0
            print(
                f"{name:<24}"
                f"{mean_c:>9.3f}+-{std_c:<3.2f}"
                f"{mean_f:>9.3f}+-{std_f:<3.2f}"
                f"{speedup:>+10.1f}%"
                f"{mean_fa:>17.3f}"
                f"{max_diff:>13.2e}"
            )
        print()

    print("Reading the results:")
    print("  * Speedup > 0  => order='F' is faster (recommended change is a real win).")
    print("  * 'F+ascontig' reverting toward (small X) or well above (large X) the C")
    print("    baseline confirms the win is an avoided internal ingestion copy: forcing")
    print("    C back pays for exactly the copy 'F' skips, and that copy scales with X.")
    print("  * max|diff| is 0 for tree models (bit-identical). The linear path uses")
    print("    np.dot, whose BLAS summation order differs by layout, so C vs F differ")
    print("    at ULP scale; those tiny diffs can compound through the recursive feed-")
    print("    back loop (visible here on synthetic non-stationary data).")
    print("  * NOTE: order='F' requires resetting X.flags.writeable=True after each")
    print("    predict, or CatBoost (which takes a read-only zero-copy view) crashes on")
    print("    the next step. This guard is baked into run_pipeline here and already")
    print("    exists in the multi-series loop, but NOT yet in the single-series one.")


if __name__ == "__main__":
    main()
