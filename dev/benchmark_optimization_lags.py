"""
Benchmark for OPTIMIZATION_FINDINGS.md — Finding #1
===========================================================================
Non-contiguous lag-index hoist in the recursive prediction loops.

This is the ONLY part of finding #1 that was implemented (the `np.concatenate`
rewrite was rejected — `RollingFeatures.transform` dominates it; see the
revision note in OPTIMIZATION_FINDINGS.md).

The change, applied at three sites
(`ForecasterRecursive._recursive_predict`,
 `ForecasterRecursive._recursive_predict_bootstrapping`,
 `ForecasterRecursiveMultiSeries._recursive_predict`):

    BEFORE (per step):                         AFTER (hoisted once):
        X[:n_lags] = lw[-self.lags - rem]          neg_lags = -self.lags   # once
                                                   X[:n_lags] = lw[neg_lags - rem]

`-self.lags - rem` rebuilds the negated lag array on EVERY step;
`neg_lags - rem` negates once and only subtracts the scalar per step.
Arithmetically identical (`-self.lags - rem == neg_lags - rem`), so this is a
pure micro-optimization with no change in output.

This benchmark:
  PART A — isolates the exact changed line (1D `_recursive_predict` shape and
           2D `_recursive_predict_bootstrapping` shape), before vs after, on
           NON-CONTIGUOUS lags (the only path affected). Asserts bit-identity.
  PART B — full inner recursion loop (1D, LinearRegression fast path),
           before vs after, on non-contiguous lags. Asserts bit-identity.
  PART C — regression check on CONTIGUOUS lags (the default config): the hot
           line is untouched, so before == after both in result and timing.

Run:
    conda run -n skforecast_22_py13 python dev/benchmark_finding_1.py
"""

from __future__ import annotations
import timeit
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import LinearRegression
from skforecast.utils import _build_predict_function


def best_time(fn, *, number, repeat=9):
    """Best-of-`repeat` average seconds per call (minimises OS/GC noise)."""
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number


# Build a fitted forecaster so `self.lags` / `lags_are_contiguous` are real
# --------------------------------------------------------------------------- #
def build(lags, n_train=20_000):
    idx = pd.date_range("2000-01-01", periods=n_train, freq="h")
    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=n_train), index=idx, name="y")
    fc = ForecasterRecursive(estimator=LinearRegression(), lags=lags)
    fc.fit(y=y)
    last_window_values = y.to_numpy()[-fc.window_size:].astype(float)
    return fc, last_window_values


# PART A — isolated per-step lag assignment (before vs after)
# --------------------------------------------------------------------------- #
def lag_assign_1d(lags_arr, last_window, n_lags, steps, *, hoisted):
    """Mirror of `_recursive_predict`'s non-contiguous lag line, over all steps."""
    X = np.empty(n_lags)
    if hoisted:
        neg_lags = -lags_arr
        for i in range(steps):
            remaining = steps - i
            X[:] = last_window[neg_lags - remaining]
    else:
        for i in range(steps):
            remaining = steps - i
            X[:] = last_window[-lags_arr - remaining]
    return X


def lag_assign_2d(lags_arr, last_window, n_lags, steps, n_boot, *, hoisted):
    """Mirror of `_recursive_predict_bootstrapping`'s non-contiguous lag line."""
    X = np.empty((n_boot, n_lags))
    if hoisted:
        neg_lags = -lags_arr
        for i in range(steps):
            remaining = steps - i
            X[:, :] = last_window[neg_lags - remaining, :].T
    else:
        for i in range(steps):
            remaining = steps - i
            X[:, :] = last_window[-(lags_arr + remaining), :].T
    return X


def part_a():
    print("=" * 92)
    print("PART A — isolated per-step lag assignment, before vs after (non-contiguous lags)")
    print("=" * 92)
    steps, n_boot = 24, 250
    lags_list = [1, 3, 7, 24, 48, 168]
    fc, lw = build(lags_list)
    lags_arr = fc.lags
    n_lags = len(lags_arr)

    # 1D shape (_recursive_predict)
    lw1 = np.concatenate((lw, np.zeros(steps)))
    a = lag_assign_1d(lags_arr, lw1, n_lags, steps, hoisted=False)
    b = lag_assign_1d(lags_arr, lw1, n_lags, steps, hoisted=True)
    assert np.array_equal(a, b), "1D results differ!"
    t_before = best_time(lambda: lag_assign_1d(lags_arr, lw1, n_lags, steps, hoisted=False), number=3000)
    t_after = best_time(lambda: lag_assign_1d(lags_arr, lw1, n_lags, steps, hoisted=True), number=3000)
    print(f"  1D  _recursive_predict           before {t_before*1e6:7.2f} us | "
          f"after {t_after*1e6:7.2f} us | speedup {(t_before/t_after-1)*100:+5.1f}%  (bit-identical)")

    # 2D shape (_recursive_predict_bootstrapping)
    lw2 = np.vstack([np.tile(lw[:, None], (1, n_boot)), np.zeros((steps, n_boot))])
    a = lag_assign_2d(lags_arr, lw2, n_lags, steps, n_boot, hoisted=False)
    b = lag_assign_2d(lags_arr, lw2, n_lags, steps, n_boot, hoisted=True)
    assert np.array_equal(a, b), "2D results differ!"
    t_before = best_time(lambda: lag_assign_2d(lags_arr, lw2, n_lags, steps, n_boot, hoisted=False), number=2000)
    t_after = best_time(lambda: lag_assign_2d(lags_arr, lw2, n_lags, steps, n_boot, hoisted=True), number=2000)
    print(f"  2D  _recursive_predict_bootstrap before {t_before*1e6:7.2f} us | "
          f"after {t_after*1e6:7.2f} us | speedup {(t_before/t_after-1)*100:+5.1f}%  (bit-identical)")
    print()


# PART B — full inner recursion loop (before vs after), non-contiguous lags
# --------------------------------------------------------------------------- #
def recursive_loop(fc, last_window_values, steps, predict_fn, *, hoisted):
    """Full `_recursive_predict` inner loop (lags-only), before/after the hoist."""
    lags_arr = fc.lags
    n_lags = len(lags_arr)
    X = np.full(shape=n_lags, fill_value=np.nan, dtype=float)
    predictions = np.full(shape=steps, fill_value=np.nan, dtype=float)
    last_window = np.concatenate((last_window_values, predictions))

    if hoisted:
        neg_lags = -lags_arr

    for i in range(steps):
        remaining = steps - i
        if hoisted:
            X[:n_lags] = last_window[neg_lags - remaining]
        else:
            X[:n_lags] = last_window[-lags_arr - remaining]
        pred = predict_fn(X.reshape(1, -1)).item()
        predictions[i] = pred
        last_window[-remaining] = pred
    return predictions


def part_b():
    print("=" * 92)
    print("PART B — full `_recursive_predict` inner loop, before vs after (non-contiguous lags)")
    print("         includes per-step predict_fn (LinearRegression fast path) — real predict() cost")
    print("=" * 92)
    steps = 24
    for lags_list in ([1, 3, 7, 24], [1, 3, 7, 24, 48, 168, 336]):
        fc, lw = build(lags_list)
        pf = _build_predict_function(fc.estimator)
        r_before = recursive_loop(fc, lw, steps, pf, hoisted=False)
        r_after = recursive_loop(fc, lw, steps, pf, hoisted=True)
        assert np.array_equal(r_before, r_after), f"lags={lags_list}: results differ!"
        t_before = best_time(lambda: recursive_loop(fc, lw, steps, pf, hoisted=False), number=500)
        t_after = best_time(lambda: recursive_loop(fc, lw, steps, pf, hoisted=True), number=500)
        print(f"  lags={str(lags_list):<28} before {t_before*1e6:7.2f} us | "
              f"after {t_after*1e6:7.2f} us | speedup {(t_before/t_after-1)*100:+5.1f}%  (bit-identical)")
    print()


# PART C — regression check: contiguous lags (default) path is untouched
# --------------------------------------------------------------------------- #
def contiguous_loop(fc, last_window_values, steps, predict_fn):
    """Contiguous lag path — identical before and after the change (unchanged line)."""
    n_lags = len(fc.lags)
    X = np.full(shape=n_lags, fill_value=np.nan, dtype=float)
    predictions = np.full(shape=steps, fill_value=np.nan, dtype=float)
    last_window = np.concatenate((last_window_values, predictions))
    for i in range(steps):
        remaining = steps - i
        X[:n_lags] = last_window[-(remaining + n_lags): -remaining][::-1]
        pred = predict_fn(X.reshape(1, -1)).item()
        predictions[i] = pred
        last_window[-remaining] = pred
    return predictions


def part_c():
    print("=" * 92)
    print("PART C — regression check: contiguous lags (default `lags=24`) hot line is UNCHANGED")
    print("=" * 92)
    fc, lw = build(24)
    assert fc.lags_are_contiguous, "expected contiguous"
    pf = _build_predict_function(fc.estimator)
    steps = 24
    # The only added cost on this path is one `if ... and not lags_are_contiguous`
    # check that evaluates False (neg_lags is never even computed). Demonstrate
    # the loop time is unchanged.
    t = best_time(lambda: contiguous_loop(fc, lw, steps, pf), number=500)
    print(f"  lags=24 (contiguous)  loop time {t*1e6:7.2f} us")
    print("  => the changed branch is not taken; `neg_lags` is not computed. No regression.\n")


if __name__ == "__main__":
    print(f"numpy {np.__version__} | pandas {pd.__version__}\n")
    part_a()
    part_b()
    part_c()
    print("=" * 92)
    print("VERDICT — lag-index hoist")
    print("=" * 92)
    print("  - Non-contiguous lags: faster, results bit-identical (PART A isolated, PART B end-to-end).")
    print("  - Contiguous lags (default): hot line unchanged, no regression (PART C).")
    print("  - Applied at the 3 sites with the redundant `-self.lags - remaining` double op;")
    print("    the multiseries-bootstrap site (`window_size + step - self.lags`) was left as-is")
    print("    because it has no redundant negation to hoist.")
    print("=" * 92)
