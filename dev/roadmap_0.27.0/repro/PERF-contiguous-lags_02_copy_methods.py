"""
PERF-contiguous-lags, script 02: cost of obtaining the lag matrix of
ForecasterRecursive (lag_1 first) in C order, with several methods, compared
with the current view of `sliding_window_view` (strides (8, -8)).

Also the cost of `pd.isna(X).any()` (the check of `_filter_nan_train_X_y`) on
the view and on the contiguous copy.

Run from the root of the repository:
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_02_copy_methods.py
"""

import timeit

import numpy as np
import pandas as pd


def best(fn, number, repeat=7):
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number * 1e3


rng = np.random.default_rng(0)
print("| rows | lags | view (current) | ascontiguousarray(view) | column loop "
      "| reversed y + ascontiguousarray | np.take 2D index | isna view | isna C |")
print("|---|---|---|---|---|---|---|---|---|")
for n in (1_000, 20_000, 200_000):
    for k in (24, 168):
        y = rng.normal(size=n + k)
        lags = np.arange(1, k + 1)

        def view():
            sw = np.lib.stride_tricks.sliding_window_view(y, k)[:-1]
            return sw[:, ::-1]

        def ascontig():
            return np.ascontiguousarray(view())

        def column_loop():
            X = np.empty((n, k))
            for j, lag in enumerate(lags):
                X[:, j] = y[k - lag: k - lag + n]
            return X

        def reversed_y():
            y_rev = y[::-1]
            sw = np.lib.stride_tricks.sliding_window_view(y_rev, k)[1:]
            return np.ascontiguousarray(sw[::-1])

        idx = (np.arange(n)[:, None] + (k - lags)[None, :])

        def take2d():
            return np.take(y, idx)

        ref = ascontig()
        for f in (column_loop, reversed_y, take2d):
            assert np.array_equal(f(), ref), f.__name__
            assert f().flags.c_contiguous
        number = max(1, int(2e6 / (n * k)))
        V = view()
        C = ref
        print(
            f"| {n:,} | {k} | {best(view, number * 10):.4f} | "
            f"{best(ascontig, number):.3f} | {best(column_loop, number):.3f} | "
            f"{best(reversed_y, number):.3f} | {best(take2d, number):.3f} | "
            f"{best(lambda: pd.isna(V).any(), number):.3f} | "
            f"{best(lambda: pd.isna(C).any(), number):.3f} |"
        )
