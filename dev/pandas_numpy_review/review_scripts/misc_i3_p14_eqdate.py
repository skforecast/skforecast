"""I3 / P14: ForecasterEquivalentDate, NaN semantics per offset type and agg_func,
and exactness of agg_func(arr, axis=1) vs np.apply_along_axis(agg_func, 1, arr)."""
import _env
import warnings
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterEquivalentDate

idx = pd.date_range("2020", periods=60, freq="D")
y = pd.Series(np.arange(60, dtype=float), index=idx)
y.iloc[-7] = np.nan


def custom_mean(x):  # no `axis` argument
    return np.sum(x) / len(x)


print("--- I3: NaN in y at position -7, n_offsets=2 ---")
for agg in (np.mean, np.median, np.nanmean, np.max, custom_mean):
    for off in (7, pd.offsets.Day(7)):
        f = ForecasterEquivalentDate(offset=off, n_offsets=2, agg_func=agg)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                f.fit(y)
                p = f.predict(1).iloc[0]
            except Exception as e:
                p = f"{type(e).__name__}: {str(e)[:60]}"
        print(f"{agg.__name__:12s} {off!r:12s} pred={p}  warnings={[x.category.__name__ for x in w]}")

print("--- I3b: same forecaster, DateOffset: fit (binning) vs predict semantics ---")
f = ForecasterEquivalentDate(offset=pd.offsets.Day(7), n_offsets=2, agg_func=np.mean)
f.fit(y, store_in_sample_residuals=True)
print("n in-sample residuals stored:", len(f.in_sample_residuals_),
      "| any NaN:", np.isnan(f.in_sample_residuals_).any(),
      "| rows available:", len(y) - 14)
f2 = ForecasterEquivalentDate(offset=7, n_offsets=2, agg_func=np.mean)
f2.fit(y, store_in_sample_residuals=True)
print("int offset: n residuals:", len(f2.in_sample_residuals_), "any NaN:",
      np.isnan(f2.in_sample_residuals_).any())

print("--- P14: agg(arr, axis=1) vs apply_along_axis, bitwise ---")
rng = np.random.default_rng(0)
for func in (np.mean, np.median, np.nanmean, np.sum, np.std):
    for width in (2, 3, 7, 8, 9, 16, 17, 64, 129, 1000):
        diffs = 0
        for scale in (1.0, 1e8):
            arr = rng.normal(size=(2000, width)) * scale + scale
            for a in (arr, np.asfortranarray(arr), arr.T.copy().T):
                r1 = func(a, axis=1)
                r2 = np.apply_along_axis(func, 1, a)
                diffs += int((r1 != r2).sum())
        if diffs:
            print(f"{func.__name__:8s} width={width:5d} rows differing={diffs}")
print("(no line above = bitwise identical for that func/width)")
print("predict path, axis=0 with shape (n_offsets, steps): first n_offsets with a difference")
for func in (np.mean, np.median, np.nanmean):
    first = None
    for n_off in range(2, 40):
        arr = rng.normal(size=(n_off, 500)) * 1e3
        if not np.array_equal(func(arr, axis=0), np.apply_along_axis(func, 0, arr)):
            first = n_off
            break
    print(f"   {func.__name__:8s}: {first}")
arr = rng.normal(size=(5, 3))
for func in (custom_mean, lambda x: x[0]):
    try:
        func(arr, axis=1)
    except TypeError as e:
        print("user callable without axis:", e)
# axis-less callables that silently give a different shape
print("np.ptp-like lambda x: x.max()-x.min() with axis kw absent -> TypeError; "
      "callable accepting **kw but ignoring axis:",
      (lambda x, **kw: np.mean(x))(arr, axis=1).shape, "vs expected (5,)")
print(_env.versions())
