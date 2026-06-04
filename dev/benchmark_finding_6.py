"""
Benchmark for OPTIMIZATION_FINDINGS.md — Finding #6
===================================================
"`_create_predict_inputs` (single series) per-call dtype/column dict rebuilds"

Finding #6 proposes three changes to
`ForecasterRecursive._create_predict_inputs` (skforecast/recursive/_forecaster_recursive.py,
actual lines ~1512-1566; the doc's "1521-1568" references are stale):

  (1) Drop the `copy=True` in
        last_window.iloc[-window_size:].to_numpy(copy=True).ravel()
      claiming "transform_numpy returns a new array regardless" so the copy is
      redundant.

  (2) Cache the column-name and dtype signatures at fit time and replace
        exog.columns.tolist() != self.exog_names_in_       -> tuple compare
        exog.dtypes.to_dict() == self.exog_dtypes_out_      -> object-array compare
      claiming the per-call dict/list builds cost ~10-30 us each.

  (3) Slice before to_numpy:
        exog.to_numpy()[:steps]  ->  exog.iloc[:steps].to_numpy()

This script tests each claim against the REAL skforecast code path.

Run:
    conda run -n skforecast_22_py13 python dev/benchmark_finding_6.py

An explicit verdict per change is printed at the bottom.
"""

from __future__ import annotations

import timeit
import numpy as np
import pandas as pd

from skforecast.recursive import ForecasterRecursive
from skforecast.utils import transform_numpy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


# --------------------------------------------------------------------------- #
# Timing helper                                                               #
# --------------------------------------------------------------------------- #
def best_time(fn, *, number, repeat=7):
    """Best-of-`repeat` average seconds per call (minimises OS/GC noise)."""
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number


def us(seconds):
    return seconds * 1e6


# --------------------------------------------------------------------------- #
# Build a realistic fitted forecaster with exog                              #
# --------------------------------------------------------------------------- #
def make_forecaster(n_exog_cols, n_obs=2000, n_future=50, estimator=None, lags=24):
    rng = np.random.default_rng(123)
    idx = pd.date_range("2000-01-01", periods=n_obs + n_future, freq="h")
    cols = [f"exog_{i}" for i in range(n_exog_cols)]
    full_exog = pd.DataFrame(
        rng.standard_normal((n_obs + n_future, n_exog_cols)), index=idx, columns=cols
    )
    y = pd.Series(
        rng.standard_normal(n_obs).cumsum(), index=idx[:n_obs], name="y"
    )
    exog_train = full_exog.iloc[:n_obs]
    exog_future = full_exog.iloc[n_obs:]  # future-indexed, starts one step ahead
    forecaster = ForecasterRecursive(
        estimator=estimator if estimator is not None else LinearRegression(),
        lags=lags,
    )
    forecaster.fit(y=y, exog=exog_train)
    return forecaster, y, exog_future


# =========================================================================== #
print("=" * 78)
print("FINDING #6 BENCHMARK — _create_predict_inputs micro-optimizations")
print("=" * 78)

# --------------------------------------------------------------------------- #
# PART A — isolated cost of the three targeted operations                     #
# --------------------------------------------------------------------------- #
print("\n" + "-" * 78)
print("PART A — Isolated per-call cost of the targeted operations")
print("-" * 78)

for n_cols in (5, 20, 50):
    exog = pd.DataFrame(
        np.random.standard_normal((100, n_cols)),
        columns=[f"exog_{i}" for i in range(n_cols)],
    )
    names_in = exog.columns.tolist()
    names_in_tuple = tuple(names_in)
    dtypes_out = exog.dtypes.to_dict()
    dtypes_out_values = np.array([dtypes_out[k] for k in exog.columns], dtype=object)
    steps = 24

    # --- column-name comparison ---
    t_list = best_time(
        lambda: exog.columns.tolist() != names_in, number=20000
    )
    t_tuple = best_time(
        lambda: tuple(exog.columns) != names_in_tuple, number=20000
    )

    # --- dtype comparison ---
    t_dict = best_time(
        lambda: exog.dtypes.to_dict() == dtypes_out, number=20000
    )
    t_arr = best_time(
        lambda: bool((exog.dtypes.values == dtypes_out_values).all()), number=20000
    )

    # --- to_numpy slice (exog longer than steps) ---
    t_np_then_slice = best_time(lambda: exog.to_numpy()[:steps], number=20000)
    t_slice_then_np = best_time(lambda: exog.iloc[:steps].to_numpy(), number=20000)

    print(f"\n  n_cols = {n_cols}")
    print(f"    columns: tolist()!=list  {us(t_list):7.2f} us   "
          f"tuple()!=tuple {us(t_tuple):7.2f} us   "
          f"saved {us(t_list - t_tuple):6.2f} us")
    print(f"    dtypes : to_dict()==dict {us(t_dict):7.2f} us   "
          f"arr==arr .all() {us(t_arr):7.2f} us   "
          f"saved {us(t_dict - t_arr):6.2f} us")
    print(f"    slice  : to_numpy()[:s]  {us(t_np_then_slice):7.2f} us   "
          f"iloc[:s].to_numpy() {us(t_slice_then_np):7.2f} us   "
          f"saved {us(t_np_then_slice - t_slice_then_np):6.2f} us")

# --------------------------------------------------------------------------- #
# PART B — share of full predict() represented by these operations            #
# --------------------------------------------------------------------------- #
print("\n" + "-" * 78)
print("PART B — Targeted ops as a share of a full predict() call")
print("-" * 78)
print("  (the doc claims '~5-15% off each predict() call')")

for label, estimator, lags in [
    ("LinearRegression, 24 lags", LinearRegression(), 24),
    ("RandomForest(50),  24 lags", RandomForestRegressor(n_estimators=50, random_state=0), 24),
]:
    for n_cols in (20, 50):
        forecaster, y, exog_future = make_forecaster(
            n_cols, estimator=estimator, lags=lags
        )
        steps = 24

        t_predict = best_time(
            lambda: forecaster.predict(steps=steps, exog=exog_future),
            number=200, repeat=5,
        )
        t_create = best_time(
            lambda: forecaster._create_predict_inputs(
                steps=steps, last_window=None, exog=exog_future,
                check_inputs=True),
            number=200, repeat=5,
        )

        # cost of just the three targeted lines, summed
        names_in = forecaster.exog_names_in_
        dtypes_out = forecaster.exog_dtypes_out_
        ex = exog_future
        t_targeted = (
            best_time(lambda: ex.columns.tolist() != names_in, number=20000)
            + best_time(lambda: ex.dtypes.to_dict() == dtypes_out, number=20000)
            + best_time(lambda: ex.to_numpy()[:steps], number=20000)
        )

        print(f"\n  {label}, n_exog={n_cols}")
        print(f"    full predict()            : {us(t_predict):9.1f} us")
        print(f"    _create_predict_inputs()  : {us(t_create):9.1f} us "
              f"({100 * t_create / t_predict:4.1f}% of predict)")
        print(f"    3 targeted ops (summed)   : {us(t_targeted):9.1f} us "
              f"({100 * t_targeted / t_predict:4.1f}% of predict)")

# --------------------------------------------------------------------------- #
# PART C — change (1): is transform_numpy really 'a new array regardless'?     #
# --------------------------------------------------------------------------- #
print("\n" + "-" * 78)
print("PART C — change (1) premise check: does transform_numpy always copy?")
print("-" * 78)

arr = np.arange(10, dtype=float)
out_none = transform_numpy(array=arr, transformer=None, fit=False)
print(f"  transformer_y=None -> returns SAME object? "
      f"{out_none is arr}  (doc says it returns a new array 'regardless')")

# What 'last_window_values' aliases when copy=True is dropped and transformer_y=None
lw = pd.DataFrame({"y": np.arange(48, dtype=float)})
v_copy = lw.iloc[-24:].to_numpy(copy=True).ravel()
v_view = lw.iloc[-24:].to_numpy(copy=False).ravel()
print(f"  to_numpy(copy=True).ravel()  shares memory with DataFrame? "
      f"{np.shares_memory(v_copy, lw.to_numpy())}")
print(f"  to_numpy(copy=False).ravel() shares memory with DataFrame? "
      f"{np.shares_memory(v_view, lw.to_numpy())}  <- aliases user data if copy dropped")

# --------------------------------------------------------------------------- #
# PART D — change (2): proposed caching code crashes with column-expanding     #
#          transformer_exog (KeyError indexing post-transform dict by in-names)#
# --------------------------------------------------------------------------- #
print("\n" + "-" * 78)
print("PART D — change (2) correctness: proposed cache code vs one-hot exog")
print("-" * 78)

rng = np.random.default_rng(0)
idx = pd.date_range("2000-01-01", periods=500, freq="h")
y = pd.Series(rng.standard_normal(500).cumsum(), index=idx, name="y")
exog_cat = pd.DataFrame(
    {"cat": rng.choice(["a", "b", "c"], size=500)}, index=idx
)
ohe = make_column_transformer(
    (OneHotEncoder(sparse_output=False), ["cat"]),
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")

fc = ForecasterRecursive(estimator=LinearRegression(), lags=12,
                         transformer_exog=ohe)
fc.fit(y=y, exog=exog_cat)

print(f"  exog_names_in_         = {fc.exog_names_in_}")
print(f"  exog_dtypes_out_ keys  = {list(fc.exog_dtypes_out_.keys())}")
try:
    cached = np.array(
        [fc.exog_dtypes_out_[k] for k in fc.exog_names_in_], dtype=object
    )
    print(f"  proposed cache built OK: {cached}")
    bug = False
except KeyError as e:
    print(f"  proposed cache code raised KeyError: {e}")
    print("  -> change (2) as written crashes whenever transformer_exog "
          "renames/expands columns")
    bug = True

# --------------------------------------------------------------------------- #
# VERDICT
# --------------------------------------------------------------------------- #
print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
print("""
Change (1) drop copy=True:
  - PREMISE WRONG: transform_numpy(transformer=None) is a pass-through
    (returns the SAME object), so the copy is NOT made redundant by it.
  - Dropping copy=True makes last_window_values a VIEW into the user's
    last_window DataFrame (when transformer_y is None and differentiation
    is None). Nothing mutates it today, so it is "safe today", but it trades
    a cheap, defensive isolation (one window_size-length float array) for
    fragile aliasing of user-owned memory. Negligible, unmeasurable saving.
    -> NOT WORTH IT.

Change (2) cache name/dtype signatures:
  - The proposed code is BUGGY: it indexes exog_dtypes_out_ (keyed by
    POST-transform column names) using exog_names_in_ (PRE-transform names).
    With any column-expanding/renaming transformer_exog (one-hot,
    ColumnTransformer) this raises KeyError at fit time (PART D).
  - Even fixed, PART A shows the dict/list builds cost only a few us, and
    PART B shows _create_predict_inputs is a small fraction of predict();
    the saving is far below the claimed 5-15% of predict().
    -> REAL micro-cost, but tiny; proposed code must be rewritten to key on
       the post-transform names; not worth the API surface for the gain.

Change (3) slice before to_numpy:
  - Only helps when exog has many more rows than steps; in the common case
    (exog length == steps) iloc[:steps] adds pandas overhead (PART A).
    -> WASH; marginal at best.

Overall: the "5-15% off each predict()" headline is not supported. #6 is at
most a couple of micro-tweaks worth single-digit microseconds, and change (2)
as written is incorrect.
""")
