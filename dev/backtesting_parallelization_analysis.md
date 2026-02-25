# Backtesting Parallelization Analysis

## Profiling Baseline

Measured on `_backtesting_forecaster` with `DummyRegressor`, `lags=24`, `steps=30`,
`n=10_000`, `refit=True`, `n_jobs=1`, producing 167 folds.

```
Total time: 1.02 s

Line                                          Time      % Time
------------------------------------------------------------
forecaster = deepcopy(forecaster)             2.23 ms    0.2%
cv = deepcopy(cv)                             0.20 ms    0.0%
folds = cv.split(X=y, as_pandas=False)        3.02 ms    0.3%
forecaster.fit(y.iloc[:initial_train_size])   3.75 ms    0.4%

Parallel(n_jobs=1)(...) — all folds          950.20 ms  93.1%

np.unique(np.concatenate(train_indexes))      18.23 ms   1.8%
pd.concat(backtest_predictions)               7.70 ms    0.8%
```

Key takeaway: **93.1% of total time is inside the `Parallel` call**. Everything else is noise.
With `n_jobs=1` and `DummyRegressor` (trivially fast per fold), the per-fold loky/delayed
dispatch overhead becomes the dominant bottleneck. With a real estimator the fit+predict cost
dominates instead, but the IPC waste documented below still applies.

---

## When Is Parallelism Actually Activated?

`select_n_jobs_backtesting` sets `n_jobs = cpu_count() - 1` only in these cases:

| Forecaster | Estimator | `refit` | n_jobs |
|---|---|---|---|
| `ForecasterRecursive` | non-linear (RF, GBM, …) | `True` or `False` | `cpu_count()-1` |
| `ForecasterDirect` / `ForecasterDirectMultiVariate` | any | `True` | `cpu_count()-1` |
| `ForecasterRecursive` | linear | any | `1` (sequential) |
| `ForecasterRecursive` | `LGBMRegressor(n_jobs=1)` | any | `cpu_count()-1` |
| `ForecasterRecursive` | `LGBMRegressor` (default) | any | `1` |
| `ForecasterStats` / `ForecasterEquivalentDate` | — | any | `1` |
| any | any | `int` (intermittent) | `1` |

---

## What Gets Serialized and Sent to Every Worker

When `Parallel(n_jobs>1)` dispatches via joblib's loky backend, it pickles all arguments of
`_fit_predict_forecaster` for **each fold call individually**. `kwargs_fit_predict_forecaster`
is built once and reused for every fold, meaning these objects are pickled N-times (once per fold):

### Object-by-object breakdown

| Object | Contents | Size order | Sent N times? |
|---|---|---|---|
| `forecaster` | fitted estimator + all arrays below | large (MB scale) | **Yes** |
| `├─ estimator` | sklearn model trees/coefficients as numpy arrays | depends (KB–tens of MB) | **Yes** |
| `├─ in_sample_residuals_` | up to 10,000 floats (numpy array) | ~80 KB | **Yes** |
| `├─ in_sample_residuals_by_bin_` | dict of numpy arrays, 10,000 total | ~80 KB | **Yes** |
| `├─ last_window_` | DataFrame of `window_size` rows | negligible | **Yes** |
| `├─ binner`, `binner_intervals_` | QuantileBinner state | small | **Yes** |
| `y` | the **entire** time series as pandas Series | 80 KB for 10K×float64 | **Yes** |
| `exog` | the **entire** exog DataFrame | proportional to columns | **Yes** |
| scalars (`gap`, `n_boot`, etc.) | primitive values | negligible | Yes, but irrelevant |

---

## The Core Problem: Workers Only Need Tiny Slices, but Receive the Entire Data

Inside `_fit_predict_forecaster`, here is what each fold **actually uses** from `y` and `exog`:

**`refit=False` (the only parallelizable case):**
```python
last_window_y    = y.iloc[last_window_iloc_start : last_window_iloc_end]  # window_size rows
next_window_exog = exog.iloc[test_iloc_start : test_iloc_end]              # steps rows
```

For the profiling example: `window_size=24`, `steps=30`. The worker needs **24 values** from `y`,
but receives **10,000**. That is a **416× overread** of `y` alone, per fold, for 167 folds.

**`refit=True`:**
```python
y_train = y.iloc[train_iloc_start : train_iloc_end]  # large expanding slice
```
This is sequential anyway (`n_jobs` is forced to 1 for intermittent int refit, and
`select_n_jobs_backtesting` returns `cpu_count()-1` only for non-linear `refit=True/False` cases),
but the same waste applies when it does run in parallel.

---

## The Forecaster: Serialized N Times Despite Being Read-Only

For `refit=False`, the forecaster is fitted **once** outside the loop, then passed as a kwarg
to every fold. Joblib picks the loky backend (separate processes), so the object cannot be shared
via reference — it is **fully pickled once per task**.

A `RandomForestRegressor(n_estimators=100)` with 10K training points serializes to ~5–20 MB.
For 167 folds that is **835 MB–3.3 GB of redundant IPC traffic**, none of which carries any new
information since the object is never mutated in `refit=False` folds.

Joblib does apply automatic numpy memmap for arrays larger than `max_nbytes` (default 1 MB) in
certain backends, but this **only works for bare numpy arrays passed as top-level arguments** —
not arrays embedded inside a nested sklearn object inside a pandas-aware forecaster object.
The forecaster gets cloudpickled in full each time.

---

## The Nested Function Penalty

`_fit_predict_forecaster` is defined **inside** `_backtesting_forecaster`. Joblib uses cloudpickle
to serialize it, which is slower than pickling a module-level function and carries the full closure
scope. Moving it to module level would allow joblib to serialize it by reference (just its import
path).

---

## Is the Current Strategy Optimal? No.

The optimal strategy for `refit=False` (the only case where parallelisation is meaningful) would
be:

### 1. Pre-slice data before `Parallel`
Build `(last_window_y, exog_slice)` tuples per fold in the main process — only tiny arrays — and
pass those instead of full `y` and `exog`. Each worker receives only what it needs.

Per-fold IPC cost reduction for `y`:

| Before | After |
|---|---|
| 10,000 rows (full series) | 24 rows (`window_size`) |
| **416× overread** | Exact minimum |

### 2. Broadcast the forecaster once via worker initializer
Use joblib's `initializer`/`initargs` mechanism (available in the loky backend) to send the fitted
forecaster **once** to each worker process at startup, storing it in a module-level global. Each
fold task then reads it without re-serialization.

```python
# Sketch — not yet implemented
import skforecast.model_selection._validation as _val_module

def _worker_init(forecaster_pkl):
    import pickle
    _val_module._WORKER_FORECASTER = pickle.loads(forecaster_pkl)

with Parallel(n_jobs=n_jobs, initializer=_worker_init, initargs=(pickle.dumps(forecaster),)) as p:
    results = p(delayed(_fit_predict_forecaster)(fold=fold, ...) for fold in folds)
```

This would reduce the per-fold forecaster IPC cost from `O(N × estimator_size)` to
`O(n_workers × estimator_size)`.

### 3. Move `_fit_predict_forecaster` to module level
Eliminates cloudpickle overhead and makes joblib serialize the function by name (a short string)
rather than its full bytecode + closure.

---

## Summary of Impact

| Optimization | Status | Per-fold savings |
|---|---|---|
| Pre-slice `y` and `exog` | Proposed | `O(full_series)` → `O(window_size + steps)` |
| Broadcast forecaster via worker initializer | Proposed | `O(estimator_size)` → `O(0)` |
| Move inner function to module level | Proposed | cloudpickle overhead eliminated |

Combined, all three would reduce per-fold IPC cost from
`O(estimator_size + full_series_size)` to `O(window_size + steps)` — practically near zero —
making parallelization genuinely cost-effective for heavy estimators (e.g. large random forests,
gradient boosting).

---

## Implementation Priority

| Priority | Optimization | Effort | Risk | Expected gain |
|---|---|---|---|---|
| 1 | Pre-slice `y` and `exog` | Low | Very low — pure data reshaping, no API change | High for large series |
| 2 | Move `_fit_predict_forecaster` to module level | Low | Low — requires passing extra arguments explicitly | Small but free |
| 3 | Broadcast forecaster via worker initializer | Medium | Medium — requires global state per worker, harder to test | Very high for large models |

**Optimization 1** should always be done first: it is self-contained, has no side effects, and
eliminates the largest single source of redundant traffic in the common `refit=False` path.

**Optimization 3** has the highest ceiling for heavy models (RF, GBM), but using worker-level
globals introduces statefulness that must be carefully managed (especially if the same process
pool is reused across multiple `backtesting_forecaster` calls with different forecasters).
