# Deep Review: Optimization 3 — Broadcast Forecaster via Worker Initializer

## Context

When `_backtesting_forecaster` runs with `n_jobs > 1` and `refit=False`, the fitted
forecaster is passed as a kwarg to every fold in `Parallel(n_jobs)(...)`. Joblib's loky
backend pickles it **once per fold task**. The original analysis proposes using a worker
`initializer` to send the forecaster once per worker process, storing it in a module-level
global. This document evaluates that proposal and the alternatives.

---

## 1. Measured Serialization Cost (Empirical)

Benchmarked on this machine (Python 3.11, joblib 1.5.3, sklearn, lightgbm):

### RandomForestRegressor(n_estimators=100), 10K training points, 24 features

| Operation | Per fold | 167 folds total |
|---|---:|---:|
| `pickle.dumps(rf)` | **121.2 ms** | **20.24 s** |
| `pickle.loads(pkl)` | **62.2 ms** | **10.39 s** |
| `rf.predict(X[30×24])` | **4.2 ms** | **0.70 s** |
| **Pickle size** | **91.1 MB** | — |

**Per-fold IPC round-trip (serialize + deserialize): ~183 ms.**
**Per-fold predict: ~4 ms.**

The IPC overhead is **44× the actual work** per fold. For 167 folds across 4 workers, the
`Parallel` call with loky took **61.23 s** — most of which is serialization, not prediction.

### LGBMRegressor(n_estimators=100, n_jobs=1), 10K training points, 24 features

| Operation | Per fold | 167 folds total |
|---|---:|---:|
| `pickle.dumps(lgb)` | **6.9 ms** | **1.15 s** |
| `pickle.loads(pkl)` | **2.5 ms** | **0.41 s** |
| **Pickle size** | **0.29 MB** | — |

LGBM serializes to **0.29 MB** vs RF's **91 MB** — a 314× difference. The IPC overhead for
LGBM is small enough that it's not the bottleneck. **This optimization primarily benefits
RandomForest, GradientBoosting, HistGradientBoosting** and other sklearn tree ensembles whose
pickle representation includes full tree structures in Python objects.

### Backend comparison: predict-only, RF(100), n_jobs=4, 167 folds

| Backend | Wall-clock |
|---|---:|
| `Parallel(n_jobs=4)` (loky/processes) | **61.23 s** |
| `Parallel(n_jobs=4, prefer='threads')` | **1.30 s** |
| Sequential (plain loop) | **0.87 s** |

**Threads are 47× faster than loky for predict-only workloads.** This is because threads
share the forecaster in memory (zero IPC cost) and sklearn's tree `predict()` releases the
GIL via Cython extensions.

---

## 2. Where the Forecaster Is Used During Backtesting

From the code in `_fit_predict_forecaster`:

### `refit=False` path (parallelized)

```python
# fold[5] is False — model NOT refitted
last_window_y = y.iloc[lw_start:lw_end]     # pre-sliced (optimization 1)
next_window_exog = exog.iloc[...]            # pre-sliced (optimization 1)

forecaster.predict(steps, last_window=last_window_y, exog=next_window_exog)
# Optionally: predict_bootstrapping, predict_quantiles, predict_interval, etc.
```

The forecaster is **read-only**. `predict()` needs the full forecaster state (estimator,
residuals, binner, transformers, etc.) — 100% of the pickle payload is in the read-set.
Nothing can be dropped.

### `refit=True` path (also parallelized for non-linear estimators)

```python
# fold[5] is True — model IS refitted
forecaster.fit(y=y_train, exog=exog_train, ...)
forecaster.predict(...)
```

Each worker gets a copy, refits it, then predicts. The initial forecaster state is
overwritten by `fit()`, so sending the full fitted forecaster is wasteful — an unfitted
copy would suffice. But `refit=True` workers need training data, and `fit()` dominates the
per-fold time anyway.

---

## 3. Evaluation of the Three Approaches

### 3.A. Worker Initializer (original proposal)

**Mechanism:** Use `Parallel(initializer=fn, initargs=(pickled_forecaster,))` to send the
forecaster once per worker at pool startup. Workers read it from a module-level global.

```python
_WORKER_FORECASTER = None  # module-level global

def _worker_init(forecaster_bytes):
    global _WORKER_FORECASTER
    _WORKER_FORECASTER = pickle.loads(forecaster_bytes)

def _fit_predict_forecaster(fold, last_window_y, exog_test, ...):
    forecaster = _WORKER_FORECASTER
    return forecaster.predict(...)

Parallel(n_jobs=n_jobs, initializer=_worker_init, initargs=(pickle.dumps(forecaster),))(
    delayed(_fit_predict_forecaster)(fold=fold, ...) for fold in folds
)
```

**IPC cost:** `O(n_workers × estimator_size)` instead of `O(n_folds × estimator_size)`.

For RF(100) with 4 workers and 167 folds:
- Before: 167 × 91 MB = **15.2 GB** serialized
- After: 4 × 91 MB = **364 MB** serialized
- **Savings: 97.6%**

**Pros:**
- Massive IPC reduction for large sklearn models
- joblib 1.5.3 confirms `initializer`/`initargs` pass through to loky ✓
- Works with the default loky backend (processes)

**Cons:**
1. **Module-level mutable global state.** `_WORKER_FORECASTER` is process-global. If the
   loky pool is reused across calls (joblib caches worker pools by default with `reuse='auto'`),
   a previous forecaster could leak. Mitigation: use a unique key per call:
   ```python
   _WORKER_FORECASTERS = {}  # keyed by call_id
   ```
   But this adds memory management complexity (when to clean up?).

2. **`refit=True` breaks the pattern.** When refitting, each worker mutates the forecaster.
   Sharing via initializer doesn't help — each worker needs its own mutable copy. So the
   initializer only benefits `refit=False`, the same case that threads already solve
   perfectly (see 3.C).

3. **Incompatible with `prefer='threads'`.** If we switch to threads for `refit=False`
   (approach 3.C), the initializer is unnecessary — threads share memory anyway.

4. **Harder to test.** Module-level globals require careful cleanup in unit tests. Parallel
   tests with initializers are inherently non-deterministic.

5. **`differentiator` mutation.** During `predict()`, `differentiator.fit_transform()` mutates
   the differentiator's internal state. With loky (separate processes), each worker gets a
   copy via pickle so this is safe. But with the initializer, all folds in the same worker
   share the same forecaster object — sequential execution within each worker means they'd
   see each other's differentiator state. Fix: `deepcopy(forecaster)` in the worker function
   per fold, which defeats the purpose since deepcopy is essentially as expensive as
   unpickling.

   **UPDATE:** This has been fixed in `ForecasterRecursive` — `_create_predict_inputs()` now
   creates an internal `copy(self.differentiator)` and returns it, so `predict()` is
   side-effect-free. Once the same pattern is applied to all other forecasters (see Section 8),
   this concern disappears entirely. Until then, it remains a **correctness blocker for
   forecasters with `differentiation != None` that have NOT been updated.**

6. **`predict_bootstrapping` uses `random_state`.** The forecaster's `predict_bootstrapping`
   uses `random_state` internally. With shared state across folds in the same worker, the
   RNG state would be consumed by one fold and affect the next. This produces different
   results than sequential execution if folds share the same object.

   Fix: same as above — deepcopy per fold — which again cancels the benefit.

### 3.B. Shared Memory / Memory-Mapped Arrays

**Mechanism:** Before `Parallel`, extract the heavy arrays from the estimator (tree node
arrays in sklearn), write them to a memory-mapped file, and have workers read from the mmap.

**Assessment:** Not viable.

- sklearn tree structures are deeply nested Python objects (`DecisionTreeRegressor.tree_` has
  `.value`, `.feature`, `.threshold` etc. as numpy arrays, but the `Tree` object itself is a
  Cython extension type that can't be reconstructed from raw arrays without re-fitting).
- There is no public API to deserialize a fitted sklearn tree from raw arrays.
- Even if possible, the reconstructed model would need careful validation per sklearn version.
- **Rejected: too fragile, too much undocumented internals.**

### 3.C. Thread-Based Backend (`prefer='threads'`) — **Recommended**

**Mechanism:** For `refit=False` folds, use `Parallel(n_jobs=n_jobs, prefer='threads')`.
Threads share the process address space, so the forecaster and all data live in shared memory
with zero serialization cost.

**Measured performance:**
| | Loky (processes) | Threads | Sequential |
|---|---:|---:|---:|
| RF(100), 167 predict-only folds, n_jobs=4 | 61.23 s | 1.30 s | 0.87 s |

Threads are only 49% slower than sequential (GIL contention on some Python-level code) but
**47× faster than loky**. With heavier predict workloads (more steps, window features), the
GIL-free Cython portions dominate and the thread/sequential gap narrows further.

**Pros:**
1. **Zero IPC cost.** The forecaster, data, everything is shared. No pickle at all.
2. **Zero global state.** No module-level variables, no initializer cleanup.
3. **Trivial implementation.** One-line change: add `prefer='threads'` to the `Parallel` call.
4. **No correctness risk (once all forecasters are updated).** Each `delayed` call receives
   the forecaster by reference but creates local variables during execution. Sklearn
   `predict()` is read-only and thread-safe for tree models (the heavy lifting is in Cython
   with the GIL released). The differentiator side-effect has been eliminated in
   `ForecasterRecursive` (see Section 8 for remaining forecasters).
5. **Works with all estimator sizes.** Equally effective for RF (large pickle) and LGBM (small
   pickle).

**Cons and mitigations:**
1. **`differentiator` mutation during predict — RESOLVED for ForecasterRecursive.**
   Previously, `predict()` → `_create_predict_inputs()` called
   `self.differentiator.fit_transform()` which mutated the forecaster's internal state.
   With threads sharing the same forecaster object, concurrent calls would corrupt each
   other's state.

   **Fix applied:** `_create_predict_inputs()` now does:
   ```python
   differentiator = copy(self.differentiator)
   last_window_values = differentiator.fit_transform(last_window_values)
   ```
   and returns the local `differentiator` as a 5th tuple element. `predict()`,
   `predict_bootstrapping()`, and `_predict_interval_conformal()` use this local copy for
   `inverse_transform_next_window()`. `self.differentiator` is never mutated.

   **Remaining work:** The same fix must be applied to the other forecasters that use
   differentiation (see Section 8). Once complete, **no workaround is needed in
   `_fit_predict_forecaster`** — the forecaster's `predict()` is inherently thread-safe.

2. **`random_state` in `predict_bootstrapping`.**
   Bootstrapping uses `np.random.default_rng(random_state)` — creates a *new* RNG from the
   seed each time. Does NOT consume shared state from the forecaster object. **Safe.**

3. **GIL contention for non-Cython estimators.**
   Some estimators (e.g., custom Python-only regressors) may hold the GIL during predict.
   For these, threads provide no parallelism benefit (but also no harm — just sequential
   speed). The standard sklearn ensembles (RF, GBM, HGBM) and LightGBM all release the GIL.

4. **`refit=True` is incompatible.**
   `fit()` mutates the estimator heavily and is NOT thread-safe. Threads cannot be used for
   `refit=True` folds. But `refit=True` is rarely the bottleneck (fit dominates predict), and
   the IPC cost of sending the unfitted forecaster is small (only configuration, not tree data).
   **For `refit=True`, keep the current loky backend.**

---

## 4. Recommendation

### Use `prefer='threads'` for `refit=False`, keep loky for `refit=True`

This is the simplest, safest, and most effective approach. It eliminates the forecaster
serialization bottleneck entirely with a near-zero-risk change.

**Implementation sketch:**

```python
# In _backtesting_forecaster, before the Parallel call:

if refit:
    parallel_kwargs = {}  # default loky backend
else:
    parallel_kwargs = {'prefer': 'threads'}

backtest_predictions = Parallel(n_jobs=n_jobs, **parallel_kwargs)(
    delayed(_fit_predict_forecaster)(fold=fold, ...)
    for fold in folds
)
```

**No changes needed in `_fit_predict_forecaster`** — once all forecasters have the
side-effect-free predict fix (see Section 8), `predict()` does not mutate `self` and is
inherently thread-safe. No shallow copies, no deepcopies, no workarounds.

### Why NOT the worker initializer

| Factor | Initializer | Threads |
|---|---|---|
| IPC elimination | Partial (once per worker) | **Complete** (zero) |
| Implementation complexity | Medium (globals, cleanup, keying) | **Trivial** (one kwarg) |
| Correctness risk | High (shared mutable state across folds) | **None** (predict is side-effect-free) |
| `refit=True` support | No (workers need mutable copies) | No (keep loky) |
| Testability | Hard (process globals) | **Easy** (standard threading) |
| Works across all estimator types | Yes | Yes (GIL limits parallelism for pure-Python estimators) |
| Maintenance burden | Ongoing (loky pool reuse, version compat) | **None** |

The initializer approach is strictly inferior to threads for the `refit=False` case. The only
scenario where it would be better is if `predict()` held the GIL for the entire duration,
which doesn't happen for any standard sklearn or lightgbm estimator.

---

## 5. Combined Strategy (All Three Optimizations)

| Optimization | Applies to | Effect |
|---|---|---|
| **1. Pre-slice y and exog** | All folds (refit=True and False) | Eliminates full-series serialization per fold |
| **2. Module-level function** | All folds | Eliminates cloudpickle closure overhead |
| **3. `prefer='threads'` for `refit=False`** | `refit=False` folds only | Eliminates ALL serialization (forecaster, data, everything) |

**Note:** Optimization 3 (`prefer='threads'`) makes Optimization 1 (pre-slice) **redundant
for `refit=False`** — threads share memory, so no data is serialized regardless of size. But
Optimization 1 remains valuable for `refit=True` with loky, and as a defensive measure if the
threading backend is ever unavailable.

### Expected wall-clock improvement for `refit=False`, RF(100), 10K points, 167 folds, n_jobs=4

| Configuration | Measured / Estimated |
|---|---:|
| Current (loky, full data per fold) | **61.23 s** |
| Opt 1 only (loky, pre-sliced data) | ~40 s (estimator still serialized) |
| Opt 3 only (threads) | **~1.30 s** |
| Opt 1 + 3 (threads, pre-sliced) | **~1.30 s** (opt 1 is redundant) |

---

## 6. Edge Cases and Risks

### 6.1. `ForecasterRecursiveClassifier`

Uses `predict_proba()` → same thread-safety analysis as `predict()`. Sklearn classifiers'
`predict_proba()` also releases the GIL for tree-based models. **Safe.**

### 6.2. Transformers (`transformer_y`, `transformer_exog`)

`predict()` calls `transform_numpy(y, transformer)` and `transform_dataframe(exog, transformer)`.
These use sklearn transformers' `.transform()` method, which is typically stateless (no mutation
of the transformer). **Safe.**

Exception: `StandardScaler` with `partial_fit` — but `transform()` itself is read-only. **Safe.**

### 6.3. `window_features` (RollingFeatures)

`_recursive_predict` calls `wf.transform(window_data)` per step. `RollingFeatures.transform()`
computes rolling statistics and returns new arrays without modifying `self`. **Safe.**

### 6.4. `binner.transform()` for binned residuals

`predict_bootstrapping` calls `self.binner.transform(pred)` to select the residual bin.
`QuantileBinner.transform()` is stateless. **Safe.**

### 6.5. Mixed `refit` (integer)

When `refit` is an integer (intermittent), `n_jobs` is forced to 1. Threads vs loky is
irrelevant because there's no parallelism. **No impact.**

### 6.6. Non-sklearn estimators with non-thread-safe predict

Rare but possible. If a custom estimator's `predict()` modifies internal state or is not
thread-safe, threads could produce incorrect results. **Mitigation:** Document that
`prefer='threads'` assumes thread-safe `predict()`. Provide an escape hatch:
`n_jobs='auto'` could detect non-standard estimators and fall back to loky.

In practice, all standard sklearn estimators, LightGBM, XGBoost, and CatBoost have
thread-safe `predict()` methods.

---

## 7. Implementation Priority (Revised)

| Priority | Optimization | Effort | Risk | Impact |
|---|---|---|---|---|
| 1 | Pre-slice `y` and `exog` | Low | Very low | High for `refit=True` + large series |
| 2 | `prefer='threads'` for `refit=False` | **Very low** | **None** (predict is side-effect-free) | **Very high** — eliminates all IPC overhead |
| 3 | Module-level function | Low | Low | Marginal (code quality improvement) |

**Drop the worker initializer approach.** Threads are simpler, faster, and safer for the
same use case. The initializer adds complexity with no advantage over threads.

---

## 8. Side-Effect-Free Predict: Implementation Status

The main thread-safety concern identified in this review — `predict()` mutating
`self.differentiator` — has been addressed by making `_create_predict_inputs()` create an
internal `copy(self.differentiator)` and return it as part of its output tuple. The calling
methods (`predict`, `predict_bootstrapping`, `_predict_interval_conformal`) use this local
copy for `inverse_transform_next_window()`. `self.differentiator` is never mutated.

This change eliminates the need for any workaround in `_fit_predict_forecaster` when using
`prefer='threads'`.

### Status by forecaster

| Forecaster | Predict side-effect-free | Mutation site (predict path) |
|---|---|---|
| `ForecasterRecursive` | **Done** ✅ | `_create_predict_inputs()` uses `copy(self.differentiator)` |
| `ForecasterRecursiveMultiSeries` | **Pending** ❌ | `_create_predict_inputs()` L2192: `self.differentiator_[level].fit_transform(...)` |
| `ForecasterDirect` | **Pending** ❌ | `_create_predict_inputs()` L1535: `self.differentiator.fit_transform(...)` |
| `ForecasterDirectMultiVariate` | **Pending** ❌ | `_create_predict_inputs()` L1848: `self.differentiator_[series].fit_transform(...)` |

### Notes on multi-series forecasters

- `ForecasterRecursiveMultiSeries` uses `self.differentiator_` (a dict keyed by series name).
  Each entry is a `TimeSeriesDifferentiator` instance. The predict path also creates new
  entries for unknown levels (`self.differentiator_[level] = copy(...)` at L2189), which is an
  **additional mutation** beyond `fit_transform`. Both must be addressed.
- `ForecasterDirectMultiVariate` uses `self.differentiator_` (a dict keyed by series name),
  same pattern as the multiseries forecaster.
- `ForecasterDirect` uses a single `self.differentiator`, same pattern as `ForecasterRecursive`.

### What the fix looks like (ForecasterRecursive pattern)

```python
# In _create_predict_inputs():
if self.differentiation is not None:
    differentiator = copy(self.differentiator)  # local copy, self untouched
    last_window_values = differentiator.fit_transform(last_window_values)
else:
    differentiator = None

# Return differentiator as part of the tuple
return X_predict, exog_values, predict_index, steps_ahead, differentiator

# In predict(), predict_bootstrapping(), _predict_interval_conformal():
# Unpack and use the local differentiator instead of self.differentiator
X_predict, exog_values, predict_index, steps_ahead, differentiator = (
    self._create_predict_inputs(...)
)
if differentiator is not None:
    predictions = differentiator.inverse_transform_next_window(predictions)
```

### Thread-safety summary (once all forecasters are updated)

| Component | Thread-safe | Reason |
|---|---|---|
| `estimator.predict()` | ✅ Yes | sklearn/LGBM/XGB release GIL in Cython/C++ |
| `differentiator` | ✅ Yes | Each call uses local copy (the fix) |
| `transformer_y.transform()` | ✅ Yes | Read-only (no mutation) |
| `transformer_exog.transform()` | ✅ Yes | Read-only |
| `window_features.transform()` | ✅ Yes | Read-only |
| `binner.transform()` | ✅ Yes | Read-only |
| `random_state` (bootstrapping) | ✅ Yes | Creates new RNG each call |
| `set_cpu_gpu_device()` | ⚠️ No | Mutates estimator attrs (GPU only, edge case) |
