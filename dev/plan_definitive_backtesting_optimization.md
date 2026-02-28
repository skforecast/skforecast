# Definitive Plan: Backtesting Parallelization Optimization

## Status of Previous Plans

This document supersedes:

- `plan_preslice_backtesting_forecaster.md` — Pre-slice `y`/`exog` to reduce IPC
- `review_optimization3_forecaster_broadcast.md` — `prefer='threads'` for zero-IPC predict

Both plans were implemented, benchmarked with real backtesting workloads, and the results
diverged significantly from theoretical expectations. This plan incorporates all empirical
evidence collected during the implementation cycle.

---

## 1. Key Discovery: Why `prefer='threads'` Fails in Practice

### The review doc's benchmark was misleading

The review doc (Section 1) measured raw `estimator.predict(X)` in isolation:

```
RF(100), 167 predict-only folds, n_jobs=4:
  loky:       61.23 s
  threads:     1.30 s   ← 47× faster than loky
  sequential:  0.87 s
```

This benchmark called `rf.predict(X)` directly — where the GIL is released by Cython/C++
code and threads achieve true parallelism.

### The real backtesting pipeline holds the GIL

In practice, `_fit_predict_forecaster` calls `forecaster.predict(steps, last_window, exog)`,
which executes a **long chain of Python/pandas operations** before reaching `estimator.predict()`:

```
_fit_predict_forecaster(fold, forecaster, y, exog, ...)
  │
  ├── y.iloc[start:end]                    # pandas iloc         (GIL held)
  ├── exog.iloc[start:end]                 # pandas iloc         (GIL held)
  │
  └── forecaster.predict(steps, last_window, exog)
        ├── check_predict_input(...)        # validation          (GIL held)
        ├── transform_numpy(y, transformer) # sklearn transform   (GIL held)
        ├── transform_dataframe(exog, ...)  # sklearn transform   (GIL held)
        ├── _create_predict_inputs(...)     # feature engineering  (GIL held)
        │     ├── numpy slicing, rolling features
        │     ├── build X matrix (np.column_stack)
        │     └── return X, col_names, index, ...
        │
        ├── estimator.predict(X)            # C++/Cython          (GIL RELEASED ✓)
        │
        ├── inverse_transform predictions   # pandas/numpy        (GIL held)
        └── build result DataFrame          # pandas              (GIL held)
```

The GIL-free portion (`estimator.predict(X)`) is a **small fraction** of total per-fold
time. The Python overhead (validation, feature engineering, pandas DataFrame construction)
dominates. Threads serialize all of this through the GIL, resulting in performance **worse
than sequential** due to thread management overhead and GIL contention.

### Real benchmark results (full backtesting pipeline)

Machine: Windows, Python 3.11.8, 8 cores, `n_jobs=-1`, `refit=False`:

| Config | Sequential | Loky | Threads | Threads vs Loky | Threads vs Seq |
|---|---:|---:|---:|---:|---:|
| LR / N=10k | 0.135 s | 6.254 s | 0.240 s | 26.1× better | **0.56× (slower)** |
| HGBR / N=10k | 1.908 s | 3.571 s | 4.429 s | 0.81× (worse) | **0.43× (slower)** |
| RF / N=10k | 7.492 s | 32.615 s | 8.949 s | 3.6× better | **0.84× (slower)** |
| HGBR / N=50k | 9.537 s | 6.135 s | 23.206 s | 0.26× (much worse) | **0.41× (slower)** |
| HGBR / N=100k | 20.337 s | 15.812 s | 48.462 s | 0.33× (much worse) | **0.42× (slower)** |
| HGBR Direct / N=10k | 4.881 s | 23.122 s | 14.533 s | 1.6× better | **0.34× (slower)** |

**Conclusion: threads are slower than sequential in ALL tested configurations.**

For HGBR at N≥50k, threads are 2–2.5× *slower* than sequential. This is because multiple
threads competing for the GIL on heavy pandas operations creates contention that exceeds
any benefit from parallel `estimator.predict()`.

### When loky actually helps

Loky outperforms sequential only when:
1. The estimator's predict is compute-heavy (HGBR, not LR)
2. The dataset is large enough that per-fold compute dominates IPC overhead (N ≥ 50k)
3. The estimator's pickle is small (HGBR ≈ 3 MB, not RF ≈ 91 MB)

| Estimator | Pickle size | N=10k | N=50k | N=100k |
|---|---:|---:|---:|---:|
| LinearRegression | ~0.01 MB | 46× slower | — | — |
| HGBR(50 iters) | ~3 MB | 1.9× slower | **1.55× faster** | **1.29× faster** |
| RF(50 trees) | ~45 MB | 4.4× slower | — | — |

---

## 2. Pre-slice Effectiveness (Earlier Benchmark)

Pre-slicing `y`/`exog` in the main process before `Parallel` dispatch reduces the data
portion of IPC. This was implemented and benchmarked separately:

### `refit=False`: massive IPC reduction

| N | Full pickle (y+exog) | Pre-sliced pickle | Reduction | Serialization speedup |
|---:|---:|---:|---:|---:|
| 10k | 0.31 MB | 0.004 MB | **77×** | 1.4× |
| 50k | 1.53 MB | 0.004 MB | **383×** | 4.5× |
| 100k | 3.06 MB | 0.004 MB | **769×** | 28.3× |

Each fold receives only `last_window_y` (24 rows) + `exog_test` (12 rows) instead of the
full series. The data payload shrinks from megabytes to kilobytes.

### `refit=True`: negligible improvement

| N | Full pickle | Pre-sliced pickle | Reduction | Speedup |
|---:|---:|---:|---:|---:|
| 10k | 0.31 MB | 0.29 MB | 1.1× | 0.7× (slower) |
| 100k | 3.06 MB | 2.12 MB | 1.4× | 1.1× |

Training folds need most of the data anyway (expanding window). The savings are minimal
and sometimes negative (pre-slicing overhead exceeds IPC savings).

### Impact on wall-clock time

Pre-slicing eliminates the **data** IPC cost but NOT the **forecaster** IPC cost. For loky,
the forecaster is still pickled once per fold:

| Estimator | Forecaster pickle | Data pickle (N=100k) | Pre-sliced data | % of total IPC eliminated |
|---|---:|---:|---:|---:|
| HGBR | ~3 MB | 3.06 MB | 0.004 MB | **50%** |
| RF(50) | ~45 MB | 3.06 MB | 0.004 MB | **6%** |
| RF(100) | ~91 MB | 3.06 MB | 0.004 MB | **3%** |

For HGBR, pre-slicing eliminates roughly half the per-fold IPC. For RF, it barely matters
because the forecaster pickle dominates.

### Estimated wall-clock improvement with pre-slice

For HGBR at N=100k (where loky is already 1.29× faster than sequential):
- Current loky: 15.8 s (with full data IPC)
- With pre-slice: ~12–13 s estimated (eliminating ~50% of IPC)
- Sequential: 20.3 s
- Estimated speedup vs sequential: ~1.6× (up from 1.29×)

This is a **meaningful improvement** for compute-heavy estimators at scale.

---

## 3. Action Plan

### 3.1. Remove `prefer='threads'` ✅ IMMEDIATE

**File:** `skforecast/model_selection/_validation.py`, line 451

**Current code:**
```python
# Use threads for refit=False: predict() is read-only and thread-safe...
parallel_kwargs = {'prefer': 'threads'} if not refit else {}
```

**Change to:**
```python
parallel_kwargs = {}
```

**Rationale:** Benchmark proves threads are slower than sequential in ALL cases. The comment
about GIL release was correct for `estimator.predict()` in isolation but wrong for the full
`forecaster.predict()` pipeline which does extensive Python/pandas work.

### 3.2. Implement pre-slice `_prepare_fold_data` for loky — PRIORITY HIGH

Re-implement `_prepare_fold_data` (was implemented then reverted when we pivoted to threads).
This reduces the data portion of IPC when loky is used.

**New module-level function:**

```python
def _prepare_fold_data(
    folds: list,
    y: pd.Series,
    exog: pd.Series | pd.DataFrame | None
) -> list[dict]:
    """
    Pre-slice y and exog for each fold to minimize IPC serialization cost
    when using joblib's loky backend (n_jobs > 1).

    Each fold receives only the rows it actually needs instead of the full
    series. For refit=False folds this reduces per-fold data from N rows
    to window_size + steps rows (typically 100-1000× smaller).
    """
    fold_data = []
    for fold in folds:
        if fold[5] is False:
            data = {
                'last_window_y': y.iloc[fold[2][0]:fold[2][1]],
                'exog_test': (
                    exog.iloc[fold[3][0]:fold[3][1]] if exog is not None else None
                ),
                'y_train': None,
                'exog_train': None,
            }
        else:
            data = {
                'last_window_y': None,
                'exog_test': (
                    exog.iloc[fold[3][0]:fold[3][1]] if exog is not None else None
                ),
                'y_train': y.iloc[fold[1][0]:fold[1][1]],
                'exog_train': (
                    exog.iloc[fold[1][0]:fold[1][1]] if exog is not None else None
                ),
            }
        fold_data.append(data)
    return fold_data
```

**Updated `_fit_predict_forecaster` signature:**

```python
def _fit_predict_forecaster(
    fold,
    forecaster,
    y_train,           # Pre-sliced or None
    last_window_y,     # Pre-sliced or None
    exog_train,        # Pre-sliced or None
    exog_test,         # Pre-sliced or None
    store_in_sample_residuals,
    gap, interval, interval_method, n_boot,
    use_in_sample_residuals, use_binned_residuals,
    random_state, return_predictors, is_regression
) -> pd.DataFrame:
```

The function no longer receives the full `y` and `exog` — no iloc slicing inside the
worker. It receives only the pre-sliced data it needs.

**Updated Parallel dispatch:**

```python
fold_data_list = _prepare_fold_data(folds, y, exog)

if show_progress:
    fold_items = tqdm(list(zip(folds, fold_data_list)), total=len(folds))
else:
    fold_items = zip(folds, fold_data_list)

backtest_predictions = Parallel(n_jobs=n_jobs)(
    delayed(_fit_predict_forecaster)(
        fold=fold,
        forecaster=forecaster,
        y_train=fd['y_train'],
        last_window_y=fd['last_window_y'],
        exog_train=fd['exog_train'],
        exog_test=fd['exog_test'],
        store_in_sample_residuals=store_in_sample_residuals,
        ...
    )
    for fold, fd in fold_items
)
```

**Expected impact:**
- HGBR at N≥50k: ~20-30% wall-clock improvement (from eliminating ~50% of IPC)
- RF at any N: minimal improvement (forecaster pickle dominates)
- LR at any N: no improvement (n_jobs=1 from select_n_jobs_backtesting)

### 3.3. Keep module-level `_fit_predict_forecaster` ✅ ALREADY DONE

Already moved to module level in this session. Eliminates cloudpickle closure overhead.
No further changes needed.

### 3.4. Keep side-effect-free predict (differentiator copy) ✅ PARTIALLY DONE

The differentiator copy pattern in `_create_predict_inputs` ensures `predict()` doesn't
mutate `self.differentiator`. This is a **code correctness improvement** independent of
parallelization strategy:

- Prevents state corruption if `predict()` is called multiple times
- Makes the API contract clearer (predict is read-only)
- Prepares for future parallelization improvements

### Status:

| Forecaster | Status | Notes |
|---|---|---|
| `ForecasterRecursive` | **Done** ✅ | Commit `f87e5bf` |
| `ForecasterDirect` | **Done** ✅ | This session, 286 tests pass |
| `ForecasterRecursiveMultiSeries` | **Pending** ❌ | `self.differentiator_[level]` (dict) |
| `ForecasterDirectMultiVariate` | **Pending** ❌ | `self.differentiator_[series]` (dict) |

The multi-series forecasters use a dict of differentiators keyed by series name. The fix
is the same pattern: `copy(self.differentiator_[key])` → use local copy → return in tuple.

**Priority: MEDIUM.** These are code quality improvements. They do NOT enable threads
(threads are proven slower), but they prevent subtle bugs with repeated predict calls.

### 3.5. Do NOT implement worker initializer — REJECTED

The worker initializer approach (Section 3.A of the review doc) is the only technique that
could solve the **forecaster serialization** bottleneck for large sklearn models like RF.
However, it is rejected for the following reasons:

1. **Complexity:** Module-level mutable globals, worker pool reuse management, cleanup
2. **Correctness risk:** Shared state across folds in the same worker (bootstrapping RNG,
   binner state, etc.)
3. **Limited benefit:** RF with N=10k is 7.5s sequential. Even if loky could parallelize it,
   the gain would be ~3-4s. Not worth the complexity.
4. **The real bottleneck for RF is `estimator.predict()` time, not IPC:**
   At N=10k with 24 lags, predict is ~4 ms/fold × 250 folds ≈ 1s. The rest is Python
   overhead and IPC. Even with zero IPC, the Python overhead serialized through the GIL
   (threads) or duplicated in processes (loky) limits the achievable speedup.

### 3.6. Future: Reduce Python overhead in `predict()` — NOT IN SCOPE

The root cause of poor parallelization is that `forecaster.predict()` spends most of its
time in Python/pandas code (validation, feature engineering, DataFrame construction). The
ultimate fix is to optimize this code path:

- Replace pandas iloc with numpy array operations
- Cache or skip redundant input validation (`check_inputs=False` already exists)
- Use numpy arrays instead of DataFrames internally during predict
- Move feature engineering to Cython/numba

This is a **major refactor** (affects all forecasters) and is out of scope for this
optimization round. Documenting it here for future reference.

---

## 4. Revised `select_n_jobs_backtesting` — CONSIDER

The current heuristic assigns `n_jobs = cpu_count() - 1` for non-linear estimators. Our
benchmark shows this is often counterproductive:

| Estimator | N=10k | N=50k | N=100k |
|---|---|---|---|
| HGBR | 1.9× **slower** | 1.55× faster | 1.29× faster |
| RF(50 trees) | 4.4× **slower** | — | — |

**Possible improvements:**

1. **Minimum dataset size threshold:** Only parallelize when N exceeds a threshold (e.g.,
   30k). Below that, sequential is faster due to process creation + IPC overhead.

2. **Estimator pickle size awareness:** For large-pickle estimators (RF, GBM with many
   trees), the IPC cost per fold is too high. Could check `n_estimators` or do a trial
   pickle to estimate size.

3. **Adaptive approach:** Run the first fold sequentially, measure wall-clock time. If it's
   below a threshold (e.g., 50ms), keep n_jobs=1. Otherwise, use n_jobs=cpu_count()-1.

**Decision: DEFERRED.** The current heuristic works well enough. Pre-slicing (3.2) will
improve the cases where loky is borderline beneficial. Changing the heuristic risks breaking
users who have tuned their workflows around the current behavior.

---

## 5. Summary: What to Implement

| # | Action | Effort | Impact | Status |
|---|---|---|---|---|
| 1 | Remove `prefer='threads'` | 2 min | Fixes regression (threads slower than seq) | **DO NOW** |
| 2 | Pre-slice `_prepare_fold_data` | 2 hours | ~20-30% loky speedup for HGBR at N≥50k | **DO NEXT** |
| 3 | Module-level function | Done | Eliminates cloudpickle overhead | ✅ |
| 4 | Differentiator copy (Direct) | Done | Code correctness | ✅ |
| 5 | Differentiator copy (MultiSeries) | 2 hours | Code correctness | **PENDING** |
| 6 | Differentiator copy (DirectMultiVariate) | 1 hour | Code correctness | **PENDING** |
| 7 | Worker initializer | — | — | **REJECTED** |
| 8 | Optimize predict() Python overhead | Many days | Would enable effective parallelization | **FUTURE** |

### Implementation order:

1. **Remove `prefer='threads'`** — immediate, prevents performance regression
2. **Pre-slice `_prepare_fold_data`** — reduces loky IPC, helps the cases where loky works
3. **Differentiator copy for MultiSeries/DirectMultiVariate** — code correctness, no urgency
4. **(Future) Optimize predict overhead** — biggest potential gain, biggest effort

---

## 6. Lessons Learned

1. **Micro-benchmarks lie.** Measuring `estimator.predict()` in isolation showed threads as
   47× faster than loky. Real backtesting showed threads **slower than sequential**. Always
   benchmark the full pipeline.

2. **Python overhead dominates.** For moderate dataset sizes (N ≤ 50k), the Python code
   around `estimator.predict()` (validation, feature engineering, pandas operations) takes
   more time than the actual prediction. This makes threading ineffective (GIL).

3. **Loky has a high fixed cost.** Process creation, pool management, and per-fold
   serialization create a significant overhead. Loky only pays off when per-fold compute
   is substantial (HGBR at N ≥ 50k).

4. **Pre-slicing helps loky but can't fix threads.** Reducing data IPC is valuable for loky
   but irrelevant for threads (threads share memory anyway). Since threads don't work for
   our use case, pre-slicing is the main lever for improving loky performance.

5. **The GIL is the fundamental bottleneck.** Until skforecast's predict pipeline is
   rewritten to minimize Python overhead (numpy-only, no pandas in the hot path), neither
   threads nor processes will achieve good parallel scaling.

---

## Appendix: Benchmark Environments

### Real backtesting benchmark (`benchmark_loky_vs_threads.py`)

- OS: Windows
- Python: 3.11.8
- CPU: 8 cores
- joblib: 1.5.3
- sklearn: latest
- `n_jobs=-1` (all cores)
- `refit=False`, `fixed_train_size=True`
- `steps=12`, `lags=24`
- Configurations: LR, HGBR, RF at N=10k/50k/100k

### Pre-slice benchmark (`benchmark_preslice_backtesting.py`)

- Same environment
- Measured pickle size and serialization time with/without pre-slicing
- `refit=False` and `refit=True` scenarios
- N=10k, 50k, 100k with 5 exog columns
