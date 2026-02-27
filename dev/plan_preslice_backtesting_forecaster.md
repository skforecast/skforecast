# Implementation Plan: Pre-slice `y` and `exog` in `backtesting_forecaster`

## 1. Motivation

In `_backtesting_forecaster`, the full `y` Series and `exog` DataFrame are passed as kwargs
to every fold inside the `Parallel` call. When `n_jobs > 1` (loky backend), joblib pickles
these objects **once per fold**, even though each worker only uses a small slice.

The goal is to pre-slice the data in the main process so that each worker receives only
the rows it actually needs, dramatically reducing IPC serialization cost.

---

## 2. When Does This Matter? (Realistic Estimator Analysis)

### When parallelism is active (`n_jobs > 1`)

From `select_n_jobs_backtesting`, the cases where `n_jobs = cpu_count() - 1`:

| Forecaster | Estimator | refit | n_jobs |
|---|---|---|---|
| `ForecasterRecursive` | RF, GBM, etc. (non-linear) | `True` or `False` | `cpu_count()-1` |
| `ForecasterRecursive` | `LGBMRegressor(n_jobs=1)` | `True` or `False` | `cpu_count()-1` |
| `ForecasterRecursiveMultiSeries` | RF, GBM, `LGBMRegressor(n_jobs=1)` | any | `cpu_count()-1` |

**NOT parallelized (n_jobs=1):**

| Forecaster | Estimator | Why |
|---|---|---|
| `ForecasterRecursive` | `LGBMRegressor()` (default n_jobs=-1) | Internal LGBM parallelism conflicts |
| `ForecasterRecursive` | Linear models | Too fast to benefit |
| `ForecasterDirect` / `ForecasterDirectMultiVariate` | any | Parallelization applied during fitting |
| `ForecasterStats` / `ForecasterEquivalentDate` | — | Sequential by design |

### Key insight: the typical `LGBMRegressor()` user gets `n_jobs=1`

Users of `LGBMRegressor()` with default `n_jobs=-1` never trigger backtesting
parallelism. Pre-slicing has **zero effect** for them. Only users who explicitly set
`LGBMRegressor(n_jobs=1)`, or use `RandomForestRegressor`, `GradientBoostingRegressor`,
`HistGradientBoostingRegressor`, etc., benefit.

### Per-scenario breakdown

#### Scenario A: `refit=False` + non-linear estimator → **biggest win**

- Worker path: `fold[5] is False`
- Data accessed per fold:
  - `y.iloc[last_window_start:last_window_end]` → **window_size rows**
  - `exog.iloc[test_start:test_end]` → **steps rows**
- Currently sent: **entire `y`** (N rows) + **entire `exog`** (N rows)
- Savings: N → window_size + steps per fold
- Example: N=100,000, window_size=168, steps=24 → **520× less data per fold**

This is the most impactful case because:
1. `predict()` for tree models is fast (ms), so IPC overhead is proportionally large
2. The forecaster is also serialized per fold (separate optimization, not covered here)

#### Scenario B: `refit=True` + non-linear estimator → **modest IPC savings, dominated by fit cost**

- Worker path: `fold[5] is True`
- Data accessed per fold:
  - `y.iloc[train_start:train_end]` → up to initial_train_size rows (large)
  - `exog.iloc[train_start:train_end]` → same range
  - `exog.iloc[test_start:test_end]` → steps rows
- Currently sent: entire `y` + entire `exog`
- Savings with `fixed_train_size=True`:
  - Avoids sending data before `train_start` and after `test_end`
  - Typical: 10-30% reduction in data volume
- Savings with `fixed_train_size=False` (expanding):
  - Early folds: significant savings (small training set)
  - Late folds: minimal savings (training set ≈ full series)

**However**, with `refit=True`, the dominant cost per fold is `estimator.fit()` (seconds for
LGBM/RF on large data), not IPC. The IPC savings are real but **barely visible in wall-clock 
time**.

#### Summary of expected impact

| Scenario | n_jobs | IPC gain | Wall-clock impact |
|---|---|---|---|
| `refit=False` + RF/GBM | `cpu_count()-1` | **Massive** (100-500×) | **High** — IPC is dominant cost |
| `refit=True` + RF/GBM | `cpu_count()-1` | Modest (10-30%) | **Low** — fit dominates |
| `refit=False` + LGBM(default) | `1` | None | None — no parallelism |
| `refit=True` + LGBM(default) | `1` | None | None — no parallelism |
| `refit=False` + LGBM(n_jobs=1) | `cpu_count()-1` | **Massive** | **High** |

---

## 3. Current Code Structure

```
_backtesting_forecaster(forecaster, y, cv, metric, exog, ...)
│
├── forecaster = deepcopy(forecaster)
├── folds = cv.split(X=y, as_pandas=False)
├── forecaster.fit(y[:initial_train_size], ...)       # Initial fit
│
├── def _fit_predict_forecaster(fold, forecaster, y, exog, ...):
│   │   # Defined INSIDE _backtesting_forecaster (closure)
│   ├── if fold[5] is False:
│   │       last_window_y = y.iloc[lw_start:lw_end]          # ← uses slice of y
│   ├── else:
│   │       y_train = y.iloc[train_start:train_end]           # ← uses slice of y
│   │       exog_train = exog.iloc[train_start:train_end]     # ← uses slice of exog
│   │       forecaster.fit(y=y_train, exog=exog_train, ...)
│   ├── next_window_exog = exog.iloc[test_start:test_end]     # ← uses slice of exog
│   └── return pred
│
├── kwargs = {"forecaster": forecaster, "y": y, "exog": exog, ...}  # ← full data
├── Parallel(n_jobs)(delayed(_fit_predict)(fold=f, **kwargs) for f in folds)
│
├── # Post-processing: metrics, fold labels, etc.
└── return metric_values, backtest_predictions
```

### Fold structure (from `TimeSeriesFold.split`)

Each fold is a list:
```
[fold_number, [train_start, train_end], [lw_start, lw_end], [test_start, test_end], [test_gap_start, test_gap_end], fit_forecaster]
 fold[0]       fold[1]                   fold[2]              fold[3]                 fold[4]                        fold[5]
```

---

## 4. Proposed Changes

### 4.1. Move `_fit_predict_forecaster` to module level

Move the function from inside `_backtesting_forecaster` to module level. This:
- Eliminates cloudpickle closure overhead (joblib serializes by import path)
- Improves testability
- Makes the pre-slicing change cleaner (no closure over `y`/`exog`)

The function already receives all its inputs via arguments (not via closure), so this is
purely a structural move.

**New signature:**
```python
def _fit_predict_forecaster(
    fold,
    y_train,              # Pre-sliced training data (or None if refit=False) 
    last_window_y,        # Pre-sliced last window (or None if refit=True)
    exog_train,           # Pre-sliced training exog (or None)
    exog_test,            # Pre-sliced test exog (or None)
    forecaster,
    store_in_sample_residuals,
    gap,
    interval,
    interval_method,
    n_boot,
    use_in_sample_residuals,
    use_binned_residuals,
    random_state,
    return_predictors,
    is_regression
) -> pd.DataFrame:
```

The function no longer receives `y` and `exog` — it receives only the pre-sliced data
it actually needs.

### 4.2. Pre-slice in the main process

Before the `Parallel` call, build per-fold data tuples:

```python
def _prepare_fold_data(folds, y, exog):
    """
    Pre-slice y and exog for each fold to minimize IPC serialization cost.
    Returns a list of dicts, one per fold, with only the data each fold needs.
    """
    fold_data = []
    for fold in folds:
        if fold[5] is False:
            # No refit: worker only needs last_window + test exog
            data = {
                'y_train': None,
                'last_window_y': y.iloc[fold[2][0]:fold[2][1]],
                'exog_train': None,
                'exog_test': exog.iloc[fold[3][0]:fold[3][1]] if exog is not None else None,
            }
        else:
            # Refit: worker needs training data + test exog
            data = {
                'y_train': y.iloc[fold[1][0]:fold[1][1]],
                'last_window_y': None,
                'exog_train': (
                    exog.iloc[fold[1][0]:fold[1][1]] if exog is not None else None
                ),
                'exog_test': exog.iloc[fold[3][0]:fold[3][1]] if exog is not None else None,
            }
        fold_data.append(data)
    return fold_data
```

### 4.3. Update the `Parallel` dispatch

```python
fold_data_list = _prepare_fold_data(folds, y, exog)

# Wrap folds + data together for tqdm
fold_items = list(zip(folds, fold_data_list))
if show_progress:
    fold_items = tqdm(fold_items)

kwargs_fit_predict = {
    "forecaster": forecaster,
    "store_in_sample_residuals": store_in_sample_residuals,
    "gap": gap,
    "interval": interval,
    "interval_method": interval_method,
    "n_boot": n_boot,
    "use_in_sample_residuals": use_in_sample_residuals,
    "use_binned_residuals": use_binned_residuals,
    "random_state": random_state,
    "return_predictors": return_predictors,
    "is_regression": is_regression,
}

backtest_predictions = Parallel(n_jobs=n_jobs)(
    delayed(_fit_predict_forecaster)(
        fold=fold,
        y_train=fold_data['y_train'],
        last_window_y=fold_data['last_window_y'],
        exog_train=fold_data['exog_train'],
        exog_test=fold_data['exog_test'],
        **kwargs_fit_predict
    )
    for fold, fold_data in fold_items
)
```

Note: `y` and `exog` are **no longer present** in the kwargs. The `forecaster` is still sent
per fold (that is a separate optimization — not in scope here).

### 4.4. Update `_fit_predict_forecaster` body

```python
def _fit_predict_forecaster(
    fold, y_train, last_window_y, exog_train, exog_test,
    forecaster, store_in_sample_residuals, gap, interval, 
    interval_method, n_boot, use_in_sample_residuals, use_binned_residuals, 
    random_state, return_predictors, is_regression
) -> pd.DataFrame:

    if fold[5] is False:
        # Not refitting — use pre-sliced last_window_y directly
        pass  # last_window_y already set
    else:
        # Refit on pre-sliced training data
        forecaster.fit(
            y=y_train, 
            exog=exog_train, 
            store_in_sample_residuals=store_in_sample_residuals
        )
        last_window_y = None

    next_window_exog = exog_test

    # Compute steps from the exog_test or from the fold indices
    test_iloc_start = fold[3][0]
    test_iloc_end   = fold[3][1]
    steps = test_iloc_end - test_iloc_start

    # ... rest of the function remains the same, using last_window_y and next_window_exog ...
```

---

## 5. Edge Cases to Handle

### 5.1. `ForecasterDirect` with `gap > 0`

Uses `fold[4]` (test_no_gap indices) to compute steps. This is metadata from the fold 
list, not data slicing. No impact from pre-slicing — `fold` is still passed intact.

### 5.2. `gap > 0` for non-ForecasterDirect

Handled by `pred = pred.iloc[gap:]` after prediction. The `exog_test` slice already 
covers `test_start:test_end` which includes the gap. No change needed.

### 5.3. `initial_train_size is None` (externally fitted forecaster)

When `initial_train_size` is `None`, no initial fit is done and `folds[0][5]` is set to 
`False` by the split method. The pre-slice path for `fold[5] is False` applies: only
`last_window_y` and `exog_test` are sent. Works correctly.

### 5.4. `interval` with `method='conformal'`

Uses the same `last_window_y` and `next_window_exog` as point predictions. No impact.

### 5.5. `return_predictors=True`

`create_predict_X` also uses `last_window_y` and `next_window_exog`. No impact.

### 5.6. `ForecasterRecursiveClassifier`

Uses `predict_proba` with same `steps`, `last_window`, and `exog` args. No impact.

### 5.7. `tqdm` progress bar

Currently `tqdm` wraps the raw folds list. After the change, it wraps
`zip(folds, fold_data_list)`. Each iteration yields a `(fold, fold_data)` tuple.
The progress bar tracks iterations, not the tuple contents, so it works identically.

### 5.8. `fold_labels` post-processing

After the `Parallel` call, the code builds `fold_labels` from the original `folds` list.
This is separate from `fold_items` and is not affected.

### 5.9. `train_indexes` for metric computation

After the `Parallel` call, `y.iloc[train_indexes]` is still called using the full `y` 
in the main process (not in workers). This is correct and unaffected.

---

## 6. Implementation Steps

### Step 1: Write `_fit_predict_forecaster` at module level

- Move the function from inside `_backtesting_forecaster` to module level (above it).
- Change signature to accept `y_train`, `last_window_y`, `exog_train`, `exog_test` 
  instead of `y` and `exog`.
- Simplify body: remove iloc slicing (data arrives pre-sliced).
- Compute `steps` from `fold[3]` indices instead of from the data.

### Step 2: Write `_prepare_fold_data` at module level

- New helper function that takes `folds`, `y`, `exog` and returns a list of dicts.
- One dict per fold with keys: `y_train`, `last_window_y`, `exog_train`, `exog_test`.

### Step 3: Update `_backtesting_forecaster`

- Remove the inline `_fit_predict_forecaster` definition.
- Add call to `_prepare_fold_data` before the `Parallel` block.
- Remove `y` and `exog` from `kwargs_fit_predict_forecaster`.
- Update `Parallel` dispatch to iterate over `zip(folds, fold_data_list)`.
- Adapt `tqdm` wrapping to the new iterable.

### Step 4: Run existing tests

```bash
pytest skforecast/model_selection/tests/ -x -q
```

All existing backtesting tests must pass without modification. The change is purely 
internal — no public API change.

### Step 5: Add targeted test for data size (optional but recommended)

Write a test that patches `Parallel` to capture the actual arguments sent to each fold 
and asserts that `y_train` / `last_window_y` have the expected sizes:

```python
def test_preslice_refit_false_sends_only_window_size():
    """When refit=False, each fold should receive last_window_y of size window_size."""
    # Setup: n=10_000, window_size=24, steps=30, refit=False
    # Assert: all fold_data['last_window_y'] have len == 24
    # Assert: all fold_data['exog_test'] have len <= 30
    # Assert: all fold_data['y_train'] is None
    ...
```

---

## 7. What This Plan Does NOT Cover

- **Forecaster serialization per fold**: The fitted forecaster is still pickled once per
  fold. This requires a different optimization (worker initializer or thread-based backend)
  and is out of scope.

- **`_backtesting_forecaster_multiseries`**: The multi-series path has a similar structure
  but different data shapes (dict of series, different slicing logic). A separate plan
  should address it.

- **`_backtesting_stats`**: Uses a different fold processing function. Not affected.

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Pandas index mismatch after slicing | Very low | High | Pre-sliced data retains DatetimeIndex; forecaster uses it for prediction dates. Covered by existing tests. |
| Memory increase from pre-slicing all folds upfront | Very low | Low | `.iloc` on pandas returns views for contiguous slices, not copies. Even if copies are materialized, each slice is tiny for `refit=False`. |
| `refit=True` pre-slice for large expanding window | Low | Low | Training slices are large anyway. Pre-slicing avoids only the tail beyond `test_end`. Net memory is neutral or positive. |
| Breaks `ForecasterDirect` gap handling | Very low | Medium | `fold[4]` is still passed intact. Only `test_iloc_start/end` from `fold[3]` are used for slicing, which is the same as before. Covered by existing tests. |
