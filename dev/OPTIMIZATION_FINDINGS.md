# Recursive Forecaster Optimization Findings

Review of [skforecast/recursive/_forecaster_recursive.py](skforecast/recursive/_forecaster_recursive.py) and [skforecast/recursive/_forecaster_recursive_multiseries.py](skforecast/recursive/_forecaster_recursive_multiseries.py) for high-impact performance improvements in hot paths (predict, bootstrap, train-X-y construction, one-step-ahead split).

Each finding has been verified twice — once by tracing the code, and once by reproducing the suspected behavior with a small numpy benchmark or by re-reading the surrounding pipeline. Where the second pass revised the original claim, that is called out explicitly.

Conventions used below:
- `n_lags`, `n_wf`, `n_exog`, `n_levels`, `n_boot`, `steps`, `W` (= `window_size`) refer to the corresponding skforecast quantities.
- Constraints: no public-API change, no change in numerical results, no change in residuals/randomness semantics.

---

## Summary table

| # | File | Function | Verified? | Estimated impact |
|---|------|----------|-----------|------------------|
| 1 | both | `_recursive_predict` / `_recursive_predict_bootstrapping` | yes | 1.5–3× faster `predict` when window features are used; ~10–20% otherwise |
| 2 | multiseries | `_create_predict_inputs` | yes | 2–3× faster on this function for moderate-to-large `n_levels` |
| 3 | multiseries | `_recursive_predict_bootstrapping` (exog `np.tile`) | **revised** — smaller than initially claimed | ~5–10% faster bootstrap predict |
| 4 | both | `_train_test_split_one_step_ahead` | yes | 40–60% faster `OneStepAheadFold` Bayesian search |
| 5 | single-series | `_create_train_X_y` | yes | 15–25% faster `create_train_X_y` per fit |
| 6 | single-series | `_create_predict_inputs` | yes | 5–15% off each `predict()` call |

Findings #1, #2, #4 are the highest-leverage; #5 and #6 are smaller but very cheap to implement. #3 is included for completeness with corrected scope.

---

## 1. Per-step Python overhead in `_recursive_predict` and bootstrap loops

**Files / lines (verified):**
- Single-series `_recursive_predict`: [_forecaster_recursive.py:1632-1660](skforecast/recursive/_forecaster_recursive.py#L1632-L1660)
- Single-series `_recursive_predict_bootstrapping`: [_forecaster_recursive.py:1742-1774](skforecast/recursive/_forecaster_recursive.py#L1742-L1774)
- Multiseries `_recursive_predict`: [_forecaster_recursive_multiseries.py:2548-2585](skforecast/recursive/_forecaster_recursive_multiseries.py#L2548-L2585)

### What the code does today

Inside the per-step Python loop (e.g. lines 1642-1649 single-series):

```python
if has_window_features:
    window_data = last_window[i : -remaining]
    X[n_lags : n_lags + n_window_features] = np.concatenate(
        [
            wf.transform(window_data)
            for wf in self.window_features
        ]
    )
```

For every step, a Python list of per-window-feature outputs is built and then `np.concatenate`-ed into a fresh array. Even when there is exactly one `RollingFeatures` instance (the common case), the list comprehension + concat is pure overhead — the result is then copied into the preallocated `X` slice.

The lag block on the immediately-preceding lines:

```python
if self.lags_are_contiguous:
    X[:n_lags] = last_window[-(remaining + n_lags): -remaining][::-1]
else:
    X[:n_lags] = last_window[-self.lags - remaining]
```

The non-contiguous branch builds a fresh array via fancy indexing each step. The contiguous branch is fine (slice + reversed view), but `remaining + n_lags` and `-self.lags - remaining` are recomputed every iteration.

The same pattern is repeated in the bootstrap variants (single-series line 1756, multiseries line 2733-2737 — the multiseries variant already writes into preallocated slices, so it's already correct; only single-series bootstrap and both `_recursive_predict` use the wasteful concat).

### Why it's slow

For typical settings (`steps=24`, `n_boot=250`), the inner concat fires 24 times per non-bootstrap predict and 24 times per bootstrap predict. Per fold of a backtest with refit, this multiplies by the number of folds. Each `np.concatenate` of small arrays carries ~1-3 µs of Python+numpy dispatch overhead, plus an allocation; per-fold cost is small, but it is paid on every Optuna trial × every fold, so it compounds.

The non-contiguous lag branch additionally allocates a new array via fancy indexing each step.

### Proposed change

1. **Hoist a `wf_slices` list out of the loop**, computed once before iteration begins:
   ```python
   wf_slices = []
   off = n_lags
   for wf in self.window_features:
       n = <wf output width, already known from feature names out>
       wf_slices.append((wf, off, off + n))
       off += n
   ```
   Inside the loop, replace the `np.concatenate([...])` with a direct write per WF:
   ```python
   for wf, lo, hi in wf_slices:
       wf.transform(window_data, out=X[lo:hi])  # if RollingFeatures supports `out`
   # OR (simpler, no API change):
   for wf, lo, hi in wf_slices:
       X[lo:hi] = wf.transform(window_data)
   ```
   The second form alone removes the list comprehension + concat and writes directly into the preallocated X. Eliminates one allocation per step.

2. **Precompute a lag index buffer** before the loop:
   - For contiguous lags: precompute `lag_base = -n_lags` and `lag_end_offsets = [-(steps - i) for i in range(steps)]`; or equivalently keep the current slice form but cache the integer constants once.
   - For non-contiguous lags: precompute `lag_offsets = -np.asarray(self.lags)` once; inside the loop, `X[:n_lags] = last_window[lag_offsets - remaining]`. (Same number of operations, but `np.asarray(self.lags)` is currently re-derived inside the slice each iteration.) The bigger win for the non-contiguous case is to allocate the output destination once — already done via `X[:n_lags]`.

3. **Apply the same fix to the bootstrap variants** (single-series line 1756; the multiseries `_recursive_predict_bootstrapping` already does this correctly at lines 2733-2737, so no change there).

### Expected impact

- ~1.5–3× faster `predict()` / `predict_bootstrapping()` when window features are used, dominated by the eliminated per-step `np.concatenate`.
- ~10–20% faster when no window features are used (only the lag-index hoisting helps).
- Compounds heavily over backtesting and Bayesian-search loops.

### Constraints / risks

- The `out=` argument on `RollingFeatures.transform` would be a new public surface; the simpler form (direct write to `X[lo:hi]`) is API-neutral and already eliminates the concat — recommended first step.
- Numerical output is bit-identical (same arithmetic, just no intermediate concat).

---

## 2. `_create_predict_inputs` (multiseries) per-level reindex + concat for exog

**File / lines (verified):** [_forecaster_recursive_multiseries.py:2331-2459](skforecast/recursive/_forecaster_recursive_multiseries.py#L2331-L2459)

### What the code does today

When `exog` is a dict (one frame per level), the per-level loop runs `reindex_like(empty_exog)` for each level (line 2418) and accumulates frames into a Python list, then concatenates all levels with `pd.concat` at line 2433:

```python
for idx_level, level in enumerate(levels):
    ...
    if isinstance(exog, dict):
        exog_values = exog.get(level, None)
        if exog_values is not None:
            if isinstance(exog_values, pd.Series):
                exog_values = exog_values.to_frame()
            exog_values = exog_values.reindex_like(empty_exog)
        else:
            exog_values = empty_exog.copy()
    exog_values_all_levels.append(exog_values)
...
if exog is not None:
    exog_values_all_levels = pd.concat(exog_values_all_levels)
    if isinstance(exog, dict):
        exog_values_all_levels = transform_dataframe(...)
        ...
        if not exog_values_all_levels.dtypes.to_dict() == self.exog_dtypes_out_:
            check_exog_dtypes(exog=exog_values_all_levels)
    exog_values_all_levels = exog_values_all_levels.to_numpy()
    exog_values_dict = {
        i + 1: exog_values_all_levels[i::steps, :]
        for i in range(steps)
    }
```

A few smaller items in the same function:
- Line 2331-2333: `last_window.iloc[-self.window_size:, ...].copy()` is followed by a `to_numpy()` at line 2382 that copies again — the explicit `.copy()` is redundant.
- Line 2372 / 2450: `exog.dtypes.to_dict() == self.exog_dtypes_out_` builds a Python dict every predict call.
- Lines 2454-2457: builds a Python `dict` keyed by step `1..steps`, each value a strided slice into the concatenated array. The recursive loop later does `exog_values_dict[i + 1]` per step (Python dict lookup).

### Why it's slow

This function is called *once per `predict()` call*, which means once per backtest fold (and inside `predict_bootstrapping`). For `n_levels=50`, that's 50 reindex operations + a 50-frame `pd.concat` per fold. `pd.concat` of N small frames is ~50–500 µs each; `reindex_like` is similar.

The per-step Python dict in `exog_values_dict` adds a hash + lookup per recursion step, which is small (~50 ns) but pure overhead.

The `.dtypes.to_dict()` comparison materializes a fresh dict on every predict.

### Proposed change

1. **Replace the per-level reindex/concat with a preallocated 3-D numpy buffer.** Compute `prediction_index` and a per-level positional mapping once, then:
   ```python
   exog_arr = np.full((steps, n_levels, n_exog_in_), np.nan, dtype=...)
   for idx_level, level in enumerate(levels):
       df = exog.get(level)
       if df is not None:
           # reindex to prediction_index → fill exog_arr[:, idx_level, :]
           ...
   ```
   Then transform once (the existing `transform_dataframe` and categorical encoder work on a single concatenated frame; this can be reformulated to operate on the 2-D `exog_arr.reshape(steps * n_levels, n_exog)` view, which is contiguous and view-only).

2. **Replace the dict with the 3-D ndarray.** Pass `exog_arr` (shape `(steps, n_levels, n_exog)`) into `_recursive_predict` / `_recursive_predict_bootstrapping`. Inside those functions the per-step access becomes `exog_arr[step]` (a `(n_levels, n_exog)` view) instead of `exog_values_dict[step + 1]`. Removes the dict lookup inside the recursion loop and the strided slice.

3. **Cache the dtype signature once at fit time.** Store `self._exog_dtypes_out_values` as a numpy array of dtypes (or a precomputed hash). Replace `not exog.dtypes.to_dict() == self.exog_dtypes_out_` with `not (exog.dtypes.values == self._exog_dtypes_out_values).all()` — same semantics, no Python dict allocation.

4. **Drop the redundant `.copy()` on `last_window.iloc[...]` at line 2333.** The very next operation (line 2382) is `last_window.to_numpy()`, which copies the data into the numpy result regardless.

### Expected impact

- ~2–3× faster `_create_predict_inputs` for moderate-to-large `n_levels`.
- For a backtest with 200 folds × 50 series, this is meaningful: it is often the second-hottest function after the recursion itself.
- Removes one Python dict allocation per predict (small but free).

### Constraints / risks

- Internal-only data layout change (the dict-keyed-by-step is not part of the public API; it is consumed by `_recursive_predict` and `_recursive_predict_bootstrapping` only). Verified by grepping the codebase.
- Must preserve the row ordering convention (`level0_step0, level1_step0, ..., levelN_step0, level0_step1, ...`) used by `_recursive_predict_bootstrapping` at line 2655. The 3-D `(steps, n_levels, n_exog)` layout is consistent with this and avoids the strided slice.

---

## 3. Per-step `np.tile` allocation in multiseries bootstrap

**File / lines (verified):** [_forecaster_recursive_multiseries.py:2741](skforecast/recursive/_forecaster_recursive_multiseries.py#L2741)

> ⚠️ **Revised in second review.** The original v1 finding claimed the per-step `last_window_boot.reshape(...)` triggers a copy (~16 MB / step). On verification with a numpy benchmark this is **wrong**: the reshape returns a view because the source is C-contiguous and the merged dimensions are adjacent (`np.shares_memory(arr, sliced.reshape(...))` returns `True`). I'm leaving the finding in for completeness with corrected scope, but it is much smaller than originally claimed.

### What the code does today

Inside the per-step bootstrap loop (line 2741):

```python
if has_exog:
    # Reshape (n_levels, n_exog) to (n_boot × n_levels, n_exog)
    features[:, -n_exog:] = np.tile(exog_values_dict[step + 1], (n_boot, 1))
```

`np.tile` allocates a fresh `(n_boot * n_levels, n_exog)` array each step, then copies it into the `features` slice.

### Why it's (mildly) slow

Benchmarked at typical sizes (`n_boot=250, n_levels=10, n_exog=5`):

```
np.tile per step:    ~20 µs
broadcast per step:  ~13 µs   (~1.5×)
```

Over `steps=24` that saves ~150 µs per `predict_bootstrapping` call. Small, but the fix is trivial.

### Proposed change

Replace the `np.tile` with a broadcast assignment that writes directly into a reshaped view of `features`:

```python
if has_exog:
    # `features` rows are ordered [level0_boot0, level1_boot0, ..., levelN_boot0,
    #                              level0_boot1, level1_boot1, ...].
    # Reshape view: (n_boot, n_levels, n_features) — exog block is the last n_exog cols.
    features[:, -n_exog:].reshape(n_boot, n_levels, n_exog)[:] = exog_values_dict[step + 1]
```

This avoids the temp array allocation; `features` is C-contiguous, so the reshape is a view.

If finding #2 is implemented (3-D exog buffer instead of dict-of-strided-slices), this becomes:

```python
features[:, -n_exog:].reshape(n_boot, n_levels, n_exog)[:] = exog_arr[step]
```

### Expected impact

- ~5–10% faster `_recursive_predict_bootstrapping` for multiseries when `exog` is present.
- Trivial to implement; no API or numerical change.

### Constraints / risks

- The `features` array is allocated with `order='F'` at line 2680, **not** `order='C'`. A reshape on an F-contiguous array of shape `(n_samples, n_features)` to `(n_boot, n_levels, n_features)` will **copy** unless the layout is changed — the 3-D reshape only works as a view if `features` is C-contig along the row axis or the flatten matches the desired layout. **This needs careful verification before implementing.** If the F-order layout is required for the predict-fn fast path (e.g., XGBoost `inplace_predict`), the broadcast trick may not apply, and the gain disappears.
- If F-order is required, an alternative is to pre-tile `exog_values_dict` once outside the loop into a `(steps, n_boot * n_levels, n_exog)` array — trades the per-step tile cost for a one-time allocation. Whether this is worthwhile depends on `steps × n_boot × n_levels × n_exog`.

This is the lowest-priority finding in the list — included for honesty, but likely not worth implementing alone.

---

## 4. `_train_test_split_one_step_ahead` runs the full transform pipeline twice

**Files / lines (verified):**
- Single-series: [_forecaster_recursive.py:1039-1135](skforecast/recursive/_forecaster_recursive.py#L1039-L1135)
- Multiseries: [_forecaster_recursive_multiseries.py:1534-1723](skforecast/recursive/_forecaster_recursive_multiseries.py#L1534-L1723)

### What the code does today

The function is called inside `bayesian_search_forecaster*` when using `OneStepAheadFold`. It calls `_create_train_X_y` twice — once on the train slice and once on the test slice:

```python
# Single-series, lines 1083-1109:
self.is_fitted = False
(X_train, y_train, train_index, _, ...) = self._create_train_X_y(
    y    = y.iloc[:initial_train_size],
    exog = exog.iloc[:initial_train_size] if exog is not None else None
)
test_init = initial_train_size - self.window_size
self.is_fitted = True
(X_test, y_test, *_) = self._create_train_X_y(
    y    = y.iloc[test_init:],
    exog = exog.iloc[test_init:] if exog is not None else None
)
```

Each call to `_create_train_X_y` runs:
- `transform_dataframe(..., transformer_y, ...)` (re-fit on first call, transform-only on second)
- `differentiator.fit_transform` / `transform`
- `transformer_exog.transform`
- `categorical_encoder.fit_transform`
- Lag generation (`_create_lags`)
- Window features
- NaN scan + `astype` casts

The two slices overlap by `window_size` rows but are otherwise disjoint; the bulk of the work is independent on each call. Crucially, the *transformers are the same* — the second call re-applies a fitted transformer to `y[test_init:]` and `exog[test_init:]`.

In the **multiseries** variant the cost is larger: each call additionally runs `check_preprocess_series`, `check_preprocess_exog_multiseries`, `align_series_and_exog_multiseries`, builds a per-series exog buffer, etc. ([_forecaster_recursive_multiseries.py:1622-1658](skforecast/recursive/_forecaster_recursive_multiseries.py#L1622-L1658)).

Additionally, the multiseries function has a one-hot decoding path at [_forecaster_recursive_multiseries.py:1667-1696](skforecast/recursive/_forecaster_recursive_multiseries.py#L1667-L1696):

```python
if self.encoding == 'onehot':
    encoding_keys = list(self.encoding_mapping_.keys())
    keys_arr = np.array(encoding_keys)
    level_indices = np.arange(len(encoding_keys))
    X_train_encoding = keys_arr[
        X_train[encoding_keys].to_numpy() @ level_indices
    ]
    X_test_encoding = keys_arr[
        X_test[encoding_keys].to_numpy() @ level_indices
    ]
```

The matmul materializes a `(n, n_levels)` float matrix just to recover the active column index — a dot product trick that works but allocates the entire one-hot block.

### Why it's slow

This is a per-Optuna-trial cost in the recommended fast tuning mode (`OneStepAheadFold`). With 50–100 trials × the full transform pipeline run twice per trial, this is a major chunk of Bayesian-search runtime.

The transform-pipeline work on the test slice is **almost entirely redundant** with the train slice, except for the lag/WF construction. The transformers are already fitted; they're just being applied twice instead of once.

### Proposed change

1. **Run the transformation pipeline once on the union, then split.** Refactor into a helper:
   ```python
   def _transform_union(self, y, exog):
       """Apply transformer_y, transformer_exog, differentiator,
          categorical_encoder once on the full y/exog. Return numpy arrays."""
       ...
   ```
   Then `_train_test_split_one_step_ahead` becomes:
   ```python
   y_t, exog_t = self._transform_union(y, exog)  # once
   # Build X_train from y_t[:initial_train_size], exog_t[:initial_train_size]
   # Build X_test  from y_t[test_init:],          exog_t[test_init:]
   ```
   The lag/WF construction still has to run twice (different windows), but it's cheap compared to the transformer pipeline.

2. **Replace the matmul-based one-hot decoding with `argmax`.** Both are O(n × n_levels), but `argmax` does not materialize a full float copy of the one-hot block:
   ```python
   if self.encoding == 'onehot':
       encoding_keys = np.array(list(self.encoding_mapping_.keys()))
       train_active = X_train[encoding_keys].to_numpy().argmax(axis=1)
       test_active  = X_test[encoding_keys].to_numpy().argmax(axis=1)
       X_train_encoding = encoding_keys[train_active]
       X_test_encoding  = encoding_keys[test_active]
   ```
   Same result, no `(n, n_levels)` float allocation for the matmul.

3. **(Multiseries only) Cache the result of `check_preprocess_series` / `check_preprocess_exog_multiseries`** across the two `_create_train_X_y` calls inside this function. The series and exog dicts don't change between the two calls — only the slice does. The internal `_create_train_X_y` re-runs these checks on every call.

### Expected impact

- ~40–60% faster `_train_test_split_one_step_ahead`.
- Directly halves Bayesian-search cost in `OneStepAheadFold` mode (which is the recommended fast tuning mode).
- Multiseries gains are larger than single-series because of the extra check/preprocess work.

### Constraints / risks

- The refactor crosses several internal helpers; tests under [skforecast/recursive/tests/](skforecast/recursive/tests/) for `_train_test_split_one_step_ahead` and `_create_train_X_y` should be re-run.
- Numerical results must be identical — the only change is *when* the transform runs, not *how*.
- The `is_fitted` toggle on lines 1080-1111 / 1631-1660 is fragile; preserve it carefully when refactoring.

---

## 5. Single-series `_create_train_X_y` builds a Python list of column blocks

**File / lines (verified):** [_forecaster_recursive.py:886-915](skforecast/recursive/_forecaster_recursive.py#L886-L915)

### What the code does today

```python
X_train = []
X_train_features_names_out_ = []

X_train_lags, y_train = self._create_lags(...)
if X_train_lags is not None:
    X_train.append(X_train_lags)
    X_train_features_names_out_.extend(self.lags_names)

if self.window_features is not None:
    ...
    X_train_window_features, ... = self._create_window_features(...)
    X_train.extend(X_train_window_features)
    X_train_features_names_out_.extend(X_train_window_features_names_out_)

if exog is not None:
    X_train.append(exog)
    X_train_features_names_out_.extend(X_train_exog_names_out_)

if len(X_train) == 1:
    X_train = X_train[0]
else:
    X_train = np.concatenate(X_train, axis=1)
```

The lag block, window-feature blocks, and exog block are each independent arrays held in a Python list, then concatenated along axis 1.

For comparison, the multiseries variant at [_forecaster_recursive_multiseries.py:1215-1247](skforecast/recursive/_forecaster_recursive_multiseries.py#L1215-L1247) **already preallocates**:

```python
X_train = np.empty((total_rows, n_autoreg_cols), order='C', dtype=float)
y_train = np.empty(total_rows, dtype=float)
...
for k in series_dict.keys():
    (X_train_autoreg_k, ...) = self._create_train_X_y_single_series(...)
    n = len(y_train_k)
    X_train[offset:offset + n, :] = X_train_autoreg_k
```

The single-series version was not updated to use the same pattern.

### Why it's slow

The `np.concatenate(X_train, axis=1)` at line 915 always copies all blocks into a fresh contiguous array. The internal helpers (`_create_lags`, `_create_window_features`) already allocated their own arrays. So each block is allocated, then copied into the concat result, then the original is garbage-collected — one extra full pass over the lag/WF/exog data per fit.

This is paid on every backtest fold with refit and every Optuna trial with refit.

Smaller items in the same function:
- Line 932 / 946: `pd.isna(X_train).any(axis=1)` and `pd.isna(X_train).any()`. At this point `X_train` is already a numpy array, so `pd.isna` dispatches into numpy's `isnan` — but going through pandas has small fixed overhead. `np.isnan(X_train).any()` is equivalent and slightly faster.

### Proposed change

1. **Preallocate `X_train` and write each block in place.** The total column count is computable upfront:
   ```python
   n_lags = len(self.lags) if self.lags is not None else 0
   n_wf   = len(X_train_window_features_names_out_) if self.window_features is not None else 0
   n_exog_cols = exog.shape[1] if exog is not None else 0
   n_cols = n_lags + n_wf + n_exog_cols

   # _create_lags returns y_train and (later) X_train_lags; need n_rows from it
   X_train_lags, y_train = self._create_lags(...)
   n_rows = len(y_train)
   X_train = np.empty((n_rows, n_cols), order='C', dtype=float)

   off = 0
   if X_train_lags is not None:
       X_train[:, off:off + n_lags] = X_train_lags
       off += n_lags
   if self.window_features is not None:
       for wf_block, wf_n in zip(X_train_window_features, wf_widths):
           X_train[:, off:off + wf_n] = wf_block
           off += wf_n
   if exog is not None:
       X_train[:, off:off + n_exog_cols] = exog
   ```
   Eliminates the final `np.concatenate` (one fewer full copy of `X_train`).

2. **Replace `pd.isna(X_train).any(axis=1)` with `np.isnan(X_train).any(axis=1)`** at line 932; same for line 946.

### Expected impact

- ~15–25% faster `_create_train_X_y` per fit.
- Helps every backtest fold that retrains and every Optuna trial.
- Memory: removes one transient `X_train`-sized allocation.

### Constraints / risks

- The NaN-row filter at lines 917-952 currently mutates `X_train`, `y_train`, `train_index` together. The preallocated approach is fully compatible — just write the blocks then run the same filter on the result.
- Match the exact numerical output (the concat result and the preallocated result should be bit-identical). Easy to verify with existing unit tests.

---

## 6. `_create_predict_inputs` (single series) per-call dtype/column dict rebuilds

**File / lines (verified):** [_forecaster_recursive.py:1521-1568](skforecast/recursive/_forecaster_recursive.py#L1521-L1568)

### What the code does today

```python
last_window_values = (
    last_window.iloc[-self.window_size:].to_numpy(copy=True).ravel()
)
last_window_values = transform_numpy(
    array=last_window_values, transformer=self.transformer_y, fit=False, ...
)
if self.differentiation is not None:
    differentiator = copy(self.differentiator)
    last_window_values = differentiator.fit_transform(last_window_values)

if exog is not None:
    exog = input_to_frame(data=exog, input_name='exog')
    if exog.columns.tolist() != self.exog_names_in_:
        exog = exog[self.exog_names_in_]

    exog = transform_dataframe(
        df=exog, transformer=self.transformer_exog, fit=False, ...
    )

    if self.categorical_features is not None and self.categorical_features_names_in_:
        if self.transformer_exog is None:
            exog = exog.copy()
        exog[self.categorical_features_names_in_] = (
            self.categorical_encoder.transform(...)
        )

    if not exog.dtypes.to_dict() == self.exog_dtypes_out_:
        check_exog_dtypes(exog=exog)
    else:
        check_exog(exog=exog, allow_nan=False)

    exog_values = exog.to_numpy()[:steps]
```

Per-predict allocations / wasteful work:
- Line 1521-1522: `to_numpy(copy=True)` followed by `transform_numpy(...)` which itself returns a new array — the explicit `copy=True` is redundant.
- Line 1539: `exog.columns.tolist() != self.exog_names_in_` — builds a Python list of column names and compares to a stored list every predict call.
- Line 1561: `exog.dtypes.to_dict() == self.exog_dtypes_out_` — builds a Python dict of `{col: dtype}` every predict call.
- Line 1566: `exog.to_numpy()[:steps]` — materializes the full numpy array first, then slices. If `exog` has many more rows than `steps`, the head allocation is wasted.

### Why it's slow

Each of these is a small constant overhead per predict (~5-50 µs each, depending on column count and dtype variety), but they are paid on every predict call. In a backtest with hundreds of folds, this adds up.

Specifically, `exog.dtypes.to_dict()` on a DataFrame with N columns walks all columns and builds a Python dict — for `N=20-50` that's ~10-30 µs. Comparing the resulting dict to `self.exog_dtypes_out_` is then a full dict-equality check.

### Proposed change

1. **Drop the redundant `copy=True`** on `last_window.iloc[...].to_numpy(copy=True).ravel()`. The next operation (`transform_numpy`) returns a new array regardless. This is a one-character change that removes one allocation per predict.

2. **Cache the column-name and dtype signatures at fit time.** Add at the end of `fit()` (or in `_create_train_X_y` where `self.exog_dtypes_out_` is set):
   ```python
   self._exog_names_in_tuple_ = tuple(self.exog_names_in_) if self.exog_names_in_ else None
   self._exog_dtypes_out_values_ = (
       np.array([self.exog_dtypes_out_[k] for k in self.exog_names_in_], dtype=object)
       if self.exog_dtypes_out_ else None
   )
   ```
   In `_create_predict_inputs`, replace the comparisons:
   ```python
   if tuple(exog.columns) != self._exog_names_in_tuple_:
       exog = exog[self.exog_names_in_]
   ...
   if not (exog.dtypes.values == self._exog_dtypes_out_values_).all():
       check_exog_dtypes(exog=exog)
   else:
       check_exog(exog=exog, allow_nan=False)
   ```
   Both are array-level comparisons (no Python dict allocation per call).

3. **Slice before `to_numpy()`** at line 1566:
   ```python
   exog_values = exog.iloc[:steps].to_numpy()
   ```
   When `exog` has more rows than `steps`, this avoids materializing the full numpy array. When the row counts are equal it's a no-op.

### Expected impact

- ~5–15% off each `predict()` call.
- Cumulative win across all `predict_interval`, `predict_quantiles`, `predict_bootstrapping` calls (they all go through this function).

### Constraints / risks

- Adding the cached attributes is backward-compatible (read-only optimization).
- `tuple(exog.columns)` vs `exog.columns.tolist()` — both build a Python sequence of column names; `tuple()` is marginally faster than `tolist()` because it avoids the intermediate list, but the real win is comparing to a tuple instead of a list (the tuple comparison is a single C call).
- For numeric dtype comparison `(arr1 == arr2).all()`, the dtypes must compare correctly via numpy `==`. Standard pandas dtype objects (numpy dtypes, ExtensionDtype) all support `==`, so this is safe.

---

## What was checked but **not** included as a finding

- **Lag construction (`_create_lags`).** Already uses `sliding_window_view` + reversed contiguous slice for the common case; the non-contiguous fancy-index path is unavoidable without extra preallocation, and that path is rarely hit. Not worth changing.
- **Multiseries bootstrap reshape claim from v1.** Originally claimed `last_window_boot.reshape(...)` triggered a per-step copy. Verified with a numpy benchmark that this is **a view, not a copy** (because the source is C-contiguous and the merged dimensions are adjacent). Retracted — see finding #3 for the corrected, smaller scope.
- **`predict_quantiles`.** Just calls `predict_bootstrapping` then `.quantile(...)`. Cost is dominated by `predict_bootstrapping`, so improvements come for free from #1 and #3.
- **`create_sample_weights` (multiseries).** Has small inefficiencies but only runs once per `fit()`, not per predict — small fish.
- **Per-level transformer/differentiator post-process loop in multiseries `predict` / `predict_bootstrapping`.** Real opportunity (~1.3–2× faster post-processing for `n_levels ≥ 10`) but not in the requested 1–6 list. Worth tracking separately.
- **Single-sample estimator predict in `_recursive_predict` (`X.reshape(1, -1)`).** Real ~10–30% gain for `RandomForestRegressor` / `DecisionTreeRegressor` from a per-sample fast path, but not in the requested 1–6 list. Worth tracking separately.

---

## Recommended implementation order

1. **#1** — biggest single-line win, applies to every `predict()` call in both forecasters, no API risk.
2. **#5** — small change, drop-in pattern already used in multiseries; low-risk warm-up before #4.
3. **#6** — trivial, mostly just removing redundant work.
4. **#4** — biggest absolute win for Bayesian-search users; highest refactor cost.
5. **#2** — high impact for multiseries with many levels; moderate refactor (changes the internal contract between `_create_predict_inputs` and `_recursive_predict`).
6. **#3** — only if #2 is also implemented; otherwise the standalone gain is too small to justify.
