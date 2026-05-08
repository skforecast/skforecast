# Performance Optimization Findings — `ForecasterRecursiveMultiSeries`

File: `skforecast/recursive/_forecaster_recursive_multiseries.py`  
Analysis date: 2026-05-08

---

## High Impact — Scale with number of series (K)

### 1. Per-level mask rebuild in `fit()` — lines 2069–2083

**Pattern (slow):**
```python
for level in X_train_series_names_in_:
    if self.encoding == 'onehot':
        mask = X_train[level].to_numpy() == 1.
    else:
        encoded_value = self.encoding_mapping_[level]
        mask = X_train['_level_skforecast'].to_numpy() == encoded_value
```

**Why it's slow:** For each of K series, scans the full `n_total`-row column to build a mask → O(K·n).

**Fix:** Extract the level-code column **once** before the loop, then build per-level indices in a single pass using `np.argsort` / `np.searchsorted`, or `pd.Series(codes).groupby(...).indices`. Complexity: O(K·n) → O(n + K).

```python
# Before the loop
if self.encoding != 'onehot':
    level_codes = X_train['_level_skforecast'].to_numpy()
    # Build {encoded_value: bool_mask} map once
    level_mask_map = {
        v: level_codes == v
        for v in self.encoding_mapping_.values()
    }

for level in X_train_series_names_in_:
    if self.encoding == 'onehot':
        mask = X_train[level].to_numpy() == 1.
    else:
        mask = level_mask_map[self.encoding_mapping_[level]]
```

---

### 2. `.sum() > 0` per onehot column — lines 1410–1413

**Pattern (slow):**
```python
if self.encoding == 'onehot':
    X_train_series_names_in_ = [
        col for col in series_names_in_ if X_train[col].sum() > 0
    ]
```

**Why it's slow:** Calls `X_train[col].sum()` K times — K full-column scans over `n_total` rows.

**Fix:** The active levels are already known from the integer codes before one-hot expansion. Use `np.unique` on the code array and map back via `encoding_mapping_`.

```python
if self.encoding == 'onehot':
    active_codes = np.unique(X_train['_level_skforecast'].to_numpy())
    inv_map = {v: k for k, v in self.encoding_mapping_.items()}
    X_train_series_names_in_ = [inv_map[c] for c in active_codes if c in inv_map]
```

---

### 3. Per-series `pd.Series` rebuild for window features — `_create_train_X_y_single_series` ~lines 991–1003

**Pattern (slow):**
```python
y_window_features = pd.Series(y_values[n_diff:], index=y_index[n_diff:])
X_train_window_features, ... = self._create_window_features(y=y_window_features, ...)
```
Inside `_create_window_features` (line ~886), the result is sliced `.iloc[-len_train_index:]` and then `.to_numpy()`.

**Why it's slow:** A full `pd.Series` is created per series just to call `transform_batch`, which immediately converts back to numpy. With many series this is K × (Series alloc + DataFrame alloc + object-pandas conversion).

**Fix:** If the window-feature transformer has a numpy fast path, pass the numpy slice directly. Otherwise hoist the length-slice math out of `_create_window_features` so the DataFrame is never built larger than needed.

---

## Medium Impact — Inside the recursive prediction step loop

### 4. `np.concatenate` of window-feature outputs every step — lines 2549–2557

**Pattern (slow):**
```python
if has_window_features:
    window_data = last_window[i:-remaining, :]
    features[:, n_lags:n_autoreg] = np.concatenate(
        [wf.transform(window_data) for wf in self.window_features],
        axis=1
    )
```

**Why it's slow:** Every step: builds a list, allocates a concatenated temporary array, then copies into `features`. Multiplied by `steps`.

**Fix:** Precompute column-slice offsets per `wf` once **before** the loop, write each `wf.transform(...)` directly into its named slice. The bootstrapping path at ~line 2724 already uses this pattern.

```python
# Before the loop — precompute slices
wf_slices = []
col = n_lags
for wf in self.window_features:
    n = wf.n_features_out_  # number of output columns
    wf_slices.append(slice(col, col + n))
    col += n

# Inside the loop
if has_window_features:
    window_data = last_window[i:-remaining, :]
    for wf, sl in zip(self.window_features, wf_slices):
        features[:, sl] = wf.transform(window_data)
```

---

### 5. `np.vectorize` lambda over bins — lines 3315–3321

**Pattern (slow):**
```python
replace_func = np.vectorize(lambda x: correction_factor_by_bin[x])
correction_factor[:, i] = replace_func(predictions_bin)
```

**Why it's slow:** `np.vectorize` is a Python-level loop. No vectorization actually happens.

**Fix:** Build a numpy lookup table from the dict once, then use integer indexing — O(steps) numpy indexing instead of per-element Python calls.

```python
bins_sorted = sorted(correction_factor_by_bin)
lut = np.array([correction_factor_by_bin[k] for k in bins_sorted])
# predictions_bin must be integer bin ids in [0, len(lut))
correction_factor[:, i] = lut[predictions_bin]
```

---

### 6. `last_window[...][::-1].T` stride flip per step — line 2545

**Pattern (slow):**
```python
features[:, :n_lags] = last_window[-(remaining + n_lags): -remaining, :][::-1].T
```

**Why it's slow:** `[::-1]` is a negative-stride view; the assignment to `features` (column-major `order='F'`) forces a non-contiguous materialization. `.T` adds another strided read.

**Fix:** Reverse the **destination** slice instead — avoids one stride flip.

```python
features[:, n_lags-1::-1] = last_window[-(remaining + n_lags): -remaining, :].T
```

---

### 7. `np.tile` of exog every step in bootstrapping — line ~2732

**Pattern (slow):**
```python
features[:, -n_exog:] = np.tile(exog_values_dict[step + 1], (n_boot, 1))
```

**Why it's slow:** Allocates a new `(n_boot·n_levels, n_exog)` tiled array every step. Multiplied by `steps`.

**Fix:** Use broadcasting assignment to avoid the extra allocation.

```python
features[:, -n_exog:] = exog_values_dict[step + 1][np.newaxis, :]
# or reshape + broadcast if features is 3D
```

---

## Lower-but-easy Wins

### 8. Per-level `transform_numpy` over single columns in `_create_predict_inputs`

K separate calls to `transform_numpy` over single columns when all levels share the same transformer. Group levels by transformer identity and call once on a 2D slice.

### 9. Dummy NaN exog buffer when `exog is None`

When `exog is None`, a NaN-filled buffer is still created and later dropped. The `ignore_exog` flag already exists — use it earlier to skip the buffer creation and the subsequent `drop(columns=...)` entirely.

---

## Summary Table

| # | Location | Pattern | Complexity change | Expected gain |
|---|----------|---------|-------------------|---------------|
| 1 | `fit()` lines 2069–2083 | Per-level mask rebuild | O(K·n) → O(n+K) | **High** |
| 2 | `_create_train_X_y` lines 1410–1413 | `.sum()` per onehot col | O(K·n) → O(n) | **High** |
| 3 | `_create_train_X_y_single_series` ~991 | Per-series `pd.Series` wrap | O(K) allocs removed | **High** |
| 4 | `_recursive_predict` lines 2549–2557 | `np.concatenate` in step loop | O(steps) allocs removed | **Medium** |
| 5 | `_predict_interval_conformal` lines 3315–3321 | `np.vectorize` lambda | Python loop → numpy index | **Medium** |
| 6 | `_recursive_predict` line 2545 | `[::-1].T` per step | stride flip eliminated | **Medium** |
| 7 | `_recursive_predict_bootstrapping` ~2732 | `np.tile` per step | O(steps) allocs removed | **Medium** |
| 8 | `_create_predict_inputs` ~2378 | Per-level single-col transform | K calls → 1 batched | **Low–Medium** |
| 9 | `_create_train_X_y` ~1017 | Dummy NaN exog buffer | 1 alloc + drop removed | **Low** |
