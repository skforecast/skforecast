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

**Why it's slow:** For each of K series, builds a full boolean array of length n_total → O(K·n) comparisons and K×n memory.

**⚠️ Benchmark finding (2026-05-08):** Simply extracting `.to_numpy()` before the loop or pre-building a `{code: bool_mask}` dict **does not help** — both approaches still do K×n boolean comparisons and provide only ~1x speedup. The bottleneck is the comparisons themselves, not the pandas overhead.

**Correct fix:** Sort the code array once, then use `np.searchsorted` per level to get row indices directly. This avoids materialising K boolean arrays of length n. Complexity: O(n log n + K log n), returns index slices instead of bool masks.

```python
# Sort level codes once before the loop
level_codes = X_train['_level_skforecast'].to_numpy()
sort_idx     = np.argsort(level_codes, kind='stable')
sorted_codes = level_codes[sort_idx]

for level in X_train_series_names_in_:
    if self.encoding == 'onehot':
        mask = X_train[level].to_numpy() == 1.
    else:
        v  = self.encoding_mapping_[level]
        lo = np.searchsorted(sorted_codes, v, side='left')
        hi = np.searchsorted(sorted_codes, v, side='right')
        idx = sort_idx[lo:hi]   # row indices — use y_train[idx], y_pred[idx]
```

**Measured speedup (N_per_series=1950):**

| K | OLD (ms) | NEW indices (ms) | Speedup |
|--:|--:|--:|--:|
| 50 | 3.5 | 1.2 | 3.1× |
| 100 | 14.2 | 2.2 | 6.6× |
| 300 | 149.9 | 5.3 | 28× |
| 600 | 667.1 | 9.1 | 74× |
| 1000 | 1828.7 | 13.9 | 132× |

**Note:** this changes the call sites of `_binning_in_sample_residuals` from `y_train[mask]` (bool indexing) to `y_train[idx]` (integer indexing). Both are equivalent.

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

**Measured speedup (N_per_series=1950):**

| K | OLD (ms) | NEW (ms) | Speedup |
|--:|--:|--:|--:|
| 100 | 11.8 | 5.0 | 2.4× |
| 300 | 99.5 | 14.7 | 6.8× |
| 600 | 393.4 | 28.8 | 13.7× |
| 1000 | 1071.6 | 48.1 | 22.3× |

---

### 3. Per-series `pd.Series` rebuild for window features — `_create_train_X_y_single_series` ~lines 991–1003

**Pattern (slow):**
```python
y_window_features = pd.Series(y_values[n_diff:], index=y_index[n_diff:])
X_train_window_features, ... = self._create_window_features(y=y_window_features, ...)
```
Inside `_create_window_features` (line ~886), the result is sliced `.iloc[-len_train_index:]` and then `.to_numpy()`.

**Why it's slow:** For each of K series, the full pandas pipeline runs: `pd.Series` alloc → `pd.Series.rolling()` (per stat) → `pd.concat` → `iloc` → `.to_numpy()`. The actual computation (rolling stats) dominates at ~1083 µs/series; the pandas allocations add ~172 µs overhead.

**Fix:** Implement a numpy fast path in `RollingFeatures` using `np.lib.stride_tricks.sliding_window_view`. The result is already numpy — no `pd.Series`, no `pd.concat`, no `.to_numpy()`.

```python
# In _create_train_X_y_single_series, instead of:
#   y_window_features = pd.Series(y_values[n_diff:], index=y_index[n_diff:])
#   X_train_wf = wf.transform_batch(y_window_features)
#   result = X_train_wf.iloc[-len_train_index:].to_numpy()

# New path (for stats that support it: mean/std/min/max/sum/median):
#   windows = np.lib.stride_tricks.sliding_window_view(y_values[n_diff:], window_size)
#   result = np.nanmean(windows[:-1], axis=1)[-len_train_index:]  # example for 'mean'
```

**Important:** `RollingFeatures.transform_batch` uses `closed='left'` (current point excluded from window). The numpy path must replicate this: position `i` uses `y[i-w:i]`, so valid rows start at index `window` (not `window-1`), and the windows to use are `sliding_window_view[:-1]`.

**Limitations of numpy fast path:**
- Only handles: `mean`, `std`, `min`, `max`, `sum`, `median`
- Stats `ewm`, `ratio_min_max`, `coef_variation` still need pandas rolling → fall back to `transform_batch`

**Measured speedup (LEN_SERIES=2000, stats=['mean','std','min','max'], window=7):**

| K | OLD (ms) | NEW (ms) | Speedup |
|--:|--:|--:|--:|
| 10 | 11.3 | 7.5 | 1.5× |
| 50 | 56.2 | 37.1 | 1.5× |
| 100 | 113.2 | 74.3 | 1.5× |
| 300 | 339.1 | 231.2 | 1.5× |
| 600 | 690.0 | 467.1 | 1.5× |

**Single-series cost breakdown:**

| Step | Time (µs) |
|------|----------:|
| `pd.Series` alloc | 155 |
| `transform_batch` (pandas rolling + concat) | 1083 ← dominates |
| `iloc[-n:] + .to_numpy()` | 18 |
| **Total OLD** | **1255** |
| `_rolling_numpy` (numpy fast path) | 760 |
| **Total NEW** | **760** |

**Assessment:** The gain is real (~1.5×) but constant — speedup does NOT grow with K. The bottleneck is the rolling computation itself (O(n) per series per stat), which numpy reduces by ~30% vs pandas. The pd.Series allocation overhead (~155 µs) is only ~12% of the total. Worth implementing if window_features are used frequently, but lower priority than opts #1/#2.

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
| 1 | `fit()` lines 2069–2083 | Per-level mask rebuild | O(K·n) → O(n log n + K log n) | **High** (132× at K=1000) |
| 2 | `_create_train_X_y` lines 1410–1413 | `.sum()` per onehot col | O(K·n) → O(n log n) | **High** (22× at K=1000) |
| 3 | `_create_train_X_y_single_series` ~991 | Per-series `pd.Series` wrap | pandas rolling → numpy sliding_window_view | **Medium** (1.5× constant) |
| 4 | `_recursive_predict` lines 2549–2557 | `np.concatenate` in step loop | O(steps) allocs removed | **Medium** |
| 5 | `_predict_interval_conformal` lines 3315–3321 | `np.vectorize` lambda | Python loop → numpy index | **Medium** |
| 6 | `_recursive_predict` line 2545 | `[::-1].T` per step | stride flip eliminated | **Medium** |
| 7 | `_recursive_predict_bootstrapping` ~2732 | `np.tile` per step | O(steps) allocs removed | **Medium** |
| 8 | `_create_predict_inputs` ~2378 | Per-level single-col transform | K calls → 1 batched | **Low–Medium** |
| 9 | `_create_train_X_y` ~1017 | Dummy NaN exog buffer | 1 alloc + drop removed | **Low** |
