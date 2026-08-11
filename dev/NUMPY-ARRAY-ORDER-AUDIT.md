# NumPy array memory-layout audit (`order='C'` vs `order='F'`)

Audit of every `np.empty(...)` / `np.full(...)` call in skforecast's core library (non-test)
source files, checking whether the array's memory layout matches its **entire lifecycle**:
creation, fill, and downstream consumption (not just the fill loop in isolation).

**Scope:** 16 source files, 63 call sites (`recursive/*.py`, `direct/*.py`, `stats/**/*.py`,
`foundation/*.py`, `preprocessing/*.py`, `deep_learning/_forecaster_rnn.py`). Test files were
excluded — layout micro-optimization in test fixtures isn't meaningful.

**Method:** for every 2D+ site where the fill pattern looked mismatched with its declared/default
order, the downstream code was traced (not assumed) to find what actually consumes the array next.
Where fill-pattern and usage-pattern disagreed, both sides were benchmarked together as one
pipeline (creation → fill → real downstream consumer — a model `.predict()` call, an elementwise
combine, a repeated read, a `.reshape()`), using `time.perf_counter` (mean/std over 50-300 reps
after warmup) plus `line_profiler` to attribute cost within the fill loop where relevant. A result
is `[NEGLIGIBLE IMPACT]` if the speedup is under 5% or the total pipeline runs under 1ms,
regardless of how compelling the fill-pattern read looked on paper. Environment: conda env
`skforecast_24_py13` (numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.0, lightgbm 4.7.0, xgboost 3.4.0,
catboost 1.2.10), Windows.

**Headline result:** naive "fill pattern says column-wise, so use `order='F'`" reasoning would have
been **wrong** for 3 of the 7 benchmarked candidates (2/3, 4, 5) once real downstream usage was
included — in one case it would have made the code up to **98% slower**. It was directionally
right, but for the wrong reason and needing real-model verification, for candidate 1. This is why
the full-lifecycle approach mattered more than a pure fill-pattern grep.

---

## Benchmarked candidates

### 1. `recursive/_forecaster_recursive.py:1816` — `X` (bootstrapping input matrix)

```python
X = np.full((n_boot, n_features), fill_value=np.nan, dtype=float)   # order='C' (default, implicit)
```

Filled by column-block writes across all rows, once per forecast step (`X[:, a:b] = ...`), then
`pred = predict_fn(X)` — a real estimator `.predict()` call — every iteration
(`_recursive_predict_bootstrapping`, lines 1842-1880). `X` is purely local: never reshaped,
ravelled, or returned, so an `order=` change is correctness-safe.

**Hypothesis:** the estimator's own computation dominates the array fill; also, each of
sklearn / LightGBM / XGBoost / CatBoost has its own input-ingestion path, so a hidden internal
copy could negate any `order='F'` win for some but not all of them. Benchmarked with real fitted
estimators (not stubs) at a cheap tier (isolates whether the effect exists) and a realistic tier
(answers whether it survives at production-scale inference cost). `n_boot=250`, `n_features=20`,
`steps=24`.

| Tier | Estimator | `order='C'` (baseline) | `order='F'` | Speedup | `order='F'` + `ascontiguousarray` before predict |
|---|---|---|---|---|---|
| cheap | sklearn `Ridge()` | 3.86 ms | 3.56 ms | **+7.8%** | 4.04 ms (-4.8%) |
| cheap | LightGBM (`n_estimators=10`) | 19.31 ms | 18.81 ms | +2.6% (negligible) | 18.80 ms |
| cheap | XGBoost (`n_estimators=10`) | 12.66 ms | 12.21 ms | +3.5% (negligible) | 12.23 ms |
| cheap | CatBoost (`n_estimators=10`) | 27.77 ms | 13.58 ms | **+51.1%** | 28.00 ms (-0.8%) |
| realistic | sklearn `RandomForestRegressor(n_estimators=100)` | 209.06 ms | 203.56 ms | +2.6% (negligible) | 272.69 ms (-30.4%, noisy) |
| realistic | LightGBM (`n_estimators=200`) | 42.45 ms | 32.90 ms | **+22.5%** | 33.95 ms |
| realistic | XGBoost (`n_estimators=200`) | 26.81 ms | 25.31 ms | **+5.6%** | 25.13 ms |
| realistic | CatBoost (`n_estimators=200`) | 30.90 ms | 17.12 ms | **+44.6%** | 31.12 ms (-0.7%) |

The `ascontiguousarray`-before-predict variant for CatBoost lands right back at the `'C'` baseline
(no better, no worse) — this confirms the mechanism: CatBoost's internal ingestion is genuinely
faster with column-major input and avoids an internal copy when given `'F'` natively; forcing a
copy back to `'C'` just pays for exactly the copy CatBoost otherwise skips. `order='F'` was never
worse than baseline for any estimator/tier tested.

**Verdict: `[NEEDS CHANGE to 'F']`.** Real, substantial win for CatBoost (+44 to +51%) and
LightGBM at realistic scale (+22.5%), a modest real win for XGBoost (+5.6%), and neutral-to-mildly-positive
for sklearn. Safe as a single unconditional change — no per-library special-casing needed since
`'F'` never lost.

---

### 2 & 3. `correction_factor` — `recursive/_forecaster_recursive_multiseries.py:3434` and `deep_learning/_forecaster_rnn.py:1716`

```python
correction_factor = np.full(shape=(steps, n_levels), fill_value=np.nan, order='C', dtype=float)
```

Filled column-by-column (`correction_factor[:, i] = ...`, loop over levels), each column computed
via `np.quantile(...)` or a Python-level `np.vectorize(replace_func)(...)` over per-level
residuals. Downstream: `lower_bound = predictions - correction_factor`;
`upper_bound = predictions + correction_factor` — elementwise against a `'C'`-ordered `predictions`.

**`line_profiler` on the fill loop** (`n_levels=1000`): **99.7%** of the non-binned fill's time is
inside the `np.quantile(...)` call itself, not the array write. For the binned variant, **94.5%**
is the per-level `np.quantile` dict comprehension and **3.6%** is the `np.vectorize` + write
combined — the write is a rounding error either way.

| `n_levels` | Branch | `order='C'` (baseline) | `order='F'` | `order='F'` + `ascontiguousarray` before combine |
|---|---|---|---|---|
| 100 | non-binned | 4.48 ms | 4.48 ms (-0.1%) | 4.35 ms |
| 100 | binned | 41.03 ms | 44.39 ms (**-8.2%**) | 40.90 ms |
| 1000 | non-binned | 44.56 ms | 55.22 ms (**-23.9%**) | 50.25 ms |
| 1000 | binned | 447.05 ms | 506.51 ms (**-13.3%**) | 466.35 ms |

**Verdict: `[OPTIMAL as 'C']`** — confirmed, do not change. The fill loop's real cost is the
per-level statistics computation, not the write; `order='F'` gains nothing there and actively
degrades the elementwise combine with the `'C'`-ordered `predictions` array, worsening as
`n_levels` grows. This is the clearest case where a naive fill-pattern read (which flagged this as
"column-wise, so use `'F'`") would have made the code up to 24% slower. Identical conclusion for
both call sites since they share the exact shape/fill/usage pattern.

---

### 4. `recursive/_forecaster_recursive_multiseries.py:3272` — `sampled_residuals` (3D, non-binned)

```python
sampled_residuals = np.full(shape=(steps, n_levels, n_boot), fill_value=np.nan, order='C', dtype=float)
```

Filled with the **middle axis fixed** per write (`sampled_residuals[:, i, :] = rng.choice(...)`,
loop over levels — mismatched for `'C'`), but read **inside the per-step loop** of
`_recursive_predict_bootstrapping` as `pred += sampled_residuals[step, :, :]` — a **leading-axis-fixed**
slice, exactly what `'C'` order makes contiguous. Fill happens once; the read happens once per step
(repeated `steps=24` times).

| `n_levels` | `order='C'` (baseline, matches the repeated read) | `order='F'` (matches the one-time fill) | `'F'` fill + `ascontiguousarray` before read loop |
|---|---|---|---|
| 100 | 5.81 ms | 6.90 ms (**-18.9%**) | 7.74 ms (-33.2%) |
| 1000 | 58.16 ms | 115.48 ms (**-98.6%**) | 130.28 ms (-124.0%) |

(For reference, the fill step *alone* does show `'F'` as ~5% faster — but that's swamped by how
much worse it makes the repeated read; the total-pipeline number is what matters.)

**Verdict: `[OPTIMAL as 'C']`** — confirmed, do not change. This is the starkest example of why
counting repetitions matters: the read happens `steps` times against the fill's once, so optimizing
for the read (i.e., keeping `'C'`) is correct even though the fill loop itself "looks" mismatched.

---

### 5. `recursive/_forecaster_recursive_multiseries.py:3261` — `sampled_residuals` (4D, binned)

```python
sampled_residuals = np.empty((n_bins, steps, n_boot, n_levels), order='C', dtype=float)
```

Filled via a nested loop `sampled_residuals[bin_idx, :, :, i] = rng.choice(...)` (both outer and
inner axis fixed). Read via fancy indexing
`sampled_residuals[predicted_bins, step, boot_indices, j]` inside the per-step, per-level loop.

`line_profiler` on the fill: **98.3%** of time is the `rng.choice(...)` call itself, not the write.

| `n_levels` | `order='C'` (baseline) | `order='F'` |
|---|---|---|
| 20 | 15.84 ms | 16.69 ms (**-5.3%**) |
| 100 | 80.35 ms | 105.24 ms (**-31.0%**) |

**Verdict: `[OPTIMAL as 'C']`** — confirmed. Contrary to the initial hypothesis that a scattered
fancy-index gather would be roughly layout-insensitive, `'F'` measurably *hurts* here too (worse
memory-access strides for the gather), on top of the fill being `rng.choice`-dominated either way.

---

### 6. `preprocessing/_preprocessing.py:1502` — `rolling_features` (`RollingFeatures.transform`)

```python
rolling_features = np.full(shape=(X.shape[1], self.n_stats), fill_value=np.nan, dtype=float)  # order='C' (default)
```

Vectorizable stats filled column-by-column (`rolling_features[:, j] = np.nanmean(window, axis=0)`,
etc.); this runs on every step of the recursive multi-series predict loop, so `X.shape[1]` can
equal `n_levels`. Downstream: immediately absorbed into `np.concatenate([...], axis=1)` in the
caller, which copies into a new `'C'`-ordered array regardless of source order.

| `X.shape[1]` (`n_levels`) | `order='C'` (baseline) | `order='F'` |
|---|---|---|
| 1 | 0.0995 ms | 0.0572 ms (+42.5%, but **<1ms total**) |
| 100 | 0.0723 ms | 0.0769 ms (-6.3%, but **<1ms total**) |
| 1000 | 0.1421 ms | 0.1370 ms (+3.6%) |

**Verdict: `[NEGLIGIBLE IMPACT]`** at every scale tested — every variant is under the 1ms absolute
threshold, so this is noise regardless of the percentage swings.

---

### 7. `stats/_ets.py:442` and `stats/_arima.py:781` — `predictions` (ETS/ARIMA forecast + intervals)

```python
predictions = np.empty((steps, 1 + 2 * n_levels), dtype=float)   # order='C' (default)
```

Column-by-column fill, but `n_levels` (confidence levels) is realistically ≤ ~5, so ≤ 11 columns.
Result is wrapped essentially as-is into a DataFrame downstream.

| `steps` | `order='C'` (baseline) | `order='F'` |
|---|---|---|
| 10 | 0.0477 ms | 0.0478 ms (-0.2%) |
| 100 | 0.0495 ms | 0.0486 ms (+1.9%) |
| 1000 | 0.0692 ms | 0.0569 ms (+17.8%, but **<1ms total**) |

**Verdict: `[NEGLIGIBLE IMPACT]`** — confirmed as expected; the column count is too small to matter
at any realistic horizon.

---

## Recommended action

Only **one** of the 63 call sites merits a change:

- `skforecast/recursive/_forecaster_recursive.py:1816` — change
  `X = np.full((n_boot, n_features), fill_value=np.nan, dtype=float)` to add `order='F'`, matching
  the equivalent (and already-correct) `features` array in
  `_forecaster_recursive_multiseries.py` (`_recursive_predict`/`_recursive_predict_bootstrapping`,
  which already use `order='F'` for this exact column-block-fill pattern). This appears to be a
  single-series/multi-series inconsistency rather than an intentional choice.

Everything else audited is either already optimal, a no-op (1-D), or correctness-constrained.

---

## Full inventory (all 63 call sites)

Sites not listed in the benchmarked section above, grouped by file. "N/A - 1D" means the array is
one-dimensional, so `order` has no effect. "OPTIMAL" means fill and downstream usage were both
traced and are already consistent with the declared/default order.

### `recursive/_forecaster_recursive.py`
| Line | Array | Verdict | Note |
|---|---|---|---|
| 1701 | `X` | N/A - 1D | Per-step feature vector, reused via contiguous slice writes |
| 1702 | `predictions` | N/A - 1D | Sequential scalar fill |
| 1819 | `predictions` (steps × n_boot) | OPTIMAL | Filled row-by-row (`predictions[i,:]=pred`), matches default `'C'` |
| 1824 | inline `np.full((steps,n_boot))` | OPTIMAL | Immediately consumed by `np.vstack`; resulting `last_window` is read/written row-wise, matching `'C'` |
| 1964 | `X_window_features` (steps × n_wf) | OPTIMAL | Explicit `order='C'`, filled row-by-row |

### `recursive/_forecaster_recursive_classifier.py`
| Line | Array | Verdict | Note |
|---|---|---|---|
| 825 | `y_encoded` | N/A - 1D | Boolean-mask fancy-index fill |
| 1580 | `last_window_values` | N/A - 1D | Boolean-mask fancy-index fill |
| 1668 | `X` | N/A - 1D | Per-step feature vector |
| 1671 | `predictions` | N/A - 1D | Sequential scalar fill |
| 1675 | `predictions` (steps × n_classes) | OPTIMAL | Filled row-by-row, matches default `'C'` |
| 1800 | `X_window_features` | OPTIMAL | Explicit `order='C'`, filled row-by-row |

### `recursive/_forecaster_recursive_multiseries.py`
| Line | Array | Verdict | Note |
|---|---|---|---|
| 1255 | `X_train` (total_rows × n_autoreg_cols) | OPTIMAL | Explicit `order='C'`; filled via contiguous row-band writes per series — row blocks are naturally contiguous under `'C'` |
| 1256 | `y_train` | N/A - 1D | |
| 1258 / 1260 | `encoded_values` | N/A - 1D | |
| 2451 | `last_window_matrix` | OPTIMAL | Explicit `order='F'`, filled column-by-column — already correct |
| 2613 | `features` (`_recursive_predict`) | OPTIMAL | Explicit `order='F'`, filled via column-block writes — already correct, and the template candidate 1 should match |
| 2619 | `predictions` (steps × n_levels) | OPTIMAL | Explicit `order='C'`, filled row-by-row |
| 2779 | `features` (`_recursive_predict_bootstrapping`) | OPTIMAL | Explicit `order='F'`, filled via column-block writes — already correct |
| 2785 | `boot_predictions` (3D) | OPTIMAL | Explicit `order='C'`; filled via leading-axis (step) slabs — contiguous under `'C'` |
| 2793 | `last_window_boot` (3D) | OPTIMAL | Explicit `order='C'`; filled/read via leading-axis slabs |
| 2978 | `X_window_features` (per-level) | OPTIMAL | Explicit `order='C'`, filled row-by-row |
| 3010 | `exog_cols` (per-level) | OPTIMAL | Explicit `order='C'`, filled row-by-row |

### `direct/_forecaster_direct.py` and `direct/_forecaster_direct_multivariate.py`
| Line | Array | Verdict | Note |
|---|---|---|---|
| 2214 / 2387 | `Xs_array` | OPTIMAL | Default `'C'`; mixed fill, but final consumption is per-row slicing (`Xs_array[i:i+1]`) into single-row prediction calls, which is exactly what `'C'` optimizes |
| 2542 / 2726 | `sampled_residuals` (predicted_bins × n_boot) | OPTIMAL | Explicit `order='C'`, filled row-by-row |

### `stats/arar/_arar_base.py`
| Line | Array | Verdict | Note |
|---|---|---|---|
| 131 | `gamma` | N/A - 1D | |
| 147 | `A` (fixed 4×4) | NEGLIGIBLE IMPACT | Tiny, fixed-size, called repeatedly inside a triple-nested loop — fits in cache regardless of order |
| 286 | `fitted` | N/A - 1D | |

### `stats/exponential_smoothing/_ets_base.py`
| Line | Array | Verdict | Note |
|---|---|---|---|
| 744 | `fitted` | N/A - 1D | Fully populated by `fill_value` itself |

### `stats/arima/_arima_base.py` (all `@njit`-compiled — numba's memory model differs from plain-numpy cache behavior)
| Line | Array | Verdict | Note |
|---|---|---|---|
| 612, 660, 1170, 1171, 1177, 2078, 2079, 2268, 2974 | various | N/A - 1D | |
| 1172 | `mm` (rd × rd) | NEGLIGIBLE IMPACT | `rd` bounded ~<30 (ARMA/seasonal/differencing order); numba-jitted, tiny |
| 2087 | `H` (n × n) | NEGLIGIBLE IMPACT | `n` = number of free ARIMA parameters, typically single/low-double-digit |

### `stats/arima/_auto_arima.py`
| Line | Array | Verdict | Note |
|---|---|---|---|
| 1022 | `results` (nmodels × 8) | OPTIMAL | Filled and read predominantly row-by-row (per-candidate-model rows); one full-column read at the end for IC sorting doesn't change the verdict given the fixed 8-column width |

### `foundation/_foundation_model.py`
| Line | Array | Verdict | Note |
|---|---|---|---|
| 996 | `pred_matrix` (steps × n_series × n_cols) | **AMBIGUOUS - Requires manual review** | Order is constrained by a downstream `.reshape(steps*n_series, n_cols)` that depends on C-contiguity; changing it is a correctness risk, not a perf tuning choice |

### `foundation/_adapters.py`
| Line | Array | Verdict | Note |
|---|---|---|---|
| 1807, 1864, 2446, 2503 | `item_id` | N/A - 1D | Fully populated by scalar `fill_value` |
| 2782 | `context_batch` (n_series × context_length) | OPTIMAL | Default `'C'`, filled row-by-row |
| 2931 | `covariates` (3D) | OPTIMAL | Default `'C'`; filled via last-axis slice writes, which is exactly what `'C'` makes contiguous |

### `preprocessing/_preprocessing.py`
| Line | Array | Verdict | Note |
|---|---|---|---|
| 229 | padding block | N/A - 1D | Immediately consumed by `np.append` |
| 2098 | `rolling_features` (categorical variant) | OPTIMAL | Default `'C'`, filled row-by-row |

### `preprocessing/_calendar.py`
| Line | Array | Verdict | Note |
|---|---|---|---|
| 886, 894 | `to_holiday` / `since_holiday` | N/A - 1D | Boolean-mask fancy-index fill |

### `deep_learning/_forecaster_rnn.py`
| Line | Array | Verdict | Note |
|---|---|---|---|
| 1356 | `last_window_matrix` | OPTIMAL | Explicit `order='F'`, filled column-by-column — already correct |
| 1395 | padding block | N/A | Fully populated by `fill_value=0.`, immediately concatenated |
