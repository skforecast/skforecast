# Foundation Module — Inefficiency Analysis

Analysis of potential inefficiencies found in `FoundationModel` and related code.  
Date: 2026-04-12

---

## 1. `check_preprocess_exog_multiseries` does unnecessary work (in `fit`)

**Location:** `utils/utils.py` line 2631, called from `_foundation_model.py` line 202

**What happens:**
- `.copy()` on every exog input (lines 2693, 2741) — exog is copied here, then
  copied again inside `adapter.fit()` when trimming to `context_length`.
- `check_exog()` per series (line 2756) — with `allow_nan=True` it's essentially
  a no-op for DataFrames (just a type + name check).
- Cross-series dtype consistency check via `pd.DataFrame` construction
  (lines 2774–2786) — irrelevant since foundation models pass each series' exog
  independently as numpy arrays.
- Validates exog column names don't collide with series names (line 2793) — ML-specific
  constraint (exog becomes columns in `X_train`), not applicable.

**Verdict: Real but low priority.** The redundant `.copy()` is the only meaningful
cost. For typical foundation model workloads (few series, small exog) it's negligible.
Would only matter with very large exog (100+ series × 50 columns × 100k rows).

**Possible fix:** Add `copy=True` parameter to `check_preprocess_exog_multiseries` and
pass `copy=False` from foundation models.

---

## 2. `align_series_and_exog_multiseries` trims NaN and reindexes (in `fit`)

**Location:** `utils/utils.py` line 2811, called from `_foundation_model.py` line 211

**What happens:**
- Trims leading/trailing NaN from each series via `first_valid_index()`/`last_valid_index()`.
- Reindexes exog to match the trimmed range.

**Correctness concern — trailing NaN trimming:**
- `series_indexes` is captured **before** alignment (line 196).
- `align_series_and_exog_multiseries` trims trailing NaN from `series_dict` (line 211).
- The **trimmed** `series_dict` goes to `adapter.fit()` → stored in `_history`.
- `training_range_` is computed from the **untrimmed** `series_indexes` (line 236).
- Result: `training_range_` says data ends at date X, but `_history` ends at an
  earlier date Y. Not currently a runtime error because `predict()` uses
  `_history.index` directly, not `training_range_`.

**Semantic concern — NaN removal:**
- Chronos2, TimesFM, and Moirai handle NaN natively in context windows (treated as
  missing observations the model can reason about).
- `align` silently removes leading/trailing NaN, reducing effective context.
- For ML forecasters this is necessary (can't train on NaN targets), for foundation
  models it's counterproductive.
- Note: only leading/trailing NaN are trimmed; middle NaN are preserved.

**Performance:** The function is cheap (one `isna().iat[]` check per series). Not a
performance issue.

**Verdict: Real correctness concern, not a performance issue.**

**Possible fix:** Skip `align_series_and_exog_multiseries` in `FoundationModel.fit()`
entirely. Replace with a simpler foundation-specific function that only does index
alignment without NaN trimming. Also recompute `training_range_` from the
post-alignment `series_dict` to avoid metadata inconsistency.

---

## 3. Double context-length trimming in `predict`

**Location:** `_foundation_model.py` lines 487–494

**What happens:** When `last_window is not None`, `predict()` trims `history_dict`
and `past_exog_dict` to `context_length` with `.iloc[-context_length:]`. But the
adapter's `predict()` may also trim internally.

**Verdict: Not a real problem.** The adapters' `predict()` does **not** trim — only
`adapter.fit()` does. The trimming in `predict()` is correct and necessary for
user-provided `last_window`. When `last_window is None`, the trimming block is skipped
entirely because `_history` was already trimmed during `fit()`.

---

## 4. Multi-series index interleaving via `np.column_stack().ravel()`

**Location:** `_foundation_model.py` lines 517–522

**What happens:**
```python
long_index = np.column_stack(
    [np.asarray(idx) for idx in per_series_indices]
).ravel()
```

**Initial concern:** Converting `DatetimeIndex` → numpy might lose type information.

**Actual behavior:** Pandas correctly reconstructs `DatetimeIndex` from numpy
`datetime64[ns]` arrays. The resulting DataFrame index **is** a `DatetimeIndex`.
All downstream operations (`.isin()`, `.loc[]`) work correctly.

**Performance:** `np.array([...]).T.ravel()` is ~25% faster than
`np.column_stack([...]).ravel()`, but absolute difference is ~2μs per call at
realistic scales (50 series × 100 steps). Negligible vs model inference time.

**Verdict: Not a real problem.** Correct and efficient enough.

---

## 5. `np.tile` vs `np.repeat` for `level_col`

**Location:** `_foundation_model.py` lines 514–522

**What happens:**
- Single-series: `np.repeat(series_names, steps)` → `["y", "y", "y", ...]`
- Multi-series: `np.tile(series_names, steps)` → `["s1", "s2", "s1", "s2", ...]`

Both correctly match the interleaving pattern of the index.

**Concern:** Single-series creates a constant array that backtesting immediately
drops (`pred.drop(columns=["level"])`).

**Verdict: Not worth changing.** The allocation is trivial (a few hundred strings).
Keeping uniform output format (always has `"level"` column) simplifies the code.
Adding a branch to skip `level_col` in single-series would add complexity for zero
measurable benefit.

---

## 6. `_prepare_past_exog` returns adapter internals by reference

**Location:** `_foundation_model.py` lines 340–341

**What happens:** When `last_window is None` and exog exists, `_prepare_past_exog`
returns `self.adapter._history_exog` directly (no copy). Similarly, `history_dict`
is assigned by reference at line 428.

**Concern:** If someone modified these dicts, they'd mutate adapter state.

**Actual behavior:** The entire `predict()` path is **read-only** on both dicts:
- `predict()` only reads indices and values.
- `adapter.predict()` calls `.to_numpy()` / `.to_frame().to_numpy()` — creates new arrays.
- The trimming block only runs when `last_window is not None` (where dicts are fresh).
- No code path writes back into these dicts.

**Verdict: Not a real problem.** Returning by reference is the correct design — copying
would waste memory for no benefit. The code is safe as-is.

---

## Summary

| # | Issue | Type | Priority | Action |
|---|-------|------|----------|--------|
| 1 | `check_preprocess_exog_multiseries` redundant copies | Performance | Low | Consider `copy` param |
| 2 | `align_series_and_exog_multiseries` NaN trimming | **Correctness** | **Medium** | Foundation-specific alignment |
| 3 | Double context-length trimming | False positive | — | None |
| 4 | Index interleaving type loss | False positive | — | None |
| 5 | `np.tile`/`np.repeat` for level_col | False positive | — | None |
| 6 | Returns adapter internals by reference | False positive | — | None |

**Only #2 warrants action** — the trailing NaN trimming is semantically wrong for
foundation models and creates a metadata inconsistency with `training_range_`.
Issue #1 is a minor optimization opportunity.
