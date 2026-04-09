# Bug Fix Plan: XGBoost `feature_types` Mismatch in Bayesian Search Cache

## Error observed

```
ValueError: ('feature types must have the same length as the number of data columns, ',
             'expected 31, got 94')
```

Triggered in `bayesian_search_forecaster` (and `bayesian_search_forecaster_multiseries`) when
`OneStepAheadFold` is used, the search space includes `lags` with multiple possible values, and
the estimator is `XGBRegressor` with categorical features enabled via
`forecaster.categorical_features`.

---

## Root-cause analysis

### How `configure_estimator_categorical_features` works for XGBoost

`configure_estimator_categorical_features` (`skforecast/utils/utils.py`) handles the four
supported estimator families differently:

| Estimator | Where categoricals are written | Lifecycle |
|-----------|-------------------------------|-----------|
| LightGBM  | `fit_kwargs['categorical_feature'] = [indices]` | Transient – recreated per split |
| CatBoost  | `fit_kwargs['cat_features'] = [indices]`         | Transient – recreated per split |
| XGBoost   | `estimator.set_params(feature_types=[...], enable_categorical=True)` | **Persistent on the estimator object** |
| sklearn HistGradientBoosting | `estimator.set_params(categorical_features=[...])` | **Persistent on the estimator object** |

LightGBM and CatBoost store the information in `fit_kwargs`, which is recreated fresh for each lag
combo inside `_train_test_split_one_step_ahead`. XGBoost and HistGradientBoosting mutate the
estimator directly via `set_params` because those frameworks do not accept the equivalent
information as a `fit()` argument.

### How the Bayesian-search cache works

`bayesian_search_forecaster` (with `OneStepAheadFold`) maintains a small dict cache
(`_cached_split`) keyed by lag combination. On a cache **miss** it calls
`_train_test_split_one_step_ahead`, which internally calls `configure_estimator_categorical_features`
and therefore sets `feature_types` on the estimator to match the number of columns produced by
those lags. On a cache **hit** it reuses the previously computed numpy arrays — but the
estimator-level params are **not** restored, so they still reflect the last lag combo that had a
cache miss.

### The failure sequence

```
Trial 0 — lags=[1,2,3]   (cache miss)
  → X_train.shape = (N, 31)
  → estimator.set_params(feature_types=[...31 items...])
  → stored: (X_train_31, ..., fit_kwargs_31)

Trial 1 — lags=[1..169]  (cache miss)
  → X_train.shape = (N, 94)
  → estimator.set_params(feature_types=[...94 items...])
  → stored: (X_train_94, ..., fit_kwargs_94)

Trial 2 — lags=[1..169]  (cache hit)
  → X_train with 94 cols reused ✓
  → estimator still has feature_types with 94 items ✓  (happened to be ok)

Trial 3 — lags=[1,2,3]   (cache hit)
  → X_train with 31 cols reused ✓
  → estimator still has feature_types[94] from Trial 1 ✗
  → XGBoost.fit() receives 31-column X but feature_types has 94 entries → CRASH
```

### Why `grid_search_forecaster` / `random_search_forecaster` are not affected

In those functions (lines ~430–478 of `_search.py`), `_train_test_split_one_step_ahead` is called
**once per lags value** in the outer loop. The inner loop only iterates over `params` — changing
hyperparameters such as `n_estimators` or `learning_rate` — which never calls
`configure_estimator_categorical_features`. The estimator's `feature_types` therefore stays in sync
with the cached arrays for the entire inner loop.

### Which estimator object is mutated

This is important for the fix. `_calculate_metrics_one_step_ahead` (`_utils.py`) uses:

```python
# _calculate_metrics_one_step_ahead (single-series path)
if type(forecaster).__name__ == 'ForecasterDirect':
    estimator = forecaster.estimators_[1]   # ← per-step clone
else:
    estimator = forecaster.estimator        # ← the shared estimator

# _predict_and_calculate_metrics_one_step_ahead_multiseries (multi-series path)
if type(forecaster).__name__ == 'ForecasterDirectMultiVariate':
    estimator = forecaster.estimators_[1]   # ← per-step clone
else:
    estimator = forecaster.estimator        # ← the shared estimator
```

`_train_test_split_one_step_ahead` mirrors this exactly:
- `ForecasterRecursive` / `ForecasterRecursiveMultiSeries`: calls `configure_estimator_categorical_features(self.estimator, ...)`
- `ForecasterDirect` / `ForecasterDirectMultiVariate`: calls `configure_estimator_categorical_features(self.estimators_[1], ...)`

The snapshot and restore must target the **same estimator object** that gets fitted, so both
checks must be combined in the helpers:
```python
if type(forecaster).__name__ in ('ForecasterDirect', 'ForecasterDirectMultiVariate'):
    estimator = forecaster.estimators_[1]
else:
    estimator = forecaster.estimator
```

### Affected code locations

| File | Approximate lines | Note |
|------|-----------|------|
| `skforecast/model_selection/_search.py` | ~820 | `_objective` in `bayesian_search_forecaster` (single-series) |
| `skforecast/model_selection/_search.py` | ~1910 | `_objective` in `bayesian_search_forecaster_multiseries` |

---

## Alternatives considered and rejected

### Option B — embed `set_params` inside `fit_kwargs` under a private key

`configure_estimator_categorical_features` would additionally write:
```python
fit_kwargs['__skforecast_estimator_set_params__'] = {'feature_types': [...], ...}
```
and the fitting functions (`_calculate_metrics_one_step_ahead` and
`_predict_and_calculate_metrics_one_step_ahead_multiseries`) would extract and apply it before
calling `estimator.fit()`.

**Rejected because:**
- `fit_kwargs` is cached as-is; extracting without mutation requires a dict copy on every fit call
  even when XGBoost is not involved (performance regression, however minor).
- Contaminates `fit_kwargs` — a `fit()`-argument dict — with non-fit parameters, making it
  semantically misleading.
- Requires Pipeline unwrapping logic to be duplicated in the fitting functions.

### Option C — change the return type of `_train_test_split_one_step_ahead`

Add a seventh return value `estimator_set_params: dict` and thread it through all callers.

**Rejected because:**
- Breaking API change to `_train_test_split_one_step_ahead`, which is semi-public.
- Requires updates in ForecasterRecursive, ForecasterDirect, `_search.py`, `_utils.py`,
  `_validation.py`, and their tests — far more invasive than the targeted fix.

---

## Chosen fix

### Core idea

A cache should also cache the side effects of the computation it stores. The side effect of
`_train_test_split_one_step_ahead` for XGBoost/HistGB is a mutation of the estimator via
`set_params`. On a cache **miss** we snapshot those params alongside the arrays; on a cache
**hit** we restore them before fitting.

### New helpers in `skforecast/utils/utils.py`

Add two small private functions immediately after `configure_estimator_categorical_features`.
Both receive a **forecaster**, not just an estimator, so they can apply the same `estimators_[1]`
vs `estimator` selection logic as `_calculate_metrics_one_step_ahead`.

---

**`_get_estimator_categorical_set_params(forecaster) -> dict`**

Returns the current values of the constructor-level params that
`configure_estimator_categorical_features` sets via `set_params`:
- XGBoost: `{'feature_types': ..., 'enable_categorical': ...}`
- HistGradientBoosting: `{'categorical_features': ...}`
- All others: `{}` (no-op on restore)

Selects the right estimator object using the same logic as `_calculate_metrics_one_step_ahead`
and `_predict_and_calculate_metrics_one_step_ahead_multiseries`:
```python
if type(forecaster).__name__ in ('ForecasterDirect', 'ForecasterDirectMultiVariate'):
    estimator = forecaster.estimators_[1]
else:
    estimator = forecaster.estimator
# unwrap Pipeline if needed
if isinstance(estimator, Pipeline):
    estimator = estimator[-1]
```

Note: `ForecasterDirectMultiVariate` must be included alongside `ForecasterDirect` because both
use per-step estimator clones (`estimators_[1]`) rather than the shared `estimator` template;
this is confirmed in both `_train_test_split_one_step_ahead` and the downstream fitting functions.

---

**`_restore_estimator_categorical_set_params(forecaster, params) -> None`**

Calls `estimator.set_params(**params)` on the same estimator selection logic. No-op when
`params` is empty.

---

### Changes to `skforecast/model_selection/_search.py`

1. **Import** the two new helpers from `..utils`.

2. **Single-series `_objective`** — cache-miss branch stores the snapshot; cache-hit branch
   restores it:

   ```python
   # cache miss
   _cached_split[lags_key] = (
       X_train, y_train, X_test, y_test, sample_weight, fit_kwargs,
       _get_estimator_categorical_set_params(forecaster_search)   # ← new
   )

   # cache hit
   (
       X_train, y_train, X_test, y_test, sample_weight, fit_kwargs,
       _estimator_cat_params                                       # ← new
   ) = _cached_split[lags_key]
   _restore_estimator_categorical_set_params(                      # ← new
       forecaster_search, _estimator_cat_params
   )
   ```

3. **Multi-series `_objective`** — identical pattern on its own `_cached_split`.

### No changes needed elsewhere

- `_train_test_split_one_step_ahead` — correct as-is.
- `configure_estimator_categorical_features` — correct as-is.
- `_calculate_metrics_one_step_ahead` — correct as-is.
- `grid_search_forecaster` / `random_search_forecaster` — not affected (see above).
- LightGBM / CatBoost paths — not affected (they use `fit_kwargs`, not `set_params`).

---

## Tests added

In `skforecast/model_selection/tests/tests_search/`:

**`test_bayesian_search_forecaster_xgboost_categorical_no_ValueError_on_cache_hit`**
(`test_bayesian_search_forecaster.py`)
- `ForecasterRecursive` + `XGBRegressor` with `categorical_features=['cat_feat']`
- `search_space` includes two lag lists: `[1, 2, 3]` and `[1, 2, 3, 4, 5]` — forces cache hits on repeated lag combos across 6 trials
- `OneStepAheadFold`, `initial_train_size=35`, `n_trials=6`, `random_state=123`
- Asserts results DataFrame is non-empty and spot-checks numeric columns with `atol=1e-4`

**`test_bayesian_search_forecaster_multiseries_xgboost_categorical_no_ValueError_on_cache_hit`**
(`test_bayesian_search_forecaster_multiseries.py`)
- Same scenario for `bayesian_search_forecaster_multiseries` using `ForecasterRecursiveMultiSeries`

**`test__get_estimator_categorical_set_params`**
(`test__get_estimator_categorical_set_params.py`)
- Parametrized: `LinearRegression` and `LGBMRegressor` → empty dict
- `XGBRegressor` fresh defaults → `{'feature_types': None, 'enable_categorical': False}`
- `XGBRegressor` after `set_params` → reflects current values
- `HistGradientBoostingRegressor` fresh defaults → `{'categorical_features': 'from_dtype'}`
- `HistGradientBoostingRegressor` after `set_params` → reflects current value
- Pipeline with XGBoost → unwraps last step
- `ForecasterDirect` (fitted) → reads from `estimators_[1]`, not from `forecaster.estimator`

**`test__restore_estimator_categorical_set_params`**
(`test__restore_estimator_categorical_set_params.py`)
- Empty params → no-op (estimator unchanged)
- `XGBRegressor` → `feature_types` and `enable_categorical` restored after overwrite
- `HistGradientBoostingRegressor` → `categorical_features` restored after overwrite
- Pipeline with XGBoost → unwraps last step and restores
- `ForecasterDirect` (fitted) → restores on `estimators_[1]`, template `estimator` untouched

**Note on test setup**: `ForecasterRecursive` deep-copies the estimator on init, so restore
tests work directly via `forecaster.estimator` rather than the original variable.

---

## File change summary

```
skforecast/utils/utils.py
  + _get_estimator_categorical_set_params(forecaster) -> dict
  + _restore_estimator_categorical_set_params(forecaster, params) -> None

skforecast/utils/__init__.py
  + explicit export of _get_estimator_categorical_set_params
  + explicit export of _restore_estimator_categorical_set_params
  (private helpers are not covered by `from .utils import *`, so explicit exports are required)

skforecast/model_selection/_search.py
  ~ import the two new helpers
  ~ bayesian_search_forecaster: extend cache tuple (6 → 7 elements) to store/restore params
  ~ bayesian_search_forecaster_multiseries: same (cache tuple grows by 1 element)

skforecast/model_selection/tests/tests_search/test_bayesian_search_forecaster.py
  + added XGBRegressor and warnings imports
  + test_bayesian_search_forecaster_xgboost_categorical_no_ValueError_on_cache_hit

skforecast/utils/tests/tests_utils/test__get_estimator_categorical_set_params.py
  + 8 tests covering: unsupported estimators, XGBoost defaults/post-configure,
    HistGBR defaults/post-configure, Pipeline unwrapping, ForecasterDirect delegation

skforecast/utils/tests/tests_utils/test__restore_estimator_categorical_set_params.py
  + 5 tests covering: empty-params no-op, XGBoost restore, HistGBR restore,
    Pipeline unwrapping, ForecasterDirect delegation
```

---

## Implementation status: COMPLETE

All changes applied. Test run result: **105 passed, 0 failures** (92 original + 13 new helper tests).

```
pytest skforecast/model_selection/tests/tests_search/test_bayesian_search_forecaster.py \
       skforecast/model_selection/tests/tests_search/test_bayesian_search_forecaster_multiseries.py \
       skforecast/utils/tests/tests_utils/test_configure_estimator_categorical_features.py -q
```

### Implementation notes

- Cannot call `configure_estimator_categorical_features` again on cache hits: it emits spurious
  `IgnoredArgumentWarning` whenever `prev_feature_types is not None` (i.e., when the estimator
  already has `feature_types` set). Using `set_params` directly avoids this.
- `ForecasterRecursiveClassifier` uses `self.estimator` and correctly falls into the `else` branch
  of the estimator-selection logic in both helpers.
- `Pipeline` is already imported at the top of `utils.py` (used by other functions), so no new
  import was required.
- The two `_cached_split` dicts in `_search.py` are fully independent (`{}` literals at lines 776
  and 1869 respectively). Each stores its own snapshot; the fix is self-contained in each closure.
