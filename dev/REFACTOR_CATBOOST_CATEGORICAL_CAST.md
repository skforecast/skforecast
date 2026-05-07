# Refactor: CatBoost categorical-column cast

> Bug fix + utility extraction across all forecasters that support
> `categorical_features`.

## TL;DR

**Bug.** When the estimator is a CatBoost model and either the
`OneStepAheadFold` validation slice contains a category not seen in
training, or the categorical exog itself contains `NaN`, `fit()` /
`_train_test_split_one_step_ahead()` crash with:

```text
ValueError: cannot convert float NaN to integer
```

**Root cause.** Four forecasters (`ForecasterRecursive`,
`ForecasterDirect`, `ForecasterDirectMultiVariate`,
`ForecasterRecursiveClassifier`) carry duplicated CatBoost-specific cast
code that does `X[:, cat_idx].astype(int)` after the `OrdinalEncoder`
has potentially produced `NaN`. `astype(int)` on `NaN` raises
`ValueError`. `ForecasterRecursiveMultiSeries` already handled this
correctly via `fillna(-1)`.

**Fix.** Two utilities (numpy + DataFrame variants) factor the cast
into a single home, applying the existing safe pattern (`NaN → -1`)
across all forecasters and the multiseries one-step-ahead search path.

**Behavior change.** Strictly limited to CatBoost: previously-crashing
inputs now train/predict cleanly. Non-CatBoost estimators
(LightGBM, XGBoost, HistGradientBoosting, LinearRegression, …) are
**unaffected** — the utilities short-circuit before doing any work.

**Validation.** 26 new unit tests for the utilities, plus all four
end-to-end bug repros now succeed. ~2 660 existing forecaster +
model_selection tests still pass.

---

## 1. The bug

### 1.1 Where the broken code lived

The duplicated CatBoost-specific cast appeared at **8 sites** across
4 forecasters (each forecaster duplicated the block in `fit()` and in
`_train_test_split_one_step_ahead()`):

| Forecaster | Sites |
|---|---|
| `ForecasterRecursive` | `fit` (line 1283-1285) and `_train_test_split_one_step_ahead` (line 1129-1133) |
| `ForecasterDirect` | `_fit_one_step_estimator` (line 137-143) and `_train_test_split_one_step_ahead` (line 1610-1618) |
| `ForecasterDirectMultiVariate` | `_train_test_split_one_step_ahead` (line 1760-1768) — `fit` reuses `_fit_one_step_estimator` from `ForecasterDirect` |
| `ForecasterRecursiveClassifier` | `fit` (line 1430-1438) and `_train_test_split_one_step_ahead` (line 1248-1257) |

Plus a DataFrame-shape variant in:

| File | Site |
|---|---|
| `ForecasterRecursiveMultiSeries.fit` | line 2020-2035 — already safe (used `fillna(-1)`) |
| `model_selection/_utils.py` | line 1506-1528 — already safe (used `fillna(-1)`) |

### 1.2 The crash, line by line

The numpy-based cast in `ForecasterRecursive` (mirrored in the other
three numpy-shape forecasters):

```python
if (
    'cat_features' in fit_kwargs
    and type(self.estimator).__name__ == 'CatBoostRegressor'
):
    cat_idx = np.array(fit_kwargs['cat_features'])
    X_train = X_train.astype(object)
    X_train[:, cat_idx] = X_train[:, cat_idx].astype(int)   # <-- crash here
```

The categorical columns at this point come from this `OrdinalEncoder`,
constructed once in `__init__`:

```python
# _forecaster_recursive.py:367-372
self.categorical_encoder = OrdinalEncoder(
    dtype                 = float,
    handle_unknown        = 'use_encoded_value',
    unknown_value         = np.nan,            # <-- NaN for unseen
    encoded_missing_value = np.nan,            # <-- NaN for missing
).set_output(transform="pandas")
```

So `NaN` enters the categorical columns whenever:

- the user has `NaN` in their categorical exog (`encoded_missing_value=np.nan`), or
- the encoder has been *fitted* on training data and now *transforms* a
  test slice that contains a level it hasn't seen
  (`unknown_value=np.nan`).

`np.array([np.nan]).astype(int)` raises
`ValueError: cannot convert float NaN to integer`.

### 1.3 Two reachable code paths

**Path A — `fit()` with NaN in categorical exog.**

The default is `dropna_from_series=False`, which keeps NaN-bearing rows
in the training matrices. The encoder then propagates the NaN, and the
cast crashes immediately on `forecaster.fit(...)`.

```python
# Repro
n = 50
y = pd.Series(np.arange(n, dtype=float),
              index=pd.date_range('2024-01-01', periods=n, freq='D'))
exog = pd.DataFrame({'cat': ['A', 'B', np.nan] * 16 + ['A', 'B']},
                    index=y.index)
ForecasterRecursive(
    estimator=CatBoostRegressor(iterations=3, verbose=0,
                                allow_writing_files=False),
    lags=3,
).fit(y=y, exog=exog)
# CRASH on fit: ValueError: cannot convert float NaN to integer
```

**Path B — `OneStepAheadFold` validation with unseen categories.**

`_train_test_split_one_step_ahead()` calls `_create_train_X_y()` twice:

1. First call: `is_fitted=False` → `categorical_encoder.fit_transform`
   on the train slice. No NaN.
2. `self.is_fitted = True`.
3. Second call: `is_fitted=True` → `categorical_encoder.transform` on
   the test slice. Unseen levels become `NaN`.

The cast block then runs on both `X_train` and `X_test`:

```python
X_test = X_test.astype(object)
X_test[:, cat_idx] = X_test[:, cat_idx].astype(int)  # <-- crash
```

Reproduced live:

```text
--- ForecasterRecursive             → CRASH
--- ForecasterDirect                → CRASH
--- ForecasterDirectMultiVariate    → CRASH
--- ForecasterRecursiveClassifier   → CRASH
--- ForecasterRecursiveMultiSeries  → safe (different code path)
```

### 1.4 User-visible footprint

Anything that uses `OneStepAheadFold` with a CatBoost forecaster and
categorical exog hits Path B:

- `grid_search_forecaster`, `random_search_forecaster`,
  `bayesian_search_forecaster` and their multiseries / multivariate
  variants,
- `backtesting_forecaster` paths that internally route through
  `_train_test_split_one_step_ahead`.

Path A is hit on `fit()` whenever the categorical exog has any `NaN`
and `dropna_from_series=False` (the default).

The probability of unseen categories in a real validation window is
non-trivial — typical CV holdouts on real data routinely contain rare
levels — so this is not a corner case.

### 1.5 What was *not* broken

- **`predict()`.** The predict path re-encodes via `transform` (so it
  produces NaN for unseen levels), but does **not** apply the int cast.
  `CatBoostRegressor.predict()` accepts float NaN in cat columns and
  treats it as a novel category at inference time, so the call
  succeeds. Verified live.
- **`fit()` without NaN.** Without NaN in either input or test slice,
  the int cast succeeds. The crash only manifests when NaN enters the
  matrix.

### 1.6 Why `ForecasterRecursiveMultiSeries` was already safe

The multiseries fit operates on a `DataFrame`, not a numpy array, and
its cast block (`fit`, line 2020-2035) was written with the right
defensive pattern:

```python
cat_cols = [X_train_features_names_out_[i] for i in fit_kwargs['cat_features']]
for col in cat_cols:
    if hasattr(X_train_estimator[col].dtype, 'categories'):
        X_train_estimator[col] = X_train_estimator[col].cat.codes.astype(int)
    else:
        X_train_estimator[col] = X_train_estimator[col].fillna(-1).astype(int)
```

Two dtypes are handled:

- `pandas.Categorical` (from `_level_skforecast` when
  `encoding='ordinal_category'`) → `.cat.codes`, which encodes missing
  as `-1` by default.
- `float` with `NaN` (from the `OrdinalEncoder` applied to exog cats)
  → `.fillna(-1).astype(int)`.

The same safe pattern lives in `model_selection/_utils.py`:1506-1528
for the multiseries one-step-ahead search path. The numpy-shape
forecasters never received this fix.

---

## 2. The solution

### 2.1 Why `-1`?

CatBoost has no concept of "missing" for `cat_features` — it always
expects an integer category id. The `OrdinalEncoder` outputs ids in
`0..n-1`, so `-1` is guaranteed not to collide with the encoder's
range. CatBoost stores its own internal mapping from supplied integer
ids; novel ids at predict time are handled via the target-encoding
fallback.

### 2.2 Why not change the encoder's `unknown_value`?

The encoder is shared across all estimator families. **LightGBM,
XGBoost (with `tree_method='hist'`), and HistGradientBoosting**
explicitly use `NaN` as a meaningful "missing-category" bucket — moving
away from `NaN` would change behavior for those estimators too. The fix
belongs at the CatBoost-specific cast site, not at the encoder.

### 2.3 The two utilities

Added to `skforecast/utils/utils.py` next to
`configure_estimator_categorical_features` (the existing single source
of truth for vendor-specific categorical wiring).

#### `cast_catboost_categorical_columns(X, fit_kwargs, estimator)` — numpy

```python
def cast_catboost_categorical_columns(
    X: np.ndarray,
    fit_kwargs: dict[str, object],
    estimator: object,
) -> np.ndarray:
    if 'cat_features' not in fit_kwargs:
        return X

    target_estimator = estimator
    if isinstance(target_estimator, Pipeline):
        target_estimator = target_estimator[-1]
    if type(target_estimator).__module__.split('.')[0] != 'catboost':
        return X

    cat_idx = np.asarray(fit_kwargs['cat_features'])
    X = X.astype(object)
    cat_block = np.asarray(X[:, cat_idx], dtype=float)
    X[:, cat_idx] = np.nan_to_num(cat_block, nan=-1).astype(int)

    return X
```

Notes on the implementation:

- `np.asarray(..., dtype=float)` is needed because the slice is already
  `object`-dtype after the prior `astype(object)` and `nan_to_num`
  requires a float array.
- `nan=-1` matches the multiseries convention.
- `.astype(int)` after the fill is now safe — no NaN remains.
- `astype(object)` returns a fresh array, so the input `X` is never
  mutated.

#### `cast_catboost_categorical_columns_dataframe(X, fit_kwargs, estimator, feature_names)` — DataFrame

```python
def cast_catboost_categorical_columns_dataframe(
    X: pd.DataFrame,
    fit_kwargs: dict[str, object],
    estimator: object,
    feature_names: list[str],
) -> pd.DataFrame:
    if 'cat_features' not in fit_kwargs:
        return X

    target_estimator = estimator
    if isinstance(target_estimator, Pipeline):
        target_estimator = target_estimator[-1]
    if type(target_estimator).__module__.split('.')[0] != 'catboost':
        return X

    X = X.copy()
    cat_cols = [feature_names[i] for i in fit_kwargs['cat_features']]
    for col in cat_cols:
        if hasattr(X[col].dtype, 'categories'):
            X[col] = X[col].cat.codes.astype(int)
        else:
            X[col] = X[col].fillna(-1).astype(int)

    return X
```

Notes on the implementation:

- `X = X.copy()` is up-front because column assignment would otherwise
  mutate the input. This matches the numpy variant's "returns a fresh
  object" contract and is required for the multiseries one-step-ahead
  search loop, which reuses the matrices produced once by
  `_train_test_split_one_step_ahead` across hyperparameter trials.
- Two dtype branches mirror the safe `MultiSeries.fit` pattern:
  `pandas.Categorical` → `.cat.codes` (NaN → `-1`), float-with-NaN →
  `fillna(-1)`.
- `feature_names` lets the function translate integer indices in
  `fit_kwargs['cat_features']` to column labels — same lookup the
  in-place code did via `X.columns[i]`.

### 2.4 Why two functions instead of one with type-sniffing

Two reasons:

1. **The branches don't share enough code.** The numpy variant needs
   `astype(object)` + integer index slicing; the DataFrame variant
   needs `cat.codes` vs `fillna` per-column. A single function would
   essentially be `if isinstance(X, pd.DataFrame): … else: …`.
2. **Callers already know which they have.** The numpy-shape
   forecasters operate on `np.ndarray`, the DataFrame-shape
   `MultiSeries` operates on `pd.DataFrame`. No introspection needed at
   the call site.

Two small functions with explicit names beat one polymorphic helper
that has to sniff input types.

### 2.5 Pipeline support

Both utilities call `Pipeline[-1]` to inspect the *last* step's module,
matching exactly what `configure_estimator_categorical_features` does
([utils.py:547-548](../skforecast/utils/utils.py#L547-L548)). This:

- Removes a regressor-vs-classifier `__name__` inspection that
  previously hardcoded both `'CatBoostRegressor'` and
  `'CatBoostClassifier'` separately.
- Converges the two helpers (configure + cast) on a single contract.
- Eliminates one inconsistency between them (the previous cast code
  did **not** unwrap Pipelines, so a Pipeline-wrapped CatBoost was
  silently skipped — a latent bug, not in scope here).

---

## 3. Behavior change matrix

The refactor's surface area is **strictly limited to CatBoost**. Every
other estimator family follows an unchanged path.

| Scenario | Before | After |
|---|---|---|
| LightGBM + cat exog (no NaN) | float NaN passed through, LightGBM uses NaN bucket | unchanged — utility no-ops |
| LightGBM + cat exog with NaN | same NaN bucket behavior | unchanged |
| XGBoost + cat exog | `set_params(enable_categorical=True)` + float matrix | unchanged — utility no-ops |
| HistGBR + cat exog | `set_params(categorical_features=...)` + float matrix | unchanged — utility no-ops |
| `LinearRegression` + cat exog | passthrough; `categorical_features=None` is required for non-supporting estimators | unchanged |
| CatBoost + clean cat exog (all levels seen, no NaN) | int cast works | same result (no NaN to fill) |
| CatBoost + cat exog with NaN | **CRASH** | **fixed**: NaN → `-1` then int cast |
| CatBoost + unseen level in CV split | **CRASH** | **fixed**: NaN → `-1` then int cast |
| `categorical_features=None` (any estimator) | `configure_estimator_categorical_features` not called → no `cat_features` key | unchanged — utility no-ops |
| Pipeline wrapping CatBoost | latent bug (cat cast skipped) | latent bug now partially exposed: cast runs, but `fit_kwargs['cat_features']` still needs sklearn Pipeline `step__param` forwarding (out of scope for this refactor) |

The only rows that change are the two CatBoost CRASH cases — exactly
the bug being fixed.

### Why non-CatBoost estimators are guaranteed unaffected

The first guard inside both utilities is:

```python
if 'cat_features' not in fit_kwargs:
    return X
```

Looking at `configure_estimator_categorical_features`:

| Estimator | What lands in `fit_kwargs` | `'cat_features'` key? |
|---|---|---|
| LightGBM | `categorical_feature` (singular) | No |
| CatBoost | `cat_features` | **Yes** |
| XGBoost | nothing (uses `set_params`) | No |
| HistGradientBoosting | nothing (uses `set_params`) | No |
| Anything else / `categorical_features=None` | nothing | No |

Only CatBoost ever puts `'cat_features'` into `fit_kwargs`, so for
every other estimator the utilities return on the first line — no
allocation, no `astype`, no copy. Bit-identical to today.

The second guard (`module == 'catboost'`) is belt-and-suspenders
against a user manually stuffing `cat_features` into their own
`fit_kwargs` while pairing it with, say, LightGBM — that's a user
error the previous code also silently mishandled.

---

## 4. Implementation details

### 4.1 New code

**`skforecast/utils/utils.py`** — added two functions immediately after
`configure_estimator_categorical_features`:

- `cast_catboost_categorical_columns` (numpy, ~30 lines)
- `cast_catboost_categorical_columns_dataframe` (pandas, ~30 lines)

Both are picked up by the `from .utils import *` re-export in
`skforecast/utils/__init__.py` — no further plumbing needed.

### 4.2 Call-site replacements

| File | Before / after |
|---|---|
| `skforecast/recursive/_forecaster_recursive.py` | 2 sites (`fit`, `_train_test_split_one_step_ahead`) → call `cast_catboost_categorical_columns` |
| `skforecast/direct/_forecaster_direct.py` | 2 sites (module-level `_fit_one_step_estimator`, `_train_test_split_one_step_ahead`) → call utility |
| `skforecast/direct/_forecaster_direct_multivariate.py` | 1 site (`_train_test_split_one_step_ahead`; `fit` reuses `_fit_one_step_estimator`) → call utility |
| `skforecast/recursive/_forecaster_recursive_classifier.py` | 2 sites (`fit`, `_train_test_split_one_step_ahead`) → call utility |
| `skforecast/recursive/_forecaster_recursive_multiseries.py` | 1 site (`fit`, DataFrame branch) → call `cast_catboost_categorical_columns_dataframe` |
| `skforecast/model_selection/_utils.py` | 1 site (multiseries one-step-ahead) → call `cast_catboost_categorical_columns_dataframe` |

Each replacement reduces a 5-9-line block to a single utility call.

### 4.3 Imports added

Each touched module imports the relevant utility from `skforecast.utils`:

- numpy variant: `cast_catboost_categorical_columns`
- DataFrame variant: `cast_catboost_categorical_columns_dataframe`

These are imported next to the existing
`configure_estimator_categorical_features` import to keep the
categorical-handling helpers grouped.

### 4.4 Net delta

- **Added**: ~70 lines (two utilities + docstrings).
- **Removed**: ~85 lines (8 numpy cast blocks + 2 DataFrame cast blocks
  collapsed to single utility calls).
- **Net**: smaller and converged on one cast policy across five
  forecasters.

---

## 5. Tests

### 5.1 New unit tests for the utilities

Two new test files in `skforecast/utils/tests/tests_utils/`, modeled on
the existing `test_configure_estimator_categorical_features.py`:

#### `test_cast_catboost_categorical_columns.py` (numpy) — 12 tests

| Group | Tests |
|---|---|
| No-op paths | `cat_features` missing from `fit_kwargs`; non-CatBoost estimators (`LinearRegression`, `LGBMRegressor`, `XGBRegressor`, `HistGradientBoostingRegressor`) parametrized |
| Happy path | cast to int when no NaN |
| Bug-fix regression | NaN values become `-1` (the original crash signature) |
| Estimator coverage | `CatBoostClassifier` triggers the same cast |
| Multi-column | NaN handling per column |
| No mutation | input array unchanged after the call |
| Pipeline | unwraps last step, casts when CatBoost, no-op otherwise |

#### `test_cast_catboost_categorical_columns_dataframe.py` (pandas) — 14 tests

Same scenarios as above, plus DataFrame-specific cases:

- **Float categorical columns** with and without NaN.
- **`pandas.Categorical` dtype**: uses `.cat.codes`, including the
  default NaN → `-1` encoding.
- **Mixed dtypes** in a single call (Categorical + float-with-NaN).
- **Copy contract**: input DataFrame is *not* mutated, and the returned
  object is a fresh frame (the property the hyperparameter-search loop
  relies on to reuse `X_train`/`X_test` across trials).

### 5.2 End-to-end bug repros (development scratch)

Four scripts under `c:\tmp/` were used to validate the fix end-to-end
during development:

- `repro_catboost_nan.py` — `_train_test_split_one_step_ahead` with
  unseen levels (single `ForecasterRecursive`).
- `repro_catboost_nan_others.py` — same scenario across `Direct`,
  `DirectMV`, `Classifier`.
- `repro_catboost_fit_nan.py` — `fit()` with NaN in cat exog (Path A).
- `repro_catboost_nan_predict.py` — `predict()` with unseen levels
  (verifies it was already working).

All four pass after the fix:

```text
fit OK
predict OK: [42.15662977 42.15662977 42.15662977]
OK, no crash
--- ForecasterDirect ---             OK
--- ForecasterDirectMultiVariate --- OK
--- ForecasterRecursiveClassifier -- OK
```

These scripts are scratch-only (not committed) — the new unit tests
cover the same scenarios at the utility level, and adding regression
tests at the forecaster level (e.g., one per forecaster covering an
unseen category in the validation slice) is a recommended follow-up.

### 5.3 Regression tests

All existing test suites for the affected modules pass:

| Suite | Result |
|---|---|
| `tests_forecaster_recursive` | 319 passed |
| `tests_forecaster_direct` + `tests_forecaster_direct_multivariate` + `tests_forecaster_recursive_classifier` + `tests_forecaster_recursive_multiseries` | 1641 passed, 1 skipped |
| `model_selection/tests` | 703 passed |
| New utility tests | 26 passed |

---

## 6. Loose ends and follow-ups

### 6.1 Pipeline + CatBoost is still partially broken

The previous code did not unwrap `Pipeline` when checking for
CatBoost — `type(self.estimator).__name__ == 'CatBoostRegressor'`
returned `False` for `Pipeline([..., CatBoostRegressor()])`, so the
cast was silently skipped. This refactor *does* unwrap the Pipeline,
so the cast now runs.

However, `configure_estimator_categorical_features` writes
`fit_kwargs['cat_features']` (not `fit_kwargs['<step>__cat_features']`),
and sklearn `Pipeline.fit()` expects the latter form to forward kwargs
to a specific step. So calling `pipeline.fit(X, y, cat_features=[...])`
still raises a sklearn-side error. **Out of scope for this refactor.**

A follow-up would either:

- detect the Pipeline case in `configure_estimator_categorical_features`
  and write the kwarg as `<final_step>__cat_features`, or
- document that Pipeline + CatBoost is unsupported and raise an
  explicit error.

### 6.2 Predict-path symmetry

`predict()` does *not* apply the cast today. CatBoost's `predict()`
accepts float NaN in `cat_features` and treats them as novel categories
gracefully — verified in `repro_catboost_nan_predict.py`. So this is
not a bug.

A defensive option would be to also apply the cast at predict time for
parity with the train side (and as insurance against future CatBoost
versions tightening predict-side validation). Optional — not done
here.

### 6.3 Recommended forecaster-level regression tests

The new unit tests cover the utilities exhaustively. A natural
follow-up is to add one regression test per forecaster (e.g. under
`tests_forecaster_*/test_train_test_split_one_step_ahead.py`) that:

1. Constructs `exog` such that the validation slice contains a
   category not seen in training (Path B), or `fit` data with NaN in
   cat exog (Path A).
2. Wraps it in `CatBoostRegressor` (or `CatBoostClassifier` for the
   classifier).
3. Asserts the call doesn't raise and that the returned matrix has no
   NaN in the categorical columns.

That would close the loop at the integration level for each
forecaster.

### 6.4 `ForecasterFoundation` capability flag

Unrelated to this refactor but discovered during the review:
`ForecasterFoundation` declares `supports_categorical_features=True`
but has no `categorical_features` argument or encoder. The flag
appears to advertise that the underlying adapter (Chronos / TabICL)
can ingest categorical exog directly. This is a documentation /
contract inconsistency, not a runtime bug. Worth flipping the flag or
plumbing a real `categorical_features` argument for API uniformity.

---

## 7. Why this design is the right shape

- **One source of truth.** Five forecasters and one search helper now
  call a single utility for the cast. Future CatBoost API changes
  (e.g. `enable_categorical` defaults, sparse-matrix support) are
  fixed in one place.
- **No behavior change for non-CatBoost.** The first-line guard
  (`'cat_features' not in fit_kwargs`) ensures every other estimator
  family takes a strict no-op path.
- **Existing safe pattern reused.** The `NaN → -1` strategy is what
  `ForecasterRecursiveMultiSeries` already used in production — no
  new conventions introduced.
- **Testable in isolation.** 26 unit tests cover both utilities at all
  branches (early returns, both dtypes for the DataFrame variant,
  Pipeline unwrapping, classifier vs regressor, mutation guarantee,
  NaN handling).
- **Drops a regressor-vs-classifier name inspection.** Previous code
  hardcoded `'CatBoostRegressor'` and `'CatBoostClassifier'`
  separately; the module-based check (`module == 'catboost'`) covers
  both, matching `configure_estimator_categorical_features`'s
  convention.
