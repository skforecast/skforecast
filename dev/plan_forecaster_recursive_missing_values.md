# Plan: Add Missing Values Support to ForecasterRecursive

## Context

`ForecasterRecursiveMultiSeries` tolerates NaN values in training series via its
`dropna_from_series` parameter. When a series has interspersed NaNs, the NaN values
propagate naturally into the lag matrix (`X_train`). The forecaster then:

1. Always drops rows where `y_train` is NaN (target cannot be NaN).
2. Depending on `dropna_from_series`: either drops rows with NaN in `X_train` (if `True`)
   or keeps them and issues a warning (if `False`, relying on the estimator to handle NaNs,
   e.g. LightGBM or XGBoost which natively support NaN features).

`ForecasterRecursive` currently rejects NaN in `y` immediately via `check_y(y=y)` in
`_create_train_X_y()`, raising a `ValueError`.

**Goal:** Mirror the multiseries strategy in `ForecasterRecursive`.

**Key structural difference:** `ForecasterRecursive` assembles `X_train` and `y_train` as
**numpy arrays** (not pandas DataFrames). Row filtering therefore uses boolean numpy masking
(`array[mask]`) rather than pandas `.iloc[mask]`. `train_index` (a pandas `Index`) must also
be filtered with `train_index[mask]`.

---

## Phase 1 — Extend `check_y()` in `skforecast/utils/utils.py`

**Function:** `check_y()` (currently at line ~718, but may shift).

**Change:** Add `allow_nan: bool = False` parameter, mirroring the existing
`check_exog(allow_nan=...)` API. When `allow_nan=True`, the `ValueError` for NaN values is
skipped.

```python
def check_y(
    y: Any,
    series_id: str = "`y`",
    allow_nan: bool = False    # NEW
) -> None:
    ...
    if not allow_nan:          # guard the existing NaN check
        if y.isna().to_numpy().any():
            raise ValueError(f"{series_id} has missing values.")
```

**Default `False`** → all other callers (`ForecasterDirect`, `ForecasterStats`,
`ForecasterRnn`, `ForecasterRecursiveClassifier`, etc.) are **unaffected**.

---

## Phase 2 — Changes to `_forecaster_recursive.py`

### 2a. `__init__()` signature and body

Add `dropna_from_series: bool = False` as a new parameter (after `differentiation`,
before `fit_kwargs`, to mirror multiseries parameter order).

```python
def __init__(
    self,
    ...
    differentiation: int | None = None,
    dropna_from_series: bool = False,      # NEW
    fit_kwargs: dict[str, object] | None = None,
    binner_kwargs: dict[str, object] | None = None,
    ...
)
```

Store it:

```python
self.dropna_from_series = dropna_from_series
```

Update the forecaster tag:

```python
"handles_missing_values_series": True,   # was False
```

### 2b. `_create_train_X_y()` — NaN handling

**Step 1:** Replace the blocking `check_y` call (currently at line ~749):

```python
# Before
check_y(y=y)

# After
check_y(y=y, allow_nan=True)
```

**Step 2:** Change the post-transformation `check_exog` call from strict to permissive
(currently at line ~849). This allows NaN values in exogenous variables to propagate to
`X_train` and be handled by the same NaN-filtering logic below, consistent with
`ForecasterRecursiveMultiSeries` behaviour:

```python
# Before
check_exog(exog=exog, allow_nan=False)

# After
check_exog(exog=exog, allow_nan=True)
```

**Step 3:** After the final `X_train` numpy array and `y_train` array are assembled
(after the existing `np.concatenate` / single-component assignment block), insert the
NaN-row-filtering logic, before the `return` statement:

```python
# --- NaN row filtering (interspersed NaN support) ---
if np.isnan(y_train).any():
    mask = ~np.isnan(y_train)
    y_train = y_train[mask]
    X_train = X_train[mask]
    train_index = train_index[mask]
    warnings.warn(
        "NaNs detected in `y_train`. They have been dropped because the "
        "target variable cannot have NaN values. Same rows have been "
        "dropped from `X_train` to maintain alignment. This is caused by "
        "interspersed NaNs in the time series.",
        MissingValuesWarning
    )

if self.dropna_from_series:
    nan_rows = pd.isna(X_train).any(axis=1)
    if nan_rows.any():
        mask = ~nan_rows
        X_train = X_train[mask]
        y_train = y_train[mask]
        train_index = train_index[mask]
        warnings.warn(
            "NaNs detected in `X_train`. They have been dropped. If "
            "you want to keep them, set `forecaster.dropna_from_series = False`. "
            "Same rows have been removed from `y_train` to maintain alignment. "
            "This is caused by interspersed NaNs in the time series.",
            MissingValuesWarning
        )
else:
    if pd.isna(X_train).any():
        warnings.warn(
            "NaNs detected in `X_train`. Some estimators do not allow "
            "NaN values during training. If you want to drop them, "
            "set `forecaster.dropna_from_series = True`.",
            MissingValuesWarning
        )

if len(y_train) == 0:
    raise ValueError(
        "All samples have been removed due to NaNs. Set "
        "`forecaster.dropna_from_series = False` or review `y` values."
    )
```

> **Note on `pd.isna` vs `np.isnan` for `X_train`:** `np.isnan()` raises a `TypeError` on
> object/string arrays. Since `X_train` can contain columns originating from non-numeric
> `exog` columns (boolean, string, object), `pd.isna()` is used instead — it handles any
> dtype safely. `y_train` is always a numeric ndarray, so `np.isnan()` remains correct there.
> In the `dropna_from_series=True` branch, `pd.isna(X_train)` is computed only **once** via
> `.any(axis=1)` (producing the per-row mask), and the result is reused for the condition
> check via `.any()` — avoiding a second full-array pass.
>
> **Note on 2D safety:** `X_train` is always 2D at this point. Even when only lags or only
> window features are present, `_create_lags` returns a 2D ndarray and window feature arrays
> are 2D. The `np.concatenate(..., axis=1)` call (or the single-element assignment) preserves
> 2D shape. So `.any(axis=1)` and `X_train[mask]` are always safe.

### 2c. `set_in_sample_residuals()`

```python
# Before
check_y(y=y)

# After
check_y(y=y, allow_nan=True)
```

This method calls `_create_train_X_y()` internally. Allowing NaN through here ensures
consistency; the filtering happens inside `_create_train_X_y()`.

### 2d. Docstrings — class-level and method-level

**Class Attributes section** — add entry:

```
dropna_from_series : bool
    If `True`, rows of `X_train` containing NaN values are dropped during training.
    If `False`, rows with NaN are kept and a `MissingValuesWarning` is issued.
```

**`__init__` Parameters section** — add entry:

```
dropna_from_series : bool, default False
    If `True`, rows of `X_train` containing NaN values are dropped during
    training. If `False`, rows with NaN are kept and a `MissingValuesWarning`
    is issued. Relevant when `y` contains interspersed NaN values or when
    `exog` contains NaN values that propagate to `X_train`.
```

**`_create_train_X_y()` docstring** — add to Notes:

```
If `y` or `exog` contain interspersed NaN values, rows where `y_train` is NaN
are always removed. Rows where `X_train` contains NaN (from lagged NaN in `y`
or from NaN in `exog`) are removed only if `dropna_from_series=True`; otherwise
a warning is issued.
```

---

## Phase 3 — Tests

### 3a. `skforecast/utils/tests/tests_utils/test_check_y.py`

- **Update** existing test `test_check_y_exception_when_y_has_missing_values` to pass
  `allow_nan=False` explicitly (documents the default clearly).
- **Add** `test_check_y_no_error_when_y_has_missing_values_and_allow_nan_true`:
  `check_y(pd.Series([0, 1, None]), allow_nan=True)` must not raise.

### 3b. `skforecast/recursive/tests/tests_forecaster_recursive/test_init.py`

- **Add** `test_init_dropna_from_series_attribute_correctly_stored` using
  `@pytest.mark.parametrize` with both `True` and `False`, verifying `forecaster.dropna_from_series == dropna_from_series`.

### 3c. `skforecast/recursive/tests/tests_forecaster_recursive/test_create_train_X_y.py`

Four new tests:

| Test name | Scenario | Expected behaviour |
|-----------|----------|--------------------|
| `test_create_train_X_y_MissingValuesWarning_when_y_has_NaN_in_target_position` | NaN at the last position (ends up in `y_train` only, no lag column affected) | drops affected row, warns `"NaNs detected in \`y_train\`..."`, correct X/y shape |
| `test_create_train_X_y_MissingValuesWarning_and_output_when_NaN_in_X_train_and_dropna_from_series_True` | NaN at an interspersed position that propagates into lags, `dropna_from_series=True` | drops rows with NaN in X_train, warns `"NaNs detected in \`X_train\`. They have been dropped..."` |
| `test_create_train_X_y_MissingValuesWarning_and_output_when_NaN_in_X_train_and_dropna_from_series_False` | same NaN position, `dropna_from_series=False` | NaN preserved in X_train, warns `"NaNs detected in \`X_train\`. Some estimators..."` |
| `test_create_train_X_y_ValueError_when_all_samples_removed_due_to_NaN` | all-NaN series | raises `ValueError("All samples have been removed due to NaNs...")` |

## Files to Modify

| File | Change |
|------|--------|
| `skforecast/utils/utils.py` | `check_y()`: add `allow_nan: bool = False` param |
| `skforecast/recursive/_forecaster_recursive.py` | `__init__`, `_create_train_X_y()` (including `check_exog` relaxation), `set_in_sample_residuals()`, `__skforecast_tags__`, docstrings |
| `skforecast/utils/tests/tests_utils/test_check_y.py` | 1 updated + 1 new test |
| `skforecast/recursive/tests/tests_forecaster_recursive/test_init.py` | 1 new test (parametrized True/False) |
| `skforecast/recursive/tests/tests_forecaster_recursive/test_create_train_X_y.py` | 4 new tests |
| `skforecast/recursive/tests/tests_forecaster_recursive/test_fit.py` | 2 new tests |

---

## Verification Steps

```bash
# 1. Utils tests
pytest skforecast/utils/tests/tests_utils/test_check_y.py -vv

# 2. ForecasterRecursive tests (all modified files)
pytest skforecast/utils/tests/tests_utils/test_check_y.py \
       skforecast/recursive/tests/tests_forecaster_recursive/test_init.py \
       skforecast/recursive/tests/tests_forecaster_recursive/test_create_train_X_y.py \
       skforecast/recursive/tests/tests_forecaster_recursive/test_fit.py \
       -vv

# 3. Full ForecasterRecursive test suite
pytest skforecast/recursive/tests/tests_forecaster_recursive/ -vv
```

### `test_fit.py` additions

Two new tests added to cover the full `fit()` → `predict()` flow:

| Test name | Estimator | `dropna_from_series` | What it checks |
|-----------|-----------|----------------------|----------------|
| `test_fit_with_interspersed_NaN_and_dropna_from_series_True` | `LinearRegression` | `True` | warns on X_train NaN, `is_fitted=True`, NaN-free `last_window_`, `predict(steps=3)` succeeds |
| `test_fit_with_interspersed_NaN_and_dropna_from_series_False` | `HistGradientBoostingRegressor` | `False` | warns on X_train NaN, `is_fitted=True`, NaN-free `last_window_`, `predict(steps=3)` succeeds |

---

## Out of Scope (possible follow-ups)

- **`ForecasterDirect`** — uses the identical `check_y()` call; the same changes apply
  symmetrically but were left for a separate PR.
- **Predict-time NaN in `last_window_`** — already handled with a `MissingValuesWarning`
  via `check_predict_input`; no change needed. If `y` ends with NaN, `last_window_` will
  contain NaN; the user is responsible for supplying a clean `last_window` at predict time.
- **Changelog / release notes** — update `changelog.md` noting that `ForecasterRecursive`
  now accepts interspersed NaN values in `y` (behaviour change from `ValueError` to
  `MissingValuesWarning` + optional row dropping).

---

## Backward Compatibility Note

Previously, calling `forecaster.fit(y=y_with_nan)` raised a `ValueError` immediately.
After this change it issues a `MissingValuesWarning` and proceeds. This is a **breaking
behaviour change** for users who relied on the `ValueError` as validation; it should be
documented in the changelog.
