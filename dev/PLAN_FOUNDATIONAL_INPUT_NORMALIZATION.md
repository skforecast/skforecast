# Plan: Unified Input Normalization in the Foundational Module

## Problem Statement

The foundational module has three layers: `ForecasterFoundational`, `FoundationalModel`, and the adapters (`Chronos2Adapter`, `TimesFM25Adapter`, `MoiraiAdapter`). Input validation and preprocessing are currently scattered across all of them, creating four concrete problems:

1. **Redundancy**: `check_y` fires twice in single-series `fit` (once in `ForecasterFoundational`, once in every adapter). The `is_fitted` guard exists in both `ForecasterFoundational` and every adapter with different exception types.

2. **Adapter bloat**: Every adapter replicates the same `isinstance` dispatch tree to handle `pd.Series`, wide `pd.DataFrame`, and `dict[str, pd.Series]` inputs. This is backend-agnostic boilerplate that has nothing to do with running Chronos, TimesFM, or Moirai inference.

3. **`FoundationalModel` gap**: Users who call `FoundationalModel` directly (bypassing `ForecasterFoundational`) receive zero input validation, normalization, or fitted-state tracking.

4. **Single/multi-series asymmetry in `ForecasterFoundational`**: `check_y` and `check_exog` are called for single-series but not for multi-series, leading to inconsistent behaviour.

---

## Proposed Architecture

### Core decision: `FoundationalModel` is the state machine

`FoundationalModel` becomes the layer where **all** preprocessing, validation, and fitted-state management happens. It is a fully self-contained forecasting engine with a clean public API. `ForecasterFoundational` becomes a thin skforecast-ecosystem adapter over it.

### Ownership by layer after the refactor

| Responsibility | Layer |
|---|---|
| Normalize `series` / `exog` / `last_window` / `last_window_exog` to `dict[str, pd.Series]` | `FoundationalModel` |
| Content validation: `check_y` per series, `steps > 0`, `quantiles` in 0–1 | `FoundationalModel` |
| `allow_exogenous` enforcement (warn + drop exog if adapter doesn't support it) | `FoundationalModel` |
| Fit-time exog alignment (`align_exog_to_series`) and column-name extraction | `FoundationalModel` |
| Fitted-state attributes: `is_fitted`, `series_names_in_`, `_is_multiseries`, `index_type_`, `index_freq_`, `training_range_`, `exog_in_`, `exog_names_in_per_series_` | `FoundationalModel` |
| Predict-time exog consistency (`validate_exog_predict`, `validate_last_window_exog`) | `FoundationalModel` |
| `NotFittedError` guard (sklearn/skforecast contract) | `ForecasterFoundational` |
| `predict_interval` interval bounds validation | `ForecasterFoundational` |
| `__repr__` / `_repr_html_` / `summary` / skforecast metadata (`fit_date`, `skforecast_version`, `forecaster_id`) | `ForecasterFoundational` |
| Context trimming to `context_length` | Adapters |
| Model-specific quantile constraints (fixed sets for TimesFM, Moirai) | Adapters |
| Lazy backend loading | Adapters |

### The key invariants

1. **Adapters always receive `dict[str, pd.Series]`** for `series` and `dict[str, pd.DataFrame | pd.Series | None]` for `exog`. No adapter ever needs to inspect the type of its inputs again.

2. **`ForecasterFoundational` reads all fitted state from `self.estimator`** — it never maintains its own parallel copies of `series_names_in_`, `index_freq_`, etc.

3. **`FoundationalModel` is fully usable standalone** — calling it directly gives the same guarantees as going through `ForecasterFoundational`.

### Single-series mode flag

When a bare `pd.Series` is the original input, `FoundationalModel` records `_is_single_series_mode = True`. Adapters use this flag **only** to shape the output (return `pd.Series` instead of `pd.DataFrame`). This is the only piece of presentation metadata they need.

---

## Detailed Changes

### 1. `_utils.py` — normalization helpers

Add two normalization functions (replacing the existing `check_preprocess_series_type` / `check_preprocess_exog_type`):

```python
def normalize_series_to_dict(
    series: pd.Series | pd.DataFrame | dict[str, pd.Series],
) -> tuple[dict[str, pd.Series], bool]:
    """
    Normalize any supported series format to dict[str, pd.Series].

    Returns
    -------
    series_dict : dict[str, pd.Series]
    is_single_series_mode : bool
        True when the original input was a pd.Series (single-series mode).
    """
```

```python
def normalize_exog_to_dict(
    exog: pd.Series | pd.DataFrame | dict | None,
    series_names: list[str],
) -> dict[str, pd.DataFrame | pd.Series | None]:
    """
    Normalize any supported exog format to dict[str, pd.DataFrame | pd.Series | None].
    Broadcasts a single pd.Series / pd.DataFrame to all series names.
    """
```

Both functions must be **idempotent**: a dict input is returned immediately without copying. Long-format DataFrames are converted; unsupported types raise `TypeError`.

The existing helpers `align_exog_to_series`, `validate_exog_fit`, `validate_exog_predict`, `validate_last_window_exog`, `assert_aligned`, and `align_exog_single` remain in `_utils.py` and are called from `FoundationalModel` rather than `ForecasterFoundational`.

---

### 2. `FoundationalModel` — full state machine

`FoundationalModel` adds all fitted-state attributes to its `__init__`:

```python
self._is_single_series_mode  = False
self._is_multiseries         = False
self.series_names_in_        = None
self.index_type_             = None
self.index_freq_             = None
self.training_range_         = None
self.exog_in_                = False
self.exog_names_in_          = None
self.exog_names_in_per_series_ = None
```

**`fit(series, exog)`**

```
1. normalize_series_to_dict(series) → (series_dict, _is_single_series_mode)
2. Store _is_single_series_mode, _is_multiseries, series_names_in_
3. check_y(s) for each s in series_dict.values()
4. If exog is not None and not self.adapter.allow_exogenous:
       issue IgnoredArgumentWarning; set exog = None
5. normalize_exog_to_dict(exog, series_names) → exog_dict
6. align_exog_to_series(series_dict, exog_dict)    # reindex to series index
7. validate_exog_fit(series_dict, exog_dict)        # returns exog_names_in_per_series_
8. Store index_type_, index_freq_, training_range_, exog_in_, exog_names_in_per_series_
9. self.adapter.fit(series_dict, exog_dict)
```

**`predict(steps, exog, quantiles, last_window, last_window_exog)`**

```
1. Validate steps > 0 (raise ValueError)
2. If quantiles is not None: validate all values in [0, 1] (raise ValueError)
3. normalize_exog_to_dict(exog) if exog is not None
4. normalize_exog_to_dict(last_window_exog) if last_window_exog is not None
5. normalize_series_to_dict(last_window) if last_window is not None
6. validate_exog_predict(exog, steps, ...)          # column names, horizon alignment
7. validate_last_window_exog(last_window_exog, last_window, exog_in_)
8. self.adapter.predict(steps, exog_dict, quantiles, last_window_dict, last_window_exog_dict)
```

---

### 3. Adapters — pure inference code

Each adapter's `fit` and `predict` are stripped to backend-only logic:

**Removed from every adapter:**
- All `isinstance(series, ...)` / `isinstance(last_window, ...)` dispatch
- All `check_y(s)` calls
- Empty-dict checks
- Long-format DataFrame rejection
- `allow_exogenous` / `IgnoredArgumentWarning` logic
- `_is_fitted` user-visible guards in `predict` (replaced by `assert self._is_fitted`)

**Retained in every adapter:**
- `context_length` trimming of history and `last_window`
- Lazy model/pipeline loading
- Model-specific quantile constraint validation (e.g. fixed sets for TimesFM, Moirai)
- The `_is_single_series_mode` read for output shaping

Each adapter's `fit` signature becomes:

```python
def fit(self, series: dict[str, pd.Series], exog: dict[str, ...] | None) -> Self:
```

And `predict`:

```python
def predict(
    self,
    steps: int,
    exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    quantiles: list[float] | None,
    last_window: dict[str, pd.Series] | None,
    last_window_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
) -> pd.Series | pd.DataFrame:
```

---

### 4. `ForecasterFoundational` — thin skforecast adapter

`ForecasterFoundational` becomes a delegator. Its only responsibilities are:

**`__init__`:**
- Validate that `estimator` is a `FoundationalModel` instance.
- Store `forecaster_id`, `creation_date`, `skforecast_version`, `python_version`.
- Expose `context_length`, `model_id`, `window_size` as convenient aliases pointing to `self.estimator`.

**`fit(series, exog)`:**
- Delegate entirely to `self.estimator.fit(series, exog)`.
- Record `fit_date`.
- No metadata extraction — read it back from `self.estimator.*` as needed.

**`predict` / `predict_interval` / `predict_quantiles`:**
- Raise `NotFittedError` if `not self.estimator.is_fitted` (sklearn contract).
- `predict_interval` / `predict_quantiles`: validate `interval` or `quantiles` arguments (these method-level API concerns belong here).
- Delegate to `self.estimator.predict(...)`.

**`__repr__` / `_repr_html_`:**
- Read all state from `self.estimator.*` properties (no local copies needed).

**Properties (delegating to `self.estimator`):**

```python
@property
def series_names_in_(self):    return self.estimator.series_names_in_
@property
def index_freq_(self):         return self.estimator.index_freq_
@property
def training_range_(self):     return self.estimator.training_range_
@property
def exog_in_(self):            return self.estimator.exog_in_
@property
def exog_names_in_(self):      return self.estimator.exog_names_in_
@property
def is_fitted(self):           return self.estimator.is_fitted
# etc.
```

---

## Migration Checklist

### `_utils.py`
- [ ] Add `normalize_series_to_dict`
- [ ] Add `normalize_exog_to_dict`
- [ ] Keep existing alignment/validation helpers unchanged

### `FoundationalModel`
- [ ] Add all fitted-state attributes to `__init__`
- [ ] Add `_is_single_series_mode` attribute
- [ ] Implement full `fit` logic (normalization → validation → alignment → adapter delegation)
- [ ] Implement full `predict` logic (normalization → validation → adapter delegation)
- [ ] Move `allow_exogenous` enforcement into `fit`
- [ ] Move `align_exog_to_series` + `validate_exog_fit` calls into `fit`
- [ ] Move `validate_exog_predict` + `validate_last_window_exog` calls into `predict`

### Adapters (`Chronos2Adapter`, `TimesFM25Adapter`, `MoiraiAdapter`)
- [ ] Update `fit` signature to accept `dict[str, pd.Series]` only
- [ ] Update `predict` signature to accept unified dict inputs
- [ ] Remove all `isinstance` input dispatch
- [ ] Remove all `check_y` calls
- [ ] Remove `allow_exogenous` / `IgnoredArgumentWarning` logic
- [ ] Replace `_is_fitted` user-visible guard in `predict` with `assert`

### `ForecasterFoundational`
- [ ] Strip all metadata extraction from `fit` (delegate to `self.estimator.fit`)
- [ ] Replace all local attribute reads with delegating properties
- [ ] Remove `check_y`, `check_exog`, `align_exog_to_series`, `validate_exog_fit` calls
- [ ] Remove `validate_exog_predict`, `validate_last_window_exog` calls from `predict*`
- [ ] Keep `NotFittedError` guard and `predict_interval` bounds validation
- [ ] Keep `__repr__` / `_repr_html_` / `summary` (reading state via properties)

### Tests
- [ ] Add unit tests for `FoundationalModel` called directly (currently untested path)
- [ ] Verify adapters raise `AssertionError` (not `ValueError`) on unfitted `predict`
- [ ] Verify `ForecasterFoundational` state mirrors `FoundationalModel` state after fit

---

## Non-Goals

- This refactor does **not** change the public API of `ForecasterFoundational`, `FoundationalModel`, or any adapter.
- The skforecast-presentation layer (`__repr__`, `summary`, `forecaster_id`, `fit_date`) stays exclusively in `ForecasterFoundational`.
- `predict_interval` bounds validation (`check_interval`, `[lower, upper]` checks) stays in `ForecasterFoundational` — it is tied to that method's existence, not to data state.
