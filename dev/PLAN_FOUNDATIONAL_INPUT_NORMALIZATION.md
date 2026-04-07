# Plan: Unified Input Normalization in the Foundational Module

## Problem Statement

The foundational module has three layers: `ForecasterFoundational`, `FoundationalModel`, and the adapters (`Chronos2Adapter`, `TimesFM25Adapter`, `MoiraiAdapter`). Input validation and preprocessing are currently scattered across all of them, creating two concrete problems:

1. **Redundancy**: `check_y` fires twice in single-series `fit` (once in `ForecasterFoundational`, once in every adapter). The `is_fitted` guard exists in both `ForecasterFoundational` and every adapter with different exception types.

2. **Adapter bloat**: Every adapter replicates the same `isinstance` dispatch tree to handle `pd.Series`, wide `pd.DataFrame`, and `dict[str, pd.Series]` inputs. This is backend-agnostic boilerplate that has nothing to do with running Chronos, TimesFM, or Moirai inference.

3. **`FoundationalModel` gap**: Users who call `FoundationalModel` directly (bypassing `ForecasterFoundational`) receive zero input validation or normalization.

4. **Single/multi-series asymmetry in `ForecasterFoundational`**: `check_y` and `check_exog` are called for single-series but not for multi-series, leading to inconsistent behaviour.

---

## Proposed Architecture

### Ownership by layer after the refactor

| Responsibility | Layer |
|---|---|
| Normalize `series` / `exog` / `last_window` / `last_window_exog` to `dict[str, pd.Series]` | `FoundationalModel` |
| Content validation: `check_y` per series, `steps > 0`, `quantiles` in 0–1 | `FoundationalModel` |
| Fitted-state consistency: `is_fitted`, `exog_names_in_`, `index_freq_`, column match | `ForecasterFoundational` |
| Skforecast metadata extraction: `_is_multiseries`, `series_names_in_`, `training_range_`, etc. | `ForecasterFoundational` |
| `allow_exogenous` enforcement | `ForecasterFoundational` |
| `predict_interval` interval validation | `ForecasterFoundational` |
| Context trimming to `context_length` | Adapters |
| Model-specific quantile constraints (fixed sets for TimesFM, Moirai) | Adapters |
| Lazy backend loading | Adapters |

### The key invariant

**Adapters always receive `dict[str, pd.Series]`** for `series` and `dict[str, pd.DataFrame | pd.Series | None]` for `exog`. No adapter ever needs to inspect the type of its inputs again.

### Single-series mode flag

When a bare `pd.Series` is the original input, `FoundationalModel` records `_is_single_series_mode = True`. Adapters use this flag **only** to shape the output (return `pd.Series` instead of `pd.DataFrame`). This is the only piece of presentation metadata they need.

---

## Detailed Changes

### 1. `_utils.py` — extend normalization helpers

Add (or extend) a single normalization function that returns a canonical dict:

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

Add a parallel function for exog:

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

Both functions must be **idempotent**: a dict input is returned immediately without copying.

`check_preprocess_series_type` and `check_preprocess_exog_type` can be refactored to delegate to these or simply replaced by them.

---

### 2. `FoundationalModel` — add normalization + basic validation

**`fit(series, exog)`**

```
Before delegating to adapter:
1. Call normalize_series_to_dict(series) → (series_dict, _is_single_series_mode)
2. Store self._is_single_series_mode
3. Call check_y(s) for each s in series_dict.values()          # content validation
4. Call normalize_exog_to_dict(exog, series_names)              # if exog is not None
5. Forward (series_dict, exog_dict) to self.adapter.fit(...)
```

**`predict(steps, exog, quantiles, last_window, last_window_exog)`**

```
Before delegating to adapter:
1. Validate steps > 0 (raise ValueError)
2. Validate quantiles in [0, 1] range (raise ValueError)
3. Normalize last_window → dict (if provided)
4. Normalize exog, last_window_exog → dict (if provided)
5. Forward normalized inputs to self.adapter.predict(...)
```

---

### 3. Adapters — remove all input dispatch boilerplate

Each adapter's `fit` and `predict` can be simplified by removing:

- All `isinstance(series, pd.Series)` / `isinstance(series, pd.DataFrame)` dispatch
- All `isinstance(last_window, ...)` dispatch
- All `check_y(series)` or `check_y(s)` calls
- The empty-dict check (now caught upstream by `normalize_series_to_dict`)
- Long-format DataFrame rejection (now caught upstream)

Each adapter now does approximately:

```python
def fit(self, series: dict[str, pd.Series], exog: dict[str, ...] | None) -> Self:
    # series is always dict[str, pd.Series] at this point
    self._history = {}
    for name, s in series.items():
        self._history[name] = s.values[-self.context_length:]
    if exog is not None:
        # store trimmed exog (Chronos2 only)
        ...
    self._is_fitted = True
    self._series_names = list(series.keys())
    return self

def predict(self, steps, exog, quantiles, last_window, last_window_exog) -> ...:
    # steps already validated, last_window already a dict
    # model-specific quantile set check only
    ...
    # run inference
```

The `_is_single_series_mode` attribute (set by `FoundationalModel`) tells the adapter how to shape the return value.

---

### 4. `ForecasterFoundational` — keep only fitted-metadata checks

**`fit`:** 
- Call `check_preprocess_series_type` (or the new `normalize_series_to_dict`) for its own metadata extraction (`_is_multiseries`, `series_names_in_`, `index_freq_`, etc.).
- Remove the redundant single-series `check_y` call (now done inside `FoundationalModel`).
- Remove the `check_exog` call (now done inside `FoundationalModel`).
- Keep `align_exog_to_series` + `validate_exog_fit` (these require fitted metadata).
- Delegate to `self.estimator.fit(normalized_series, normalized_exog)` — passing already-normalized dicts (idempotent in `FoundationalModel`).

**`predict` / `predict_interval` / `predict_quantiles`:**
- Keep `is_fitted` guard (`NotFittedError`).
- Keep `validate_exog_predict` (requires `exog_names_in_per_series_`, `index_freq_`, etc.).
- Keep `validate_last_window_exog` (requires `exog_in_`).
- Keep `check_interval` / interval bounds validation.
- Remove `steps` and `quantiles` basic range checks — now done in `FoundationalModel`.
- Remove adapter-level `is_fitted` guards (make their `_is_fitted` check an assertion, not a user-visible guard).

---

## Migration Checklist

- [ ] Add `normalize_series_to_dict` to `_utils.py`
- [ ] Add `normalize_exog_to_dict` to `_utils.py`
- [ ] Add `_is_single_series_mode` attribute to `FoundationalModel.__init__`
- [ ] Add normalization + `check_y` calls to `FoundationalModel.fit`
- [ ] Add `steps`, `quantiles`, `last_window`, `exog` normalization to `FoundationalModel.predict`
- [ ] Strip input dispatch from `Chronos2Adapter.fit` and `.predict`
- [ ] Strip input dispatch from `TimesFM25Adapter.fit` and `.predict`
- [ ] Strip input dispatch from `MoiraiAdapter.fit` and `.predict`
- [ ] Remove redundant `check_y` / `check_exog` from `ForecasterFoundational.fit` (single-series path)
- [ ] Ensure `ForecasterFoundational` passes normalized dicts to `self.estimator` (no double normalization cost)
- [ ] Update / add unit tests for `FoundationalModel` called directly (currently untested)
- [ ] Update / add unit tests for each adapter to confirm they reject non-dict inputs with a clear error (or simply via assertion)

---

## Non-Goals

- This refactor does **not** change the public API of any class.
- This refactor does **not** affect `ForecasterFoundational`'s fitted-metadata logic or `_utils.py` alignment helpers (`align_exog_to_series`, `validate_exog_predict`, etc.).
- `FoundationalModel` does **not** take on `index_freq_` or `training_range_` tracking — that remains exclusively in `ForecasterFoundational`.
