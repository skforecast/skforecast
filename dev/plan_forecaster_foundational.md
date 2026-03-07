# Plan: ForecasterFoundational + FoundationalModels Rewrite

> **Status**: Design complete — ready for implementation.  
> **Scope**: Chronos-2 only. No Chronos T5 / Chronos-Bolt support.  
> **Branch**: `feature_refactor_arima`

---

## 1. Context and Motivation

The current `skforecast/foundational/_foundational_models.py` has several critical issues that prevent correct usage with Chronos-2:

### Issues in existing `_foundational_models.py`

| Issue | Detail |
|-------|--------|
| Wrong pipeline loader | Uses `ChronosPipeline.from_pretrained()` — silently fails for Chronos-2 checkpoints |
| Fragile API dispatch | `inspect.signature()` used to detect `context` vs `inputs` param names — brittle |
| `supports_exog=False` | Incorrect — Chronos-2 supports `past_covariates` and `future_covariates` |
| Non-standard public API | Uses `data`/`forecast(h=...)`/`h` instead of skforecast-standard `y`/`predict(steps=...)`/`steps` |
| `num_samples` parameter | Irrelevant for Chronos-2 (quantile-native, not sample-based) |
| Version checks | `_chronos_supports_config_field()` / `_get_package_version()` — unnecessary complexity |
| `fit()` returns `None` | Should return `self` |
| No `ForecasterFoundational` class | No integration with backtesting / grid_search / model_selection |

### Goals

1. **Rewrite `_foundational_models.py`**: Remove `ChronosAdapter`, add `Chronos2Adapter` with correct `Chronos2Pipeline` usage.
2. **Create `_forecaster_foundational.py`**: Full skforecast-compatible forecaster class.
3. **Add `backtesting_foundational`**: New function in `model_selection/_validation.py`.
4. **Wire model_selection utilities**: Add `"ForecasterFoundational"` to the right dispatch lists in `_utils.py`.
5. **Update exports**: Both `skforecast.foundational` and `skforecast.model_selection`.

---

## 2. Chronos-2 API Reference

### Loading

```python
from chronos.chronos2.pipeline import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained(
    model_id,           # e.g. "amazon/chronos-2-base"
    device_map=None,    # optional
    torch_dtype=None,   # optional
)
```

> `Chronos2Pipeline.from_pretrained()` is also accessible via `BaseChronosPipeline.from_pretrained()` — auto-dispatches based on config.

### Primary inference: `predict_quantiles()`

```python
quantile_preds, mean_preds = pipeline.predict_quantiles(
    inputs=[input_dict, ...],       # list of dicts (one per series)
    prediction_length=steps,        # int
    quantile_levels=[0.1, ..., 0.9] # list[float]
)
```

**Returns:**
- `quantile_preds`: `list[torch.Tensor]` — each tensor shape `(n_vars, horizon, n_quantiles)`. For univariate: `(1, horizon, n_quantiles)`.
- `mean_preds`: `list[torch.Tensor]` — each tensor shape `(n_vars, horizon)`.

### Input dict format

```python
input_dict = {
    "target": np.ndarray,                          # 1D, shape (T,)
    "past_covariates": {                            # optional
        "col_name": np.ndarray,                     # 1D, shape (T,) each
    },
    "future_covariates": {                          # optional
        "col_name": np.ndarray,                     # 1D, shape (steps,) each
    },
}
```

**Key rules:**
- `past_covariates` must have the same length as `target`.
- `future_covariates` must have exactly `prediction_length` observations.
- Column names in `past_covariates` and `future_covariates` must match when both are provided.
- No `num_samples` — Chronos-2 is quantile-native, not sample-based.

### Point forecast

Use `predict_quantiles()` with `quantile_levels=[0.5]` and take median, or use `pipeline.predict()` which returns samples (less recommended for Chronos-2).

### Other methods (not used in ForecasterFoundational)

- `predict_df()` — pandas-friendly wrapper
- `fit()` — fine-tuning (not used in inference-only forecaster)
- `embed()` — embedding extraction

---

## 3. Key Design Decisions

### `last_window_` attribute

- Stored as `None` (no training data stored in the forecaster object).
- The adapter (`Chronos2Adapter`) stores `_history` internally.
- `window_size` is set to `context_length` so backtesting slices the right window.
- `last_window` is accepted as a parameter in `predict()` for backtesting compatibility — it overrides the adapter's stored history when provided.

### Exog convention

| Method | Parameter | Maps to |
|--------|-----------|---------|
| `fit(y, exog=...)` | historical exog aligned to `y` index | `past_covariates` in Chronos-2 input |
| `predict(steps, exog=...)` | future-known exog for `steps` horizon | `future_covariates` in Chronos-2 input |

- Column names must be consistent between fit-time and predict-time exog.
- If exog provided at fit must also be provided at predict (and vice versa).

### `FoundationalModels` visibility

- **User-facing** (lightweight standalone API): `FoundationalModels("amazon/chronos-2-base").fit(y).predict(steps=12)`.
- **Internal**: `ForecasterFoundational` wraps it for full skforecast ecosystem compatibility.
- Both exported from `skforecast.foundational`.

### Probabilistic output

- `predict_interval(steps, interval=[10, 90])` → DataFrame with `lower_bound`, `pred`, `upper_bound`.
- `predict_quantiles(steps, quantiles=[0.1, 0.5, 0.9])` → DataFrame with `q_0.1`, `q_0.5`, `q_0.9`.
- Both delegate to `Chronos2Pipeline.predict_quantiles()` directly.

---

## 4. Files to Modify / Create

### 4.1 `skforecast/foundational/_foundational_models.py` — **REWRITE**

**Remove entirely:**
- `ChronosAdapter` class
- `_is_chronos_2_model()` function
- `_chronos_supports_config_field()` function
- `_get_package_version()` function
- `inspect` import
- `importlib.metadata` imports
- `BaseAdapter` class (optional — keep if `FoundationalModels` may support other models later; remove if keeping minimal)

**Add:**

```python
class Chronos2Adapter:
    """
    Adapter for Amazon Chronos-2 foundational models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "amazon/chronos-2-base".
    context_length : int, optional
        Maximum number of historical observations to use as context.
        If None, the full history stored at fit time is used.
    pipeline : Chronos2Pipeline, optional
        Pre-loaded pipeline. If None, loaded lazily on first predict call.
    device_map : str, optional
        Device map string passed to `from_pretrained` (e.g. "cuda", "cpu").
    torch_dtype : optional
        Torch dtype passed to `from_pretrained`.
    predict_kwargs : dict, optional
        Additional kwargs forwarded to `predict_quantiles()`.
    """

    capabilities = ModelCapabilities(
        supports_exog=True,
        supports_multivariate=False,
        supports_probabilistic=True,
        context_length=None,    # set from model config after loading
        min_history=1,
    )

    def __init__(self, model_id, *, pipeline=None, context_length=None,
                 predict_kwargs=None, device_map=None, torch_dtype=None):
        self.model_id = model_id
        self._pipeline = pipeline
        self._history: pd.Series | None = None
        self._history_exog: pd.DataFrame | None = None
        self.context_length = context_length
        self.predict_kwargs = predict_kwargs or {}
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._is_fitted = False

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            from chronos.chronos2.pipeline import Chronos2Pipeline
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting >=2.0 is required. "
                "Install with `pip install chronos-forecasting`."
            ) from exc
        kwargs = {}
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        self._pipeline = Chronos2Pipeline.from_pretrained(self.model_id, **kwargs)

    def _build_chronos_input(self, target, past_exog=None, future_exog=None):
        """Build the dict consumed by Chronos2Pipeline.predict_quantiles()."""
        input_dict = {"target": np.asarray(target, dtype=float)}
        if past_exog is not None:
            df = past_exog if isinstance(past_exog, pd.DataFrame) else past_exog.to_frame()
            input_dict["past_covariates"] = {
                col: np.asarray(df[col], dtype=float) for col in df.columns
            }
        if future_exog is not None:
            df = future_exog if isinstance(future_exog, pd.DataFrame) else future_exog.to_frame()
            input_dict["future_covariates"] = {
                col: np.asarray(df[col], dtype=float) for col in df.columns
            }
        return input_dict

    def fit(self, y, exog=None):
        check_y(y, series_id="`y`")
        self._history = y.copy()
        self._history_exog = exog.copy() if exog is not None else None
        self._is_fitted = True
        return self

    def predict(self, steps, exog=None, quantiles=None,
                last_window=None, last_window_exog=None):
        if not self._is_fitted and last_window is None:
            raise ValueError("Call `fit` before `predict`, or pass `last_window`.")
        self._load_pipeline()

        history = last_window if last_window is not None else self._history
        past_exog = last_window_exog if last_window is not None else self._history_exog

        # Trim to context_length
        if self.context_length is not None:
            history = history.iloc[-self.context_length:]
            if past_exog is not None:
                past_exog = past_exog.iloc[-self.context_length:]

        quantile_levels = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        input_dict = self._build_chronos_input(
            target=history.to_numpy(),
            past_exog=past_exog,
            future_exog=exog,
        )

        quantile_preds, mean_preds = self._pipeline.predict_quantiles(
            inputs=[input_dict],
            prediction_length=steps,
            quantile_levels=quantile_levels,
            **self.predict_kwargs,
        )

        # quantile_preds[0] shape: (1, steps, n_q) — squeeze first dim for univariate
        q_arr = quantile_preds[0].squeeze(0)          # (steps, n_q)
        if hasattr(q_arr, "detach"):
            q_arr = q_arr.detach().cpu().numpy()

        forecast_index = expand_index(history.index, steps=steps)

        if quantiles is None:
            # Point forecast: median = quantile at 0.5
            median_idx = quantile_levels.index(0.5)
            return pd.Series(q_arr[:, median_idx], index=forecast_index, name=history.name)

        columns = [f"q_{q}" for q in quantile_levels]
        return pd.DataFrame(q_arr, index=forecast_index, columns=columns)
```

**Update `FoundationalModels`:**

```python
class FoundationalModels:
    """
    Lightweight user-facing interface for foundational time-series models.
    Currently supports Chronos-2 checkpoints only.
    
    Parameters
    ----------
    model : str
        HuggingFace model ID, e.g. "amazon/chronos-2-base".
    **kwargs :
        Forwarded to the underlying adapter (Chronos2Adapter).
    """

    def __init__(self, model: str, **kwargs) -> None:
        self.adapter = Chronos2Adapter(model_id=model, **kwargs)

    @property
    def is_fitted(self) -> bool:
        return self.adapter._is_fitted

    def fit(self, y, exog=None) -> FoundationalModels:
        self.adapter.fit(y=y, exog=exog)
        return self

    def predict(self, steps, exog=None, quantiles=None) -> pd.Series | pd.DataFrame:
        return self.adapter.predict(steps=steps, exog=exog, quantiles=quantiles)
```

**Rename `forecast()` → `predict()`, `data` → `y`, `h` → `steps` everywhere.**

---

### 4.2 `skforecast/foundational/_forecaster_foundational.py` — **NEW FILE**

Full skforecast-compatible forecaster. Structural reference: `ForecasterStats` in `skforecast/recursive/_forecaster_stats.py`.

```python
class ForecasterFoundational:
    """
    Forecaster wrapping a foundational model adapter for full skforecast ecosystem
    compatibility (backtesting, model_selection, etc.).

    Parameters
    ----------
    model : str or FoundationalModels
        HuggingFace model ID string (e.g. "amazon/chronos-2-base") or a
        pre-configured FoundationalModels instance.
    context_length : int, optional
        Number of historical observations to pass as context. Also used as
        `window_size` so backtesting slices the right window.
    **kwargs :
        Forwarded to Chronos2Adapter if `model` is a string.
    """
```

#### Required attributes (follow `ForecasterStats` conventions)

```python
# Set at init
self.model_id: str
self.context_length: int | None
self.adapter: Chronos2Adapter
self.creation_date: str            # datetime.now().strftime("%Y-%m-%d %H:%M:%S")
self.skforecast_version: str       # skforecast.__version__
self.python_version: str           # sys.version.split(" ")[0]
self.forecaster_id: str | None     # user-supplied optional label

# Set after fit
self.is_fitted: bool = False
self.fitted_date: str | None = None
self.training_range_: pd.Index | None = None   # [first_date, last_date]
self.last_window_: None = None                 # intentionally None — never stored
self.index_type_: type | None = None
self.index_freq_: str | pd.offsets.DateOffset | None = None
self.window_size: int                          # = context_length (or sentinel 1)

# skforecast tags
self.__skforecast_tags__ = {
    "library": "skforecast",
    "forecaster_name": "ForecasterFoundational",
    "forecaster_task": "univariate",
    "forecasting_scope": "single-series",
    "forecasting_strategy": "foundational",
    "index_types_supported": ["DatetimeIndex", "RangeIndex"],
    "supports_exog": True,
    "supports_probabilistic": True,
    "prediction_types": ["point", "interval", "quantiles"],
    "model_type": "foundational",
}
```

#### Methods

```python
def fit(self, y: pd.Series, exog: pd.DataFrame | pd.Series | None = None) -> ForecasterFoundational:
    """
    Fit the forecaster. Only stores index metadata; no training — the adapter
    stores the history for context-window purposes.
    """
    # validate y, exog
    # store: training_range_, index_type_, index_freq_, is_fitted, fitted_date
    # call self.adapter.fit(y, exog)
    return self

def predict(
    self,
    steps: int,
    exog: pd.DataFrame | pd.Series | None = None,
    last_window: pd.Series | None = None,
    last_window_exog: pd.DataFrame | pd.Series | None = None,
) -> pd.Series:
    """Point forecast."""

def predict_interval(
    self,
    steps: int,
    interval: list[float] = [10, 90],
    exog: pd.DataFrame | pd.Series | None = None,
    last_window: pd.Series | None = None,
    last_window_exog: pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: pred, lower_bound, upper_bound.
    Uses Chronos-2's native quantile output.
    """

def predict_quantiles(
    self,
    steps: int,
    quantiles: list[float] = [0.1, 0.5, 0.9],
    exog: pd.DataFrame | pd.Series | None = None,
    last_window: pd.Series | None = None,
    last_window_exog: pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame:
    """Returns DataFrame with columns q_0.1, q_0.5, ..."""

def __repr__(self) -> str: ...
def __repr_html__(self) -> str: ...
```

---

### 4.3 `skforecast/foundational/__init__.py` — **UPDATE**

```python
# Before
from ._foundational_models import FoundationalModels

# After
from ._foundational_models import FoundationalModels
from ._forecaster_foundational import ForecasterFoundational
```

---

### 4.4 `skforecast/model_selection/_validation.py` — **ADD**

Add two new functions following the `backtesting_stats` / `_backtesting_stats` pattern.

#### Private `_backtesting_foundational()`

Located near `_backtesting_stats()` (~line 1771).

```python
def _backtesting_foundational(
    forecaster,
    y,
    cv,
    metric,
    exog=None,
    interval=None,
    quantiles=None,
    verbose=False,
    show_progress=True,
):
    """
    Internal backtesting loop for ForecasterFoundational.
    Each fold: re-fits the adapter on the training window, then predicts test window.
    """
```

**Fold loop logic:**

```python
for fold in folds:
    train_idx, test_idx, last_window_idx = fold
    y_train = y.iloc[train_idx[0]:train_idx[1]]
    y_test  = y.iloc[test_idx[0]:test_idx[1]]
    last_window = y.iloc[last_window_idx[0]:last_window_idx[1]]  # context window

    exog_train = exog.iloc[...] if exog is not None else None
    exog_future = exog.iloc[test_idx[0]:test_idx[1]] if exog is not None else None
    last_window_exog = exog.iloc[last_window_idx[0]:last_window_idx[1]] if exog is not None else None

    forecaster.fit(y=y_train, exog=exog_train)
    preds = forecaster.predict(
        steps=len(y_test),
        exog=exog_future,
        last_window=last_window,
        last_window_exog=last_window_exog,
    )
    # collect predictions, compute metric
```

#### Public `backtesting_foundational()`

Located near `backtesting_stats()` (~line 2077).

```python
def backtesting_foundational(
    forecaster,
    y,
    cv,
    metric,
    exog=None,
    interval=None,
    quantiles=None,
    n_jobs="auto",
    verbose=False,
    show_progress=True,
    suppress_warnings=False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting for ForecasterFoundational.

    Parameters
    ----------
    forecaster : ForecasterFoundational
    y : pd.Series with DatetimeIndex and frequency
    cv : TimeSeriesFold
    metric : str or callable or list
    exog : pd.DataFrame or pd.Series, optional
        Full exog covering training + test period.
    interval : list[float], optional
        e.g. [10, 90] for 80% interval.
    quantiles : list[float], optional
        e.g. [0.1, 0.5, 0.9].
    ...
    
    Returns
    -------
    metrics_df : pd.DataFrame
    predictions_df : pd.DataFrame
    """
    if type(forecaster).__name__ != "ForecasterFoundational":
        raise TypeError(...)
    check_backtesting_input(...)
    return _backtesting_foundational(...)
```

---

### 4.5 `skforecast/model_selection/_utils.py` — **UPDATE**

Three locations to add `"ForecasterFoundational"`:

1. **`forecasters_uni` list** (~line 196):

```python
forecasters_uni = [
    "ForecasterRecursive",
    "ForecasterDirect",
    "ForecasterStats",
    "ForecasterFoundational",    # ADD
    ...
]
```

2. **`initial_train_size` required block** (~line 345):

```python
if type(forecaster).__name__ in [
    "ForecasterStats",
    "ForecasterFoundational",   # ADD
] and initial_train_size is None:
    raise ValueError(...)
```

3. **`select_n_jobs_backtesting()` early return** (~line 682):

```python
if type(forecaster).__name__ in ["ForecasterStats", "ForecasterFoundational"]:
    return 1   # Foundational models handle their own parallelism internally
```

---

### 4.6 `skforecast/model_selection/__init__.py` — **UPDATE**

```python
# Add
from ._validation import backtesting_foundational
```

---

## 5. Phase-by-Phase Implementation Plan

### Phase 1 — Rewrite `_foundational_models.py`

**Tasks:**
- [ ] Remove: `ChronosAdapter`, `BaseAdapter`, `_is_chronos_2_model`, `_chronos_supports_config_field`, `_get_package_version`, `inspect` import, `importlib.metadata` imports.
- [ ] Add: `Chronos2Adapter` with `capabilities`, `_load_pipeline()`, `_build_chronos_input()`, `fit()`, `predict()`.
- [ ] Update `FoundationalModels`: rename `forecast()` → `predict()`, `data` → `y`, `h` → `steps`; `fit()` returns `self`.
- [ ] Keep `ModelCapabilities` dataclass (used by `Chronos2Adapter`).
- [ ] Ensure `FoundationalModels.predict()` passes `last_window` and `last_window_exog` through to adapter.

**Validation:**
```python
from skforecast.foundational import FoundationalModels
import pandas as pd

m = FoundationalModels("amazon/chronos-2-base")
y = pd.Series(range(50), index=pd.date_range("2020", periods=50, freq="ME"))
m.fit(y)
pred = m.predict(steps=12)
assert len(pred) == 12
```

---

### Phase 2 — Create `_forecaster_foundational.py`

**Tasks:**
- [ ] Create file with `ForecasterFoundational` class.
- [ ] `__init__`: all null attrs + `__skforecast_tags__`.
- [ ] `fit()`: validate + store metadata only + call `adapter.fit()`.
- [ ] `predict()`: delegate to `adapter.predict()`.
- [ ] `predict_interval()`: delegate; format output columns.
- [ ] `predict_quantiles()`: delegate.
- [ ] `__repr__` / `__repr_html__`: follow `ForecasterStats` pattern.
- [ ] `set_params()`, `get_params()` for compatibility.

---

### Phase 3 — `backtesting_foundational` in `_validation.py`

**Tasks:**
- [ ] Implement `_backtesting_foundational()` with fold loop.
- [ ] Implement `backtesting_foundational()` public function.
- [ ] Handle `interval` and `quantiles` arguments.
- [ ] Support `metric` as str / callable / list (same as `backtesting_stats`).

---

### Phase 4 — Wire `model_selection/_utils.py`

**Tasks:**
- [ ] Add `"ForecasterFoundational"` to `forecasters_uni`.
- [ ] Add to `initial_train_size` required check.
- [ ] Add to `select_n_jobs_backtesting` early-return block.

---

### Phase 5 — Exports and Tests

**Tasks:**
- [ ] Update `skforecast/foundational/__init__.py`.
- [ ] Update `skforecast/model_selection/__init__.py`.
- [ ] Create `skforecast/foundational/tests/` directory with:
  - `test_foundational_models.py`: mock `Chronos2Pipeline`, test `fit/predict/predict_quantiles`.
  - `test_forecaster_foundational.py`: mock adapter, test backtesting compatibility.

---

## 6. Reference: `ForecasterStats` Attribute Conventions

Source: `skforecast/recursive/_forecaster_stats.py`, lines 240–342.

```python
# Set at __init__
self.estimator         = estimator   # the stats model
self.creation_date     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
self.skforecast_version = skforecast.__version__
self.python_version    = sys.version.split(" ")[0]
self.forecaster_id     = forecaster_id

# Initially None, set after fit
self.is_fitted         = False
self.fitted_date       = None
self.training_range_   = None
self.last_window_      = None
self.index_type_       = None
self.index_freq_       = None
self.window_size       = None   # set from model after fit
```

`__skforecast_tags__` example (from `ForecasterStats`):

```python
self.__skforecast_tags__ = {
    "library": "skforecast",
    "forecaster_name": "ForecasterStats",
    "forecaster_task": ...,
    "forecasting_scope": ...,
    "forecasting_strategy": ...,
    "index_types_supported": [...],
    "supports_exog": ...,
    "supports_probabilistic": ...,
    "prediction_types": [...],
}
```

---

## 7. Reference: `backtesting_stats` Pattern

Source: `skforecast/model_selection/_validation.py`, lines 2077–~2200.

```python
def backtesting_stats(forecaster, y, cv, metric, exog=None, ...):
    if type(forecaster).__name__ != "ForecasterStats":
        raise TypeError(
            f"`forecaster` must be an instance of `ForecasterStats`. "
            f"Got {type(forecaster).__name__}."
        )
    check_backtesting_input(
        forecaster=forecaster,
        cv=cv,
        y=y,
        exog=exog,
        ...
    )
    return _backtesting_stats(
        forecaster=forecaster,
        y=y,
        cv=cv,
        metric=metric,
        exog=exog,
        ...
    )
```

---

## 8. Import Map

```
skforecast.foundational
  ├── FoundationalModels     ← _foundational_models.py (REWRITE)
  └── ForecasterFoundational ← _forecaster_foundational.py (NEW)

skforecast.model_selection
  ├── backtesting_foundational ← _validation.py (ADD)
  └── [other existing exports]
```

---

## 9. Out of Scope

- Chronos T5 / Chronos-Bolt support (removed by user request)
- Fine-tuning / `pipeline.fit()` support
- Multi-variate series support
- `grid_search_foundational` / `bayesian_search_foundational` (future work)
- `ForecasterFoundational` integration with `grid_search_forecaster` (not applicable — no hyperparameters to tune in standard usage)
