# Plan: ForecasterFoundational + FoundationalModel Rewrite

> **Status**: Design complete — ready for implementation.  
> **Scope**: Chronos-2 only. No Chronos T5 / Chronos-Bolt support. Multi-series (batch) inference supported.  
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
| `fit(series, exog=...)` | historical exog aligned to `series` index | `past_covariates` in Chronos-2 input |
| `predict(steps, exog=...)` | future-known exog for `steps` horizon | `future_covariates` in Chronos-2 input |

- Column names must be consistent between fit-time and predict-time exog.
- If exog provided at fit must also be provided at predict (and vice versa).

### `FoundationalModel` visibility

- **User-facing** (lightweight standalone API): `FoundationalModel("amazon/chronos-2-base").fit(y).predict(steps=12)`.
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
- `BaseAdapter` class (optional — keep if `FoundationalModel` may support other models later; remove if keeping minimal)

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
    pipeline : BaseChronosPipeline, optional
        Pre-loaded pipeline. If None, loaded lazily on first predict call.
    device_map : str, optional
        Device map string passed to `from_pretrained` (e.g. "cuda", "cpu").
    torch_dtype : optional
        Torch dtype passed to `from_pretrained`.
    predict_kwargs : dict, optional
        Additional kwargs forwarded to `predict_quantiles()`.
    cross_learning : bool, default False
        If True, share information across all series in the batch when
        predicting in multi-series mode. Ignored in single-series mode.
    """

    def __init__(self, model_id, *, pipeline=None, context_length=None,
                 predict_kwargs=None, device_map=None, torch_dtype=None,
                 cross_learning=False):
        self.model_id = model_id
        self._pipeline = pipeline
        self._history: pd.Series | dict[str, pd.Series] | None = None
        self._history_exog: (
            pd.DataFrame | pd.Series
            | dict[str, pd.DataFrame | pd.Series | None] | None
        ) = None
        self.context_length = context_length
        self.predict_kwargs = predict_kwargs or {}
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.cross_learning = cross_learning
        self._is_fitted = False
        self._is_multiseries = False  # set True in fit() for DataFrame/dict input

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            from chronos import BaseChronosPipeline  # auto-dispatches to Chronos2Pipeline
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
        self._pipeline = BaseChronosPipeline.from_pretrained(self.model_id, **kwargs)

    @staticmethod
    def _to_covariate_array(col_data):
        """
        Cast numeric/bool columns to float64; leave string/categorical as-is.
        Handles pandas nullable extension dtypes correctly (avoids dtype=object
        with pd.NA sentinels that np.asarray() would produce).
        """
        if isinstance(col_data, pd.Series):
            if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
                return col_data.astype(np.float64).to_numpy()
            return col_data.to_numpy()
        arr = np.asarray(col_data)
        if (np.issubdtype(arr.dtype, np.integer)
                or np.issubdtype(arr.dtype, np.floating)
                or arr.dtype.kind == "b"):
            return arr.astype(np.float64)
        return arr

    @staticmethod
    def _normalize_exog_to_dict(exog, series_names):
        """
        Normalise exog to a per-series dict keyed by series name.

        - None  → all series mapped to None.
        - pd.Series / pd.DataFrame → broadcast identically to every series.
        - dict  → values kept per-series; missing keys mapped to None.
        """
        if exog is None:
            return {name: None for name in series_names}
        if isinstance(exog, dict):
            return {name: exog.get(name, None) for name in series_names}
        return {name: exog for name in series_names}  # broadcast to all series

    def _build_chronos_input(self, target, past_exog=None, future_exog=None):
        """
        Build the input dict consumed by the pipeline's predict_quantiles().
        Numeric/bool covariates are cast to float64 via _to_covariate_array;
        string/categorical covariates are passed as-is (Chronos-2 handles them natively).
        """
        input_dict = {"target": np.asarray(target, dtype=float)}
        if past_exog is not None:
            df = past_exog if isinstance(past_exog, pd.DataFrame) else past_exog.to_frame()
            input_dict["past_covariates"] = {
                col: Chronos2Adapter._to_covariate_array(df[col]) for col in df.columns
            }
        if future_exog is not None:
            df = future_exog if isinstance(future_exog, pd.DataFrame) else future_exog.to_frame()
            input_dict["future_covariates"] = {
                col: Chronos2Adapter._to_covariate_array(df[col]) for col in df.columns
            }
        return input_dict

    def fit(self, series, exog=None):
        """
        Store history. No model training — Chronos-2 is zero-shot.

        Supports:
        - pd.Series              → single-series mode
        - pd.DataFrame (wide)    → multi-series mode (each column = one series)
        - dict[str, pd.Series]   → multi-series mode (keys = series names)
        """
        if isinstance(series, pd.Series):
            check_y(series, series_id="`series`")
            self._is_multiseries = False
            if self.context_length is not None:
                self._history = series.iloc[-self.context_length:].copy()
                self._history_exog = (
                    exog.iloc[-self.context_length:].copy() if exog is not None else None
                )
            else:
                self._history = series.copy()
                self._history_exog = exog.copy() if exog is not None else None
        elif isinstance(series, (pd.DataFrame, dict)):
            series_dict = (
                {col: series[col].copy() for col in series.columns}
                if isinstance(series, pd.DataFrame)
                else {k: v.copy() for k, v in series.items()}
            )
            if not series_dict:
                raise ValueError("`series` must contain at least one series.")
            for name, s in series_dict.items():
                check_y(s, series_id=f"'{name}'")
                series_dict[name].name = name
            series_names = list(series_dict.keys())
            exog_dict = self._normalize_exog_to_dict(exog, series_names)
            if self.context_length is not None:
                self._history = {
                    name: s.iloc[-self.context_length:].copy()
                    for name, s in series_dict.items()
                }
                self._history_exog = {
                    name: (e.iloc[-self.context_length:].copy() if e is not None else None)
                    for name, e in exog_dict.items()
                }
            else:
                self._history = series_dict
                self._history_exog = exog_dict
            self._is_multiseries = True
        else:
            raise TypeError(
                "`series` must be a pd.Series, wide pd.DataFrame, or dict[str, pd.Series]. "
                f"Got {type(series)}."
            )
        self._is_fitted = True
        return self

    def _predict_multiseries(self, steps, exog, quantile_levels, quantiles,
                             last_window, last_window_exog):
        """
        Internal multi-series prediction: a single batched pipeline call for all series.

        Returns a long-format DataFrame:
        - Point forecast: columns ["level", "pred"]
        - Quantile forecast: columns ["level", "q_0.1", "q_0.5", …]
        The index repeats each forecast timestamp once per series
        (n_steps × n_series rows total); "level" identifies the series.
        """
        # … builds per-series input list, calls predict_quantiles once,
        # decodes quantile tensors, assembles long-format DataFrame …

    def predict(self, steps, exog=None, quantiles=None,
                last_window=None, last_window_exog=None):
        if not self._is_fitted and last_window is None:
            raise ValueError("Call `fit` before `predict`, or pass `last_window`.")
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("`steps` must be a positive integer.")
        quantile_levels = list(quantiles) if quantiles is not None else [0.1, 0.5, 0.9]
        self._load_pipeline()

        # Dispatch to multi-series path when history or last_window is multi-series
        if self._is_multiseries or isinstance(last_window, (pd.DataFrame, dict)):
            return self._predict_multiseries(
                steps, exog, quantile_levels, quantiles, last_window, last_window_exog
            )

        # --- single-series path ---
        history = last_window if last_window is not None else self._history
        past_exog = last_window_exog if last_window is not None else self._history_exog
        # Trim to context_length only when last_window is provided
        # (_history was already trimmed at fit time)
        if last_window is not None and self.context_length is not None:
            history = history.iloc[-self.context_length:]
            if past_exog is not None:
                past_exog = past_exog.iloc[-self.context_length:]

        input_dict = self._build_chronos_input(
            target=history.to_numpy(), past_exog=past_exog, future_exog=exog
        )
        quantile_preds, _ = self._pipeline.predict_quantiles(
            inputs=[input_dict],
            prediction_length=steps,
            quantile_levels=quantile_levels,
            **self.predict_kwargs,
        )
        # quantile_preds[0] shape: (n_vars, steps, n_q); n_vars==1 for univariate
        q_arr = quantile_preds[0].squeeze(0)  # (steps, n_q)
        if hasattr(q_arr, "detach"):
            q_arr = q_arr.detach().cpu().numpy()

        forecast_index = expand_index(history.index, steps=steps)
        if quantiles is None:
            median_idx = quantile_levels.index(0.5)
            return pd.Series(q_arr[:, median_idx], index=forecast_index, name=history.name)
        columns = [f"q_{q}" for q in quantile_levels]
        return pd.DataFrame(q_arr, index=forecast_index, columns=columns)
```

**`FoundationalModel` class** (already implemented — no changes needed):

```python
class FoundationalModel:
    """
    Lightweight user-facing interface for foundational time-series models.
    Currently supports Chronos-2 checkpoints only.

    Parameters
    ----------
    model : str
        HuggingFace model ID, e.g. "autogluon/chronos-2-small".
    cross_learning : bool, default False
        If True, share information across all series in the batch when
        predicting in multi-series mode. Ignored in single-series mode.
    **kwargs :
        Forwarded to the underlying adapter (Chronos2Adapter):
        context_length, pipeline, device_map, torch_dtype, predict_kwargs.
    """

    def __init__(self, model: str, *, cross_learning: bool = False, **kwargs) -> None:
        self.adapter = Chronos2Adapter(model_id=model, cross_learning=cross_learning, **kwargs)

    @property
    def is_fitted(self) -> bool:
        return self.adapter._is_fitted

    def fit(self, series, exog=None) -> FoundationalModel:
        self.adapter.fit(series=series, exog=exog)
        return self

    def predict(self, steps, exog=None, quantiles=None,
                last_window=None, last_window_exog=None) -> pd.Series | pd.DataFrame:
        return self.adapter.predict(
            steps=steps, exog=exog, quantiles=quantiles,
            last_window=last_window, last_window_exog=last_window_exog,
        )
```

---

### 4.2 `skforecast/foundational/_forecaster_foundational.py` — **NEW FILE**

Full skforecast-compatible forecaster. Primary structural reference: `ForecasterStats` in `skforecast/recursive/_forecaster_stats.py`.

**Review against `ForecasterRecursive`:** Only one thing from `ForecasterRecursive` is relevant and already included in this plan — `predict_quantiles` (Chronos-2 is quantile-native). Everything else (`predict_bootstrapping`, `set_out_sample_residuals`, `lags`, `window_features`, `differentiation`) is inapplicable. `transformer_y` / `transformer_exog` are intentionally excluded: Chronos-2 operates on raw values without an sklearn preprocessing pipeline.

```python
class ForecasterFoundational:
    """
    Forecaster wrapping a FoundationalModel for full skforecast ecosystem
    compatibility (backtesting, model_selection, etc.).

    Parameters
    ----------
    estimator : FoundationalModel
        A configured `FoundationalModel` instance (e.g.
        ``FoundationalModel("autogluon/chronos-2-small", context_length=512)``).
    forecaster_id : str, optional
        User-supplied label for the forecaster instance.
    """
```

#### Required attributes (follow `ForecasterStats` conventions)

```python
# Set at init
self.estimator: FoundationalModel  # the FoundationalModel instance passed in
self.creation_date: str            # datetime.now().strftime("%Y-%m-%d %H:%M:%S")
self.skforecast_version: str       # skforecast.__version__
self.python_version: str           # sys.version.split(" ")[0]
self.forecaster_id: str | None     # user-supplied optional label

# Derived from estimator at init
self.window_size: int              # = estimator.adapter.context_length or 1

# Set after fit
self.is_fitted: bool = False
self.fit_date: str | None = None               # renamed from fitted_date → matches ForecasterStats
self.training_range_: pd.Index | None = None   # [first_date, last_date]
self.last_window_: None = None                 # intentionally None — never stored
self.extended_index_: pd.Index | None = None   # tracks index seen beyond training (needed for last_window backtesting)
self.index_type_: type | None = None
self.index_freq_: str | pd.offsets.DateOffset | None = None
self.series_name_in_: str | None = None        # name of series used in training
self.exog_in_: bool = False                    # True if exog was used during training
self.exog_names_in_: list[str] | None = None   # exog column names seen at fit
self.exog_type_in_: type | None = None         # type of exog (Series or DataFrame)

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
def fit(self, series: pd.Series, exog: pd.DataFrame | pd.Series | None = None) -> ForecasterFoundational:
    """
    Fit the forecaster. Only stores index metadata; no training — the adapter
    stores the history for context-window purposes.
    """
    # validate series, exog
    # store: training_range_, extended_index_, index_type_, index_freq_,
    #         series_name_in_, exog_in_, exog_names_in_, exog_type_in_,
    #         is_fitted, fit_date
    # call self.estimator.fit(series, exog)
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

def summary(self) -> None:
    """Show forecaster information (delegates to print(self))."""

def __repr__(self) -> str: ...
def _repr_html_(self) -> str: ...  # note: underscore prefix, not __repr_html__
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

Located after `_backtesting_stats()` (appended at end of file).

**Signature:**
```python
@manage_warnings
def _backtesting_foundational(
    forecaster,
    series,                  # pd.Series | pd.DataFrame | dict
    cv,
    metric,
    add_aggregated_metric=True,
    exog=None,
    interval=None,           # e.g. [10, 90]
    quantiles=None,          # e.g. [0.1, 0.5, 0.9]
    verbose=False,
    show_progress=True,
    suppress_warnings=False,
):
```

**Key implementation notes:**
- Import `_check_preprocess_series_type` inside function (avoids circular import)
- `deepcopy_forecaster(forecaster)` + `deepcopy(cv)` at start
- `_check_preprocess_series_type(series)` → `is_multiseries, series_names, series_norm`
- `cv.set_params({'window_size': ..., 'return_all_indexes': False, 'verbose': verbose})`
- Reference series for `cv.split()`: first column/value for multi, series itself for single
- Initial fit on `folds[0][1]` then `folds[0][5] = False`
- `@manage_warnings` decorator handles suppress_warnings

**Fold structure (6 elements, `return_all_indexes=False`):**

| fold[i] | Content |
|---------|----------|
| fold[0] | fold number (int) |
| fold[1] | [train_start, train_end+1] |
| fold[2] | [window_start, window_end+1] — context window (window_size obs) |
| fold[3] | [test_with_gap_start, test_with_gap_end+1] — steps + gap |
| fold[4] | [test_no_gap_start, test_no_gap_end+1] — actual test for metrics |
| fold[5] | should_refit (bool) |

**Fold loop logic:**

```python
for fold in folds_tqdm:
    fold_number, (train_start, train_end), (window_start, window_end), \
        (test_gap_start, test_gap_end), (test_start, test_end), should_refit = fold

    steps_with_gap = test_gap_end - test_gap_start
    last_window = series_norm.iloc[window_start:window_end]       # context (pd.Series/df/dict)
    last_window_exog = exog.iloc[window_start:window_end]          # (if exog is not None)
    exog_test = exog.iloc[test_gap_start:test_gap_end]             # (if exog is not None)

    if should_refit:
        forecaster.fit(series=series_norm.iloc[train_start:train_end], exog=exog_train)

    pred = forecaster.predict(steps=steps_with_gap, ..., last_window=last_window)
    # OR: predict_interval(...) / predict_quantiles(...)

    # Slice to actual test period (remove gap rows)
    test_index = ref_series.iloc[test_start:test_end].index
    pred = pred.loc[test_index]  # single; or pred[pred.index.isin(test_index)] for multi
    pred.insert(0, 'fold', fold_number)
```

**For dict/DataFrame slicing**, two inner helper functions are defined locally:
```python
def _slice_series(s, i, j):
    return {k: v.iloc[i:j] for k, v in s.items()} if isinstance(s, dict) else s.iloc[i:j]

def _slice_exog(e, i, j):
    if e is None: return None
    if isinstance(e, dict): return {k: (v.iloc[i:j] if v is not None else None) for k, v in e.items()}
    return e.iloc[i:j]
```

**Multi-series metrics path:**
```python
# Convert to (idx, level) MultiIndex
backtest_predictions = backtest_predictions.rename_axis('idx').set_index('level', append=True)
# For quantile-only output: derive 'pred' from q_0.5 or closest quantile
metrics_levels = _calculate_metrics_backtesting_multiseries(
    series=series_norm, predictions=backtest_predictions_for_metrics[['pred']],
    folds=folds, span_index=span_index, window_size=forecaster.window_size,
    metrics=metrics, levels=series_names, add_aggregated_metric=add_aggregated_metric,
)
backtest_predictions = backtest_predictions.reset_index('level').rename_axis(None)
```

**Single-series metrics path** (same as `backtesting_stats`):
```python
y_train = ref_series.iloc[train_indexes]
y_true  = ref_series.loc[backtest_predictions_for_metrics.index]
y_pred  = backtest_predictions_for_metrics['pred']  # or q_0.5 for quantile mode
```

#### Public `backtesting_foundational()`

Located after `backtesting_stats()` (appended at end of file).

```python
def backtesting_foundational(
    forecaster,
    series,                  # pd.Series | pd.DataFrame | dict
    cv,
    metric,
    add_aggregated_metric=True,
    exog=None,
    interval=None,
    quantiles=None,
    verbose=False,
    show_progress=True,
    suppress_warnings=False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

**Guards:**
1. `type(forecaster).__name__ != 'ForecasterFoundational'` → TypeError
2. `interval is not None and quantiles is not None` → ValueError (cannot combine)
3. `check_backtesting_input(forecaster, cv, series=series, metric=metric, exog=exog, interval=interval, ...)`
4. Delegate to `_backtesting_foundational(...)`

---

### 4.5 `skforecast/model_selection/_utils.py` — **UPDATE**

Three locations to update:

1. **New `elif` branch in `check_backtesting_input`** (after the `forecasters_multi_dict` block,
   NOT in `forecasters_uni` — `ForecasterFoundational` accepts pd.Series/DataFrame/dict):

```python
elif forecaster_name == "ForecasterFoundational":
    if not isinstance(series, (pd.Series, pd.DataFrame, dict)):
        raise TypeError(
            f"`series` must be a pandas Series, DataFrame or dict. Got {type(series)}."
        )
    data_name = 'series'
    data_length = (
        max(len(v) for v in series.values() if v is not None)
        if isinstance(series, dict)
        else len(series)
    )
```

   Also, the `exog` check for `ForecasterFoundational` must accept dict exog:
```python
elif forecaster_name == "ForecasterFoundational":
    pass  # checks done in forecaster.fit() / predict()
```

2. **`initial_train_size is None` block** (~line 468) — add `'ForecasterFoundational'`:

```python
if forecaster_name in ['ForecasterStats', 'ForecasterFoundational', 'ForecasterEquivalentDate']:
    raise ValueError(
        f"When using {forecaster_name}, `initial_train_size` must be an "
        f"integer smaller than the length of `{data_name}` ({data_length})."
    )
```

3. **`select_n_jobs_backtesting()` early return** (~line 683) — must come BEFORE
   `forecaster.estimator` access to avoid AttributeError on ForecasterFoundational:

```python
if forecaster_name in ('ForecasterStats', 'ForecasterFoundational'):
    n_jobs = 1
    return n_jobs
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
- [x] Remove: `ChronosAdapter`, `BaseAdapter`, `_is_chronos_2_model`, `_chronos_supports_config_field`, `_get_package_version`, `inspect` import, `importlib.metadata` imports.
- [x] Remove `ModelCapabilities` dataclass (deleted — no longer used).
- [x] Add: `Chronos2Adapter` with `_load_pipeline()`, `_to_covariate_array()`, `_normalize_exog_to_dict()`, `_build_chronos_input()`, `_predict_multiseries()`, `fit()`, `predict()`.
- [x] `_load_pipeline()` uses `BaseChronosPipeline.from_pretrained()` (auto-dispatches to correct pipeline class).
- [x] `_to_covariate_array()`: numeric/bool → float64, string/categorical → as-is (Chronos-2 native).
- [x] Multi-series support: `fit(series)` accepts `pd.DataFrame` and `dict[str, pd.Series]`; `predict()` dispatches to `_predict_multiseries()` for batch inference.
- [x] `cross_learning` parameter added to both `Chronos2Adapter` and `FoundationalModel`.
- [x] Rename `FoundationalModels` → `FoundationalModel`.
- [x] `FoundationalModel` parameter renamed `model_id` → `model`.
- [x] `FoundationalModel.predict()` passes `last_window` and `last_window_exog` through to adapter.

**Validation:**
```python
from skforecast.foundational import FoundationalModel
import pandas as pd

m = FoundationalModel("autogluon/chronos-2-small")
y = pd.Series(range(50), index=pd.date_range("2020", periods=50, freq="ME"))
m.fit(y)
pred = m.predict(steps=12)
assert len(pred) == 12
```

---

### Phase 2 — Create `_forecaster_foundational.py`

**Tasks:**
- [x] Create file with `ForecasterFoundational` class.
- [x] `__init__(estimator: FoundationalModel, forecaster_id=None)`: store estimator + metadata attrs + `__skforecast_tags__`; derive `window_size` from `estimator.adapter.context_length`.
- [x] `fit()`: validate + store index metadata + call `estimator.fit(series, exog)`.
- [x] `predict()`: delegate to `estimator.predict()`.
- [x] `predict_interval()`: delegate; format output columns.
- [x] `predict_quantiles()`: delegate.
- [x] `__repr__` / `_repr_html_`: follow `ForecasterStats` pattern.
- [x] `set_params()` for compatibility.
- [x] `summary()` — delegates to `print(self)`.
- [x] Update `skforecast/foundational/__init__.py` to export `ForecasterFoundational`.

---

### Phase 3 — `backtesting_foundational` in `_validation.py`

**Tasks:**
- [x] Implement `_backtesting_foundational()` with fold loop (`series=` parameter, `add_aggregated_metric`, 6-element fold structure).
- [x] Implement `backtesting_foundational()` public function.
- [x] Handle `interval` and `quantiles` arguments; raise ValueError if both provided.
- [x] Multi-series metrics via `_calculate_metrics_backtesting_multiseries` (Option B: per-level + aggregated).
- [x] Support `metric` as str / callable / list (same as `backtesting_stats`).

---

### Phase 4 — Wire `model_selection/_utils.py`

**Tasks:**
- [x] Add new `elif forecaster_name == "ForecasterFoundational":` branch in `check_backtesting_input` (NOT in `forecasters_uni` — accepts Series/DataFrame/dict).
- [x] Add `'ForecasterFoundational'` to `initial_train_size is None` required check.
- [x] Add `'ForecasterFoundational'` to `select_n_jobs_backtesting` early-return block (before `forecaster.estimator` access).

---

### Phase 5 — Exports and Tests

**Tasks:**
- [x] Update `skforecast/foundational/__init__.py`.
- [x] Update `skforecast/model_selection/__init__.py`.
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
self.fit_date          = None   # ForecasterStats convention (not fitted_date)
self.training_range_   = None
self.last_window_      = None
self.extended_index_   = None
self.index_type_       = None
self.index_freq_       = None
self.series_name_in_   = None
self.exog_in_          = False
self.exog_names_in_    = None
self.exog_type_in_     = None
self.window_size       = None   # set from estimator.adapter.context_length at init
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
  ├── FoundationalModel      ← _foundational_models.py (DONE)
  └── ForecasterFoundational ← _forecaster_foundational.py (NEW)

skforecast.model_selection
  ├── backtesting_foundational ← _validation.py (ADD)
  └── [other existing exports]
```

---

## 9. Out of Scope

- Chronos T5 / Chronos-Bolt support (removed by user request)
- Fine-tuning / `pipeline.fit()` support
- Multi-variate (channel) series support — note: multi-series **batch** inference (multiple independent series in one call) **is** implemented via `Chronos2Adapter._predict_multiseries()`
- `grid_search_foundational` / `bayesian_search_foundational` (future work)
- `ForecasterFoundational` integration with `grid_search_forecaster` (not applicable — no hyperparameters to tune in standard usage)
