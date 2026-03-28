# Plan: TTMAdapter for IBM Granite TinyTimeMixers

## TL;DR
Add `TTMAdapter` class to `skforecast/foundational/_adapters.py`, following the `Chronos2Adapter`/`TimesFM25Adapter` pattern. Register under `"ibm-granite/granite-timeseries-ttm"` in `_ADAPTER_REGISTRY`. TTM DOES support exogenous variables via channel-concatenation (`allow_exogenous = True`). The adapter supports past-exog-as-context in zero-shot mode (FCM disabled); future exog requires `enable_forecast_channel_mixing=True` (fine-tuned model). No quantile support in zero-shot mode. No changes needed to `__init__.py`.

---

## Key Facts (from Research)

- **Package**: `pip install granite-tsfm` → installs `tsfm_public` namespace  
- **Key import**: `from tsfm_public.toolkit.get_model import get_model`  
- **Direct inference**: `output = model(past_values=torch.tensor(data))` where `data.shape = (batch, ctx_len, channels)`  
- **Output**: `output.prediction_outputs` shape `(batch_size, prediction_length, n_target_channels)` — de-normalized (model scales internally via `TinyTimeMixerStdScaler`)  
- **No manual scaling needed** — model handles it internally  
- **`get_model()` accepts `prediction_length=steps`** which sets `prediction_filter_length` to truncate output. Works for any steps ≤ model's native prediction length. Raises its own error if steps exceed supported length.  
- **Zero-shot only** — no quantile support; Trainer-based fine-tuning not in scope  
- **Model silently truncates context** if `past_values.shape[1] > sequence_length`

### Exogenous Variable Support (confirmed from source code)

**TTM DOES support exogenous variables** via a channel-concatenation approach:

1. **`TinyTimeMixerConfig`** key parameters:
   - `exogenous_channel_indices: list` — channel indices in `past_values` that are exogenous (known in forecast period)
   - `prediction_channel_indices: list` — channels to forecast. Auto-set to `all - exogenous` when `exogenous_channel_indices` is given
   - `enable_forecast_channel_mixing: bool` — enables `ForecastChannelHeadMixer` (FCM) block for future-exog infusion
   - `num_input_channels: int` — total channels (target + exog)

2. **Inference with exog**: 
   - `past_values` shape `(batch, ctx_len, n_targets + n_exog)` — concat target + past exog as channels
   - `future_values` shape `(batch, steps, n_targets + n_exog)` — zeros for target channels + real values for exog channels (required when FCM enabled and `exogenous_channel_indices` is set — raises `ValueError` if None)
   - `output.prediction_outputs` shape `(batch, steps, n_targets)` — filtered by `prediction_channel_indices`

3. **Why channel override works in zero-shot** (key architectural insight):
   - In `common_channel` mode (default for all pre-trained TTMs), NO weight matrix has `num_input_channels` as a dimension  
   - `patcher = nn.Linear(patch_length, d_model)` — independent of channel count  
   - `ChannelFeatureMixerBlock` (which uses `num_input_channels`) is only instantiated in `mix_channel` mode  
   - Therefore, overriding `num_input_channels` via `from_pretrained(**kwargs)` works without weight mismatch in `common_channel` mode

4. **Two exog modes supported in adapter**:
   - **Past exog only** (`exog=None` at predict, `last_window_exog` provided or stored) — model sees past exog as extra context channels but future exog unavailable; `enable_forecast_channel_mixing=False`
   - **Past + future exog** (`exog` at predict) — model sees past+future exog; requires `enable_forecast_channel_mixing=True` (needs FCM-enabled model, which needs fine-tuning or special pre-training)

5. **Practical constraint**: For zero-shot, use past-exog-only mode (FCM disabled). Future exog requires fine-tuned model. The adapter handles both cases cleanly.

---

## Steps

### Phase 1 — `TTMAdapter` class (in `_adapters.py`, lines 1795–1796, before registry)

1. **Write class docstring and class attributes**  
   - `allow_exogenous: bool = True`  
   - No `SUPPORTED_QUANTILES` (raise ValueError if `quantiles` is requested)

2. **Write `__init__`** — mirror `TimesFM25Adapter.__init__` structure  
   - params: `model_id`, `model=None`, `context_length=512`, `prefer_longer_context=True`, `freq_prefix_tuning=False`, `freq=None`  
   - Validate `context_length` is positive int  
   - Set: `self.model_id`, `self._model = model`, `self.context_length`, `self.prefer_longer_context`, `self.freq_prefix_tuning`, `self.freq`, `self._history = None`, `self._exog_history = None`, `self._is_fitted = False`, `self._is_multiseries = False`, `self._steps_loaded = None`, `self._n_target = None`, `self._n_exog = 0`

3. **Write `get_params()`** — return dict with all init params

4. **Write `set_params(**params)`**  
   - Valid keys: `model_id`, `context_length`, `prefer_longer_context`, `freq_prefix_tuning`, `freq`  
   - Reset `self._model = None` and `self._steps_loaded = None` when any param changes  
   - Validate `context_length`

5. **Write `_load_model(steps: int, n_target: int, n_exog: int = 0)`**  
   - No-op if `self._model is not None and self._steps_loaded == steps and self._n_exog == n_exog`  
   - Lazy import: `from tsfm_public.toolkit.get_model import get_model`  
   - Raise `ImportError` with install hint if missing  
   - Build config overrides for exog (if `n_exog > 0`):
     ```python
     config_overrides = {
         "num_input_channels": n_target + n_exog,
         "prediction_channel_indices": list(range(n_target)),
         "exogenous_channel_indices": list(range(n_target, n_target + n_exog)),
     }
     ```
   - Call: `get_model(self.model_id, context_length=self.context_length, prediction_length=steps, prefer_longer_context=self.prefer_longer_context, freq_prefix_tuning=self.freq_prefix_tuning, freq=self.freq, **config_overrides)`  
   - Set `model.eval()`, `self._model = model`, `self._steps_loaded = steps`, `self._n_target = n_target`, `self._n_exog = n_exog`

6. **Write `fit(series, exog=None)`** — mirror `MoiraiAdapter.fit()` but store exog:  
   - `pd.Series` → single-series: set `_is_multiseries = False`, store `self._history = series.iloc[-self.context_length:].copy()`; store `self._exog_history = exog.iloc[-self.context_length:].copy() if exog is not None else None`
   - `pd.DataFrame` or `dict` → multi-series: set `_is_multiseries = True`, store `self._history = {name: s.iloc[-ctx:].copy() ...}`; store `self._exog_history = exog` if provided  
   - Raise `TypeError` for invalid input type  
   - Set `self._is_fitted = True`, return `self`

7. **Write `_predict_single(steps, last_window, last_window_exog=None, exog=None)`**  
   - `history = last_window if last_window is not None else self._history`  
   - Trim: `if last_window is not None: history = history.iloc[-self.context_length:]`  
   - Resolve exog window: `exog_win = last_window_exog or self._exog_history` (trimmed to match history length)  
   - Import `torch` lazily  
   - Build `past_values` tensor:
     ```python
     target_arr = history.to_numpy(dtype=float).reshape(-1, 1)  # (T, 1)
     if exog_win is not None:
         exog_arr = exog_win.to_numpy(dtype=float).reshape(len(history), -1)  # (T, n_exog)
         combined = np.hstack([target_arr, exog_arr])  # (T, 1+n_exog)
     else:
         combined = target_arr
     past_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)  # (1, T, 1+n_exog)
     ```
   - Build `future_values` tensor (if `exog` at predict time):
     ```python
     if exog is not None:
         future_zeros = np.zeros((steps, 1))
         future_exog_arr = exog.to_numpy(dtype=float)[:steps]
         future_combined = np.hstack([future_zeros, future_exog_arr])
         future_tensor = torch.tensor(future_combined, dtype=torch.float32).unsqueeze(0)
     else:
         future_tensor = None
     ```
   - `with torch.no_grad(): output = self._model(past_values=past_tensor, future_values=future_tensor)`  
   - `preds = output.prediction_outputs[0, :, 0].detach().numpy()` → `(steps,)`  
   - `forecast_index = expand_index(history.index, steps=steps)`  
   - Return `pd.Series(preds, index=forecast_index, name=history.name)`

8. **Write `_predict_multiseries(steps, last_window, last_window_exog=None, exog=None)`**  
   - Resolve `history_dict` from `last_window` (DataFrame/dict) or `self._history`  
   - Trim to `context_length` when `last_window is not None`  
   - Import `torch` lazily  
   - **Loop per series** (single-channel call per series to avoid alignment issues):  
     - Build `(1, T_i, 1+n_exog)` tensor per series name  
     - Pass `future_values` if `exog` provided  
     - `output = self._model(past_values=tensor, future_values=future_tensor)` → `preds = output.prediction_outputs[0, :, 0].detach().numpy()` → `(steps,)`  
   - Use first series' index for `forecast_index = expand_index(..., steps=steps)`  
   - Build long-format DataFrame: `long_index = np.repeat(forecast_index, n_series)`, `level_col = np.tile(series_names, steps)`, `pred_col = np.column_stack([preds_per_series]).ravel()`  
   - Return `pd.DataFrame({"level": level_col, "pred": pred_col}, index=long_index)`

9. **Write `predict(steps, exog=None, quantiles=None, last_window=None, last_window_exog=None)`**  
   - Guard: `if not self._is_fitted and last_window is None: raise ValueError`  
   - Guard: `if not isinstance(steps, (int, np.integer)) or steps < 1: raise ValueError`  
   - Guard: `if quantiles is not None: raise ValueError("TTMAdapter produces point forecasts only...")`  
   - Resolve `n_target` and `n_exog` from available history/exog  
   - Call `self._load_model(steps, n_target=n_target, n_exog=n_exog)`  
   - Route: if `self._is_multiseries or isinstance(last_window, (pd.DataFrame, dict))` → `_predict_multiseries(steps, last_window, last_window_exog, exog)` else `_predict_single(steps, last_window, last_window_exog, exog)`

### Phase 2 — Registry update

10. **Update `_ADAPTER_REGISTRY`** in `_adapters.py` (line ~1797):  
    - Replace the commented-out `# "ibm/TTM": TTMAdapter,` with `"ibm-granite/granite-timeseries-ttm": TTMAdapter,`  
    - This prefix matches `ibm-granite/granite-timeseries-ttm-r1`, `...-ttm-r2`, `...-ttm-r2.1`

### Phase 3 — Tests

11. **Create** `skforecast/foundational/tests/tests_foundational_models/test_TTMAdapter.py`  
    Using `FakeTTMModel` (no torch/tsfm_public needed):  
    ```python
    class FakeTTMPredictionOutput:
        def __init__(self, prediction_outputs):
            self.prediction_outputs = prediction_outputs

    class FakeTTMModel:
        def __init__(self, steps, n_target=1):
            self._steps = steps
            self._n_target = n_target
            self.last_past_values = None
            self.last_future_values = None
        def __call__(self, *, past_values, future_values=None, **kwargs):
            self.last_past_values = past_values
            self.last_future_values = future_values
            batch = past_values.shape[0]
            preds = torch.zeros(batch, self._steps, self._n_target)
            return FakeTTMPredictionOutput(preds)

    def make_adapter(steps=12, n_exog=0, **kwargs) -> TTMAdapter:
        adapter = TTMAdapter(model_id=kwargs.pop("model_id", "ibm-granite/granite-timeseries-ttm-r2"), **kwargs)
        adapter._model = FakeTTMModel(steps=steps)
        adapter._steps_loaded = steps
        adapter._n_target = 1
        adapter._n_exog = n_exog
        return adapter
    ```

    Test cases:
    - `test_TTMAdapter_init_default_params` — check all attrs  
    - `test_TTMAdapter_allow_exogenous_is_True`  
    - `test_TTMAdapter_init_custom_context_length`  
    - `test_TTMAdapter_init_invalid_context_length_raises`  
    - `test_TTMAdapter_get_params`  
    - `test_TTMAdapter_set_params_valid`  
    - `test_TTMAdapter_set_params_invalid_key_raises`  
    - `test_TTMAdapter_set_params_resets_model`  
    - `test_TTMAdapter_fit_single_series`  
    - `test_TTMAdapter_fit_single_series_stores_exog`  
    - `test_TTMAdapter_fit_multi_series_dataframe`  
    - `test_TTMAdapter_fit_multi_series_dict`  
    - `test_TTMAdapter_fit_invalid_type_raises`  
    - `test_TTMAdapter_predict_before_fit_raises`  
    - `test_TTMAdapter_predict_quantiles_raises`  
    - `test_TTMAdapter_predict_exog_warns`  
    - `test_TTMAdapter_predict_single_series_returns_series`  
    - `test_TTMAdapter_predict_single_series_uses_last_window`  
    - `test_TTMAdapter_predict_single_series_trims_last_window`  
    - `test_TTMAdapter_predict_single_series_result_shape`  
    - `test_TTMAdapter_predict_single_series_result_index`  
    - `test_TTMAdapter_predict_single_series_with_exog`  
    - `test_TTMAdapter_predict_single_series_passes_future_values_when_exog`  
    - `test_TTMAdapter_predict_multi_series_returns_long_dataframe`  
    - `test_TTMAdapter_predict_multi_series_columns`  
    - `test_TTMAdapter_predict_multi_series_uses_last_window_dataframe`  
    - `test_TTMAdapter_predict_multi_series_uses_last_window_dict`  
    - `test_TTMAdapter_predict_multi_series_with_exog`

---

## Relevant Files

- `skforecast/foundational/_adapters.py` — insert `TTMAdapter` before `_ADAPTER_REGISTRY` (~line 1795); update registry entry at line 1797–1802  
- `skforecast/foundational/tests/tests_foundational_models/test_TTMAdapter.py` — new file, following `test_MoiraiAdapter.py`/`test_TimesFM25Adapter.py` pattern  

**Reference patterns:**
- `MoiraiAdapter.fit()` (lines 1590–1660) — single/multi-series history storage pattern  
- `MoiraiAdapter.predict()` (lines 1720–1795) — guard checks, routing pattern  
- `MoiraiAdapter._predict_multiseries()` (lines 1665–1720) — long-format DataFrame construction  
- `TimesFM25Adapter._load_model()` (lines 960–1000) — lazy import with `ImportError` hint  
- `TimesFM25Adapter.set_params()` (lines 850–925) — validation and model reset pattern  

---

## Verification

1. Run existing foundational tests to ensure no regressions: `pytest skforecast/foundational/tests/ -v`  
2. Run new TTMAdapter tests: `pytest skforecast/foundational/tests/tests_foundational_models/test_TTMAdapter.py -v`  
3. Manual smoke test (requires `granite-tsfm` installed):
   ```python
   import pandas as pd, numpy as np
   from skforecast.foundational._adapters import TTMAdapter
   y = pd.Series(np.random.randn(200), index=pd.date_range("2020", periods=200, freq="h"))
   adapter = TTMAdapter("ibm-granite/granite-timeseries-ttm-r2")
   adapter.fit(y)
   preds = adapter.predict(steps=24)
   assert isinstance(preds, pd.Series) and len(preds) == 24
   ```
4. Verify `_resolve_adapter("ibm-granite/granite-timeseries-ttm-r2")` returns `TTMAdapter`

---

## Decisions

- **Quantile support**: Not supported (zero-shot mode is point-forecast only). Raise `ValueError` with explanation if `quantiles` is requested. No `SUPPORTED_QUANTILES` on the class.
- **Device/dtype**: Not parameterized in v1. `get_model()` defaults to CPU float32.  
- **Multi-series inference**: Loop per series (single-channel call) rather than batching as n_channels, to avoid alignment issues with variable-length histories. Can be optimized later.  
- **Steps-change reloading**: Model is reloaded when `steps` changes, since `get_model()` bakes `prediction_filter_length` at load time. Also reloads when `n_exog` changes.  
- **Registry prefix**: `"ibm-granite/granite-timeseries-ttm"` — covers r1, r2, r2.1 families.  
- **`__init__.py`**: No changes needed — adapters are internal-only.  
- **Exog support**: `allow_exogenous = True`. Past-exog as context works zero-shot in `common_channel` mode (no weight mismatch). Future-exog via FCM requires fine-tuned model (handled by passing `future_values` tensor at predict time when `exog` is provided, but FCM must be pre-enabled in model config).
- **Out of scope**: Fine-tuning, async inference, quantile support, FCM-enabled model pre-training.
