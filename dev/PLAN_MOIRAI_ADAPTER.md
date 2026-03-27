# Plan: Implement MoiraiAdapter

## TL;DR
Add a `MoiraiAdapter` class to `_adapters.py` that supports **Moirai2 only** (`Salesforce/moirai-2.0-R-*`). Register it in `_ADAPTER_REGISTRY` under the prefix `"Salesforce/moirai"`. The implementation follows the same structural pattern as `Chronos2Adapter` and `TimesFM25Adapter`, using the high-level `Moirai2Forecast.predict()` API which accepts a list of arrays and handles padding/truncation internally.

---

## Key Technical Facts

### Moirai2 Output Format
- `Moirai2Forecast.predict(past_target: List[np.ndarray]) → np.ndarray (batch, num_quantiles, steps)`
- Each element of `past_target` has shape `(T, 1)` for univariate. Handles padding/truncation to `context_length` internally.
- Fixed quantile levels: `module.quantile_levels` defaults to `(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)` — 9 quantiles. Only these levels are supported; arbitrary levels are **not** interpolated.
- Point forecast uses `q=0.5` (median), which is one of the supported levels.
- `hparams_context(prediction_length=steps)` context manager overrides `prediction_length` at inference time.

### Exog support in Moirai2
`Moirai2Forecast.__init__` accepts `feat_dynamic_real_dim` (future-known covariates) and `past_feat_dynamic_real_dim` (past-only covariates), and `predict()` accepts `feat_dynamic_real` and `past_feat_dynamic_real` list arguments. **However**, the high-level `predict()` method contains a padding/truncation loop that iterates over all list-valued fields and clips every one to `context_length`:
```python
for key in data_entry:  # includes feat_dynamic_real
    data_entry[key][idx] = data_entry[key][idx][-self.hparams.context_length :, :]
```
`feat_dynamic_real` must cover `context_length + prediction_length` rows (past + future), so this truncation silently discards the future portion. The correct path for exog is `create_predictor()` (GluonTS `TFTInstanceSplitter`), which is much heavier. **Conclusion: exog is not usable via the direct `predict()` API.** Keep `IgnoredArgumentWarning` for now.

### Constructor parameters
- `model_id: str` — HuggingFace ID, e.g. `"Salesforce/moirai-2.0-R-small"`
- `module: Any | None = None` — pre-loaded `Moirai2Module` (lazy-loaded otherwise)
- `context_length: int = 2048` — max history observations to store/use

---

## Implementation Steps

### Phase 1: Class skeleton, init, get_params/set_params

1. Class `MoiraiAdapter` in `_adapters.py`, inserted before the `_ADAPTER_REGISTRY` dict. Add class-level constant:
   ```python
   SUPPORTED_QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
   ```
2. Constructor signature: `(model_id, *, module=None, context_length=2048)`
   - Validate `context_length` is positive int
   - Instance state: `model_id`, `_module` (module arg or None), `context_length`, `_forecast_obj=None`, `_history=None`, `_is_fitted=False`, `_is_multiseries=False`
3. `get_params()` returns `{model_id, context_length}`
4. `set_params(**params)` validates keys `{model_id, context_length}`; resets `_module=None` and `_forecast_obj=None` when `model_id` or `context_length` changes

### Phase 2: Module and forecast object loading

5. `_load_module()` — no-op if `_module` not None; otherwise:
   ```python
   from uni2ts.model.moirai2 import Moirai2Module
   self._module = Moirai2Module.from_pretrained(self.model_id)
   self._module.eval()
   ```
   Wrap `ImportError` with message to install `uni2ts`.

6. `_ensure_forecast_obj()` — no-op if `_forecast_obj` not None; otherwise calls `_load_module()` then:
   ```python
   from uni2ts.model.moirai2 import Moirai2Forecast
   self._forecast_obj = Moirai2Forecast(
       module=self._module,
       prediction_length=1,          # overridden per-predict via hparams_context
       context_length=self.context_length,
       target_dim=1,
       feat_dynamic_real_dim=0,
       past_feat_dynamic_real_dim=0,
   ).eval()
   ```

### Phase 3: fit()

7. `fit(series, exog=None)` — stores history, exactly like `TimesFM25Adapter.fit()`:
   - single `pd.Series`: `check_y()`, `_is_multiseries=False`, `_history = series.iloc[-context_length:]`
   - `pd.DataFrame` or dict: loop + `check_y()`, `_is_multiseries=True`, `_history = {name: s.iloc[-context_length:] for ...}`
   - `exog` silently accepted but ignored (no warning at fit time)
   - Set `_is_fitted=True`

### Phase 4: Core prediction helpers

8. `_run_inference(inputs_list, steps)` → `np.ndarray (n, num_q, steps)`:
   ```python
   self._ensure_forecast_obj()
   with self._forecast_obj.hparams_context(prediction_length=steps):
       raw = self._forecast_obj.predict(inputs_list)   # (n, num_q, steps)
   return raw
   ```
   Called with `inputs_list = [arr.reshape(-1, 1) for arr in ...]` where each arr is `float64`.

9. No `_extract_quantiles` helper needed. Raw output `raw[i]` has shape `(9, steps)` with axes matching `SUPPORTED_QUANTILES` in order. Index directly:
   ```python
   q_indices = [SUPPORTED_QUANTILES.index(q) for q in quantile_levels]
   result = raw[i][q_indices, :].T   # → (steps, n_q)
   ```

### Phase 5: predict() — single-series path

10. Validate: `_is_fitted or last_window is not None`, `steps >= 1`
11. Validate quantiles: each `q` must be in `SUPPORTED_QUANTILES` (exact match within 1e-9); raise `ValueError` otherwise — same pattern as `TimesFM25Adapter`
12. Issue `IgnoredArgumentWarning` if `exog is not None or last_window_exog is not None`
13. Call `_ensure_forecast_obj()`
14. Determine history series: `last_window if last_window else self._history`; if `last_window`, trim to context_length
15. `inputs_list = [history.to_numpy(dtype=float).reshape(-1, 1)]`
16. `raw = _run_inference(inputs_list, steps)` → `raw[0]` shape `(9, steps)`
17. `quantile_levels = list(quantiles) if quantiles is not None else [0.5]`
18. `q_indices = [SUPPORTED_QUANTILES.index(q) for q in quantile_levels]` (use `next(i for i, sq in enumerate(SUPPORTED_QUANTILES) if abs(q - sq) < 1e-9)` for float safety)
19. `result = raw[0][q_indices, :].T` → `(steps, n_q)`
20. `forecast_index = expand_index(history.index, steps=steps)`
21. Return: `None` quantiles → `pd.Series(result[:, 0], index, name=history.name)`; else → `pd.DataFrame(result, index, columns=[f"q_{q}" for q in quantile_levels])`

### Phase 6: predict() — multi-series path

22. `_predict_multiseries(steps, quantiles, last_window)`:
    - Resolve `history_dict` from `last_window` (DataFrame → dict) or `self._history`
    - If `last_window is not None`, trim each series to `context_length`
    - `series_names = list(history_dict.keys())`
    - Build batched `inputs_list = [history_dict[n].to_numpy(dtype=float).reshape(-1, 1) for n in series_names]`
    - `raw = _run_inference(inputs_list, steps)` → `(n_series, 9, steps)`
    - `quantile_levels = list(quantiles) if quantiles is not None else [0.5]`
    - `q_indices = [next(i for i, sq in enumerate(SUPPORTED_QUANTILES) if abs(q - sq) < 1e-9) for q in quantile_levels]`
    - `preds_per_series = [raw[i][q_indices, :].T for i in range(n_series)]` → each `(steps, n_q)`
    - `forecast_index = expand_index(history_dict[series_names[0]].index, steps=steps)`
    - `long_index = np.repeat(forecast_index, n_series)`
    - `level_col = np.tile(series_names, steps)`
    - Assemble long DataFrame following exact same pattern as `Chronos2Adapter._predict_multiseries()`:
      - Point: `{"level": level_col, "pred": point_matrix.ravel()}`
      - Quantiles: `{"level": level_col, "q_0.1": ..., ...}`
    - Return `pd.DataFrame(data_dict, index=long_index)`

23. In `predict()` main method, switch to `_predict_multiseries()` if `self._is_multiseries or isinstance(last_window, (pd.DataFrame, dict))`

### Phase 7: Registry update

24. Replace `# "amazon/moirai": MoiraiAdapter,` with `"Salesforce/moirai": MoiraiAdapter,` in `_ADAPTER_REGISTRY`.

---

## Relevant files
- `skforecast/foundational/_adapters.py` — the only file to modify. Insert `MoiraiAdapter` class just before the `_ADAPTER_REGISTRY` dict.

## Verification
1. Check `"Salesforce/moirai-2.0-R-small"` resolves via `_resolve_adapter()` (prefix match)
2. Check `predict()` with `quantiles=None` returns a `pd.Series` for single-series (`q=0.5` median)
3. Check `predict()` with `quantiles=[0.1, 0.5, 0.9]` returns `pd.DataFrame` with correct columns
4. Check `predict()` raises `ValueError` for a non-supported quantile level (e.g. `q=0.15`)
5. Check multi-series long-format output: shapes of `long_index`, `level_col`, and data columns match `n_steps × n_series` rows
6. Check `_ensure_forecast_obj()` is idempotent (called multiple times returns the same object)
7. Check `IgnoredArgumentWarning` fires on `exog` or `last_window_exog`
8. Check `set_params(context_length=512)` resets both `_module` and `_forecast_obj`

## Decisions
- **Moirai2 only** — no Moirai 1.x / MoE. Other prefixes continue to raise ValueError from `_resolve_adapter()`.
- **No exog support**: `IgnoredArgumentWarning` on `exog` and `last_window_exog`. Moirai2 supports covariates in principle, but the high-level `predict()` path truncates `feat_dynamic_real` to `context_length`, discarding the future portion. The GluonTS `create_predictor()` path handles it correctly but is much heavier. Not implemented for now.
- **`SUPPORTED_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`** — exact match only, same approach as `TimesFM25Adapter`. ValueError for any level outside this list.
- **No quantile interpolation** — raw output indices are used directly.
- **Batched multi-series**: single `predict()` call with all series in the list (Moirai2's `predict()` handles variable-length series internally).
- **Point forecast = median** (q=0.5), exact match in supported levels.

## Further Considerations
1. **Device handling**: `Moirai2Module.from_pretrained()` places the model on CPU by default. For GPU, users should pass a pre-loaded GPU module via the `module` parameter. A `device_map` parameter could be added later.
2. **Exog via `create_predictor()`**: if exog support is added in the future, the adapter would need to switch to the GluonTS `PyTorchPredictor` path and lose the simpler batched numpy API. This is a significant refactor.
