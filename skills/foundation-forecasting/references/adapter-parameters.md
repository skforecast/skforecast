# Foundation Adapter Parameters

`FoundationModel` resolves the adapter automatically from `model_id`. All keyword arguments passed to `FoundationModel(...)` beyond `model_id` are forwarded to the chosen adapter's `__init__`.

```python
from skforecast.foundation import FoundationModel

model = FoundationModel(
    model_id='autogluon/chronos-2-small',
    context_length=2048,
    device_map='auto',
)
```

## ChronosAdapter — Amazon Chronos-2

- **`model_id` prefix**: `autogluon/chronos`
- **`allow_exog`**: `True` (past and future covariates)
- **Quantiles**: any value in `(0, 1)`

| Parameter        | Type    | Default  | Description                                                                    |
|------------------|---------|----------|--------------------------------------------------------------------------------|
| `model_id`       | str     | —        | HuggingFace model ID (e.g. `autogluon/chronos-2-small`).                       |
| `pipeline`       | object  | `None`   | Pre-loaded `BaseChronosPipeline`. If `None`, loaded lazily on first `predict`. |
| `context_length` | int     | `8192`   | Max historical observations kept as context.                                   |
| `predict_kwargs` | dict    | `None`   | Extra kwargs forwarded to the pipeline's `predict_quantiles`.                  |
| `device_map`     | str     | `'auto'` | Device placement: `'auto'` (CUDA > MPS > CPU), `'cuda'`, `'mps'`, `'cpu'`.     |
| `torch_dtype`    | object  | `None`   | Torch dtype for `from_pretrained` (e.g. `torch.bfloat16`).                     |
| `cross_learning` | bool    | `False`  | If `True`, shares information across series in multi-series batches.           |

## TimesFMAdapter — Google TimesFM 2.5

- **`model_id` prefix**: `google/timesfm`
- **`allow_exog`**: `False`
- **Supported quantiles**: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`

| Parameter                | Type | Default | Description                                                         |
|--------------------------|------|---------|---------------------------------------------------------------------|
| `model_id`               | str  | —       | HuggingFace model ID (e.g. `google/timesfm-2.5-200m-pytorch`).      |
| `model`                  | obj  | `None`  | Pre-loaded & compiled TimesFM model. If `None`, loaded lazily.      |
| `context_length`         | int  | `512`   | Max historical observations kept as context.                        |
| `max_horizon`            | int  | `512`   | Max forecast horizon. `predict(steps=...)` must be ≤ this.          |
| `forecast_config_kwargs` | dict | `None`  | Extra kwargs forwarded to `timesfm.ForecastConfig` at compile time. |

The model is compiled lazily for the exact requested `steps` (up to `max_horizon`) to avoid unnecessary decode iterations.

## MoiraiAdapter — Salesforce Moirai-2

- **`model_id` prefix**: `Salesforce/moirai`
- **`allow_exog`**: `False`
- **Supported quantiles**: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`

| Parameter        | Type | Default  | Description                                                              |
|------------------|------|----------|--------------------------------------------------------------------------|
| `model_id`       | str  | —        | HuggingFace model ID (e.g. `Salesforce/moirai-2.0-R-small`).             |
| `module`         | obj  | `None`   | Pre-loaded `Moirai2Module`. If `None`, loaded lazily.                    |
| `context_length` | int  | `2048`   | Max historical observations kept as context.                             |
| `device`         | str  | `'auto'` | Device placement: `'auto'` (CUDA > MPS > CPU), `'cuda'`, `'mps'`, `'cpu'`. |

## TabICLAdapter — Soda-INRIA TabICL

- **`model_id` prefix**: `soda-inria/tabicl`
- **`allow_exog`**: `True` (past and future covariates)
- **Quantiles**: any value in `(0, 1)`

| Parameter            | Type  | Default  | Description                                                                      |
|----------------------|-------|----------|----------------------------------------------------------------------------------|
| `model_id`           | str   | —        | HuggingFace model ID (e.g. `soda-inria/tabicl`).                                 |
| `model`              | obj   | `None`   | Pre-instantiated `TabICLForecaster`. If `None`, created lazily on first predict. |
| `context_length`     | int   | `4096`   | Max historical observations kept as context.                                     |
| `point_estimate`     | str   | `'mean'` | Point forecast method: `'mean'` or `'median'`.                                   |
| `tabicl_config`      | dict  | `None`   | Extra kwargs forwarded to `TabICLRegressor` at inference time.                   |
| `temporal_features`  | list  | `None`   | `TimeTransform` instances applied before inference. `None` = TabICL defaults; `[]` = disable all. |

## TabPFNAdapter — Prior Labs TabPFN-TS

- **`model_id` prefix**: `priorlabs/tabpfn`
- **`allow_exog`**: `True` (known-future covariates; covariates without future values are discarded by the library)
- **Quantiles**: any value in `(0, 1)`

| Parameter             | Type  | Default    | Description                                                                      |
|-----------------------|-------|------------|----------------------------------------------------------------------------------|
| `model_id`            | str   | —          | Model ID (e.g. `priorlabs/tabpfn-ts`). Used only for adapter resolution.         |
| `model`               | obj   | `None`     | Pre-instantiated `TabPFNTSPipeline`. If `None`, created lazily on first predict. |
| `context_length`      | int   | `32768`    | Max historical observations kept as context. Lower (e.g. 4096) for faster inference. |
| `mode`                | str   | `'local'`  | `'local'` (on-device inference, CUDA > MPS > CPU) or `'client'` (Prior Labs cloud API, no GPU needed). |
| `point_estimate`      | str   | `'median'` | Ensemble aggregation for the point forecast: `'mean'`, `'median'` or `'mode'`.   |
| `tabpfn_model_config` | dict  | `None`     | Extra config forwarded to the underlying TabPFN regressor (e.g. `model_path`, `device`). |
| `temporal_features`   | list  | `None`     | `FeatureGenerator` instances applied before inference. `None` = TabPFN-TS defaults; `[]` = disable all. |

## T0Adapter — The Forecasting Company T0

- **`model_id` prefix**: `theforecastingcompany/t0`
- **`allow_exog`**: `True` (future-known covariates; historical and known-future values are concatenated into the `[context + horizon]` covariate stream)
- **Quantiles**: any value in `(0, 1)` (native levels `0.1, 0.25, 0.5, 0.75, 0.9`; other levels are produced by inference-time interpolation)

| Parameter        | Type   | Default  | Description                                                                |
|------------------|--------|----------|------------------------------------------------------------------------------|
| `model_id`       | str    | —        | HuggingFace model ID (e.g. `theforecastingcompany/t0-alpha`).             |
| `model`          | obj    | `None`   | Pre-loaded `T0Forecaster`. If `None`, loaded lazily on first `predict`.    |
| `context_length` | int    | `8192`   | Max historical observations kept as context.                               |
| `device_map`     | str    | `'auto'` | Device placement: `'auto'` (CUDA > MPS > CPU), `'cuda'`, `'mps'`, `'cpu'`. |
| `torch_dtype`    | object | `None`   | Torch dtype the loaded model is cast to (e.g. `torch.bfloat16`).           |

Point forecasts use the median (quantile `0.5`). Covariates must be numeric; encode categoricals as numbers before passing them. A series with no future exog is forecast without covariates.

**Gated checkpoints**: `theforecastingcompany/t0*` repos are gated on the Hugging Face Hub. Before first use, log in at the model page (e.g. `https://huggingface.co/theforecastingcompany/t0-alpha`) to accept its license, then authenticate locally (`hf auth login` or the `HF_TOKEN` environment variable). Skipping this step surfaces as a confusing `TypeError` about missing `T0Forecaster` constructor arguments rather than an authentication error.

## TSICLAdapter — EDF Lab TS-ICL

- **`model_id` prefix**: `taharnbl/TS-ICL`. Used only for adapter resolution; the checkpoint is always downloaded from the `taharnbl/TS-ICL` Hugging Face repository, controlled by `checkpoint_version`.
- **`allow_exog`**: `True` (past and future covariates, mirroring `ChronosAdapter`'s `past_covariates`/`future_covariates` format)
- **Quantiles**: subset of a 0.01 grid in `[0.01, 0.99]` (e.g. `0.05`, `0.5`, `0.37`); other levels raise a `ValueError`

| Parameter              | Type | Default            | Description                                                                |
|------------------------|------|--------------------|------------------------------------------------------------------------------|
| `model_id`             | str  | —                  | Model ID (e.g. `taharnbl/TS-ICL`). Used only for adapter resolution.        |
| `model`                | obj  | `None`             | Pre-instantiated `TSICL` model. If `None`, created lazily on first `predict`.|
| `checkpoint_version`   | str  | `'tsicl-v1.ckpt'`  | Checkpoint filename downloaded from the `taharnbl/TS-ICL` Hugging Face repo. |
| `context_length`       | int  | `4096`             | Max historical observations kept as context.                                |
| `device`               | str  | `'auto'`           | Device placement: `'auto'` (CUDA > MPS > CPU), `'cuda'`, `'mps'`, `'cpu'`. Verified empirically: the installed `tsicl` version currently falls back to CPU internally whenever CUDA is unavailable, regardless of the requested device, so `'mps'` has no effect on Apple Silicon. |
| `allow_auto_download`  | bool | `True`             | Whether to allow automatic download of the checkpoint from Hugging Face Hub. |

Covariates must be numeric; encode categoricals as numbers before passing them.

## NoriAdapter — Synthefy Nori

- **`model_id` prefix**: `Synthefy/Nori`
- **`allow_exog`**: `True` (known-future covariates; columns present in both the historical context and the forecast horizon are used as features, covariates without future values are ignored)
- **Quantiles**: any value in `(0, 1)`

| Parameter                | Type | Default  | Description                                                                                       |
|--------------------------|------|----------|-----------------------------------------------------------------------------------------------------|
| `model_id`               | str  | —        | Model ID (e.g. `Synthefy/Nori`). Used only for adapter resolution.                                 |
| `model`                  | obj  | `None`   | Pre-instantiated `NoriRegressor`. If `None`, created lazily on first `predict`.                    |
| `context_length`         | int  | `4096`   | Max historical observations kept as context.                                                        |
| `point_estimate`         | str  | `'mean'` | Point forecast method: `'mean'`, `'median'` or `'mode'`.                                            |
| `add_calendar_features`  | bool | `True`   | Add calendar features (month, day, day-of-week, day-of-year, quarter, hour) for `DatetimeIndex` series. Ignored for `RangeIndex`. |
| `n_fourier_terms`        | int  | `2`      | Number of Fourier (sin/cos) seasonal harmonics on the yearly/weekly cycles (or the running index for `RangeIndex` series). `0` disables them. |
| `nori_config`            | dict | `None`   | Extra kwargs forwarded to `NoriRegressor` at instantiation (e.g. `model_path`, `device`, `token`, `augmentations`). |

Nori frames forecasting as tabular in-context regression rather than a native sequence model: each series is featurized (running index, calendar features, Fourier terms, and known-future covariates) before being handed to `NoriRegressor`. Covariates must be numeric; encode categoricals as numbers before passing them.

## Tunable Parameters and Model Reload Cost

The parameters that can be changed after construction (via `set_params`, and therefore searched with `bayesian_search_foundation`) are exactly the keys returned by each adapter's `get_params()`. Any other key raises `ValueError`. Note that `model` / `pipeline` / `module` are constructor-only and are **not** settable.

Being accepted is not the same as being worth searching. Most adapters expose runtime settings that cannot improve accuracy: `device`, `device_map`, `torch_dtype`, `mode` (local vs cloud inference), `show_progress`, `allow_auto_download`, and `max_horizon` (a ceiling only, the TimesFM model is recompiled for the requested `steps` regardless). Keep these fixed. `model_id` (and `checkpoint_version` on TS-ICL) does change accuracy, but it selects a different pre-trained model, so compare those with separate searches instead of mixing them into one search space.

| Adapter | Accepted by `set_params` (`get_params()` keys) | Worth searching | Changing these forces a model reload |
|---------|-----------------------------------------------|-----------------|--------------------------------------|
| ChronosAdapter | `model_id`, `cross_learning`, `context_length`, `device_map`, `torch_dtype`, `predict_kwargs` | `context_length`, `cross_learning`, (`predict_kwargs`) | `model_id`, `device_map`, `torch_dtype` |
| TimesFMAdapter | `model_id`, `context_length`, `max_horizon`, `forecast_config_kwargs` | `context_length`, (`forecast_config_kwargs`) | **all of them** |
| MoiraiAdapter | `model_id`, `context_length`, `device` | `context_length` | **all of them** |
| TabICLAdapter | `model_id`, `context_length`, `point_estimate`, `tabicl_config`, `temporal_features`, `show_progress` | `context_length`, `point_estimate`, `temporal_features`, (`tabicl_config`) | all except `show_progress` |
| TabPFNAdapter | `model_id`, `context_length`, `mode`, `point_estimate`, `tabpfn_model_config`, `temporal_features`, `show_progress` | `context_length`, `point_estimate`, `temporal_features`, (`tabpfn_model_config`) | all except `show_progress` |
| T0Adapter | `model_id`, `context_length`, `device_map`, `torch_dtype` | `context_length` | `model_id`, `device_map`, `torch_dtype` |
| TSICLAdapter | `model_id`, `checkpoint_version`, `context_length`, `device`, `allow_auto_download` | `context_length` | `checkpoint_version`, `allow_auto_download` (`device` only clears the cached resolved device) |
| NoriAdapter | `model_id`, `context_length`, `point_estimate`, `add_calendar_features`, `n_fourier_terms`, `nori_config` | `context_length`, `point_estimate`, `add_calendar_features`, `n_fourier_terms`, (`nori_config`) | `model_id`, `nori_config` |

Parameters in parentheses are backend passthrough dicts: they can hold quality-relevant settings but are awkward to search, so treat them as advanced.

Two consequences that matter when tuning:

- **`context_length` is not uniformly cheap.** It reloads the model on TimesFM 2.5, Moirai-2, TabICL and TabPFN-TS; it is free on Chronos-2, T0, TS-ICL and Nori.
- **Reset on presence vs on change.** `TabICLAdapter`, `TabPFNAdapter` and `NoriAdapter` compare the old and new values first, so re-sampling an identical value costs nothing. `ChronosAdapter`, `TimesFMAdapter`, `MoiraiAdapter`, `T0Adapter` and `TSICLAdapter` reset whenever the key is passed, even if the value is unchanged, so on TimesFM 2.5 and Moirai-2 every single trial that samples `context_length` triggers a reload.

`model_id` forces a reload on every adapter except `TSICLAdapter`, where the checkpoint is selected by `checkpoint_version` instead.

## Common Behavior

All adapters implement the same minimal interface:

- `fit(series, exog=None)` — stores context and metadata; no training.
- `predict(steps, context, context_exog, exog, quantiles)` — returns a   `dict[str, np.ndarray]` of shape `(steps, n_quantiles)` keyed by series name.
- `get_params()` / `set_params(**kwargs)` — sklearn-style parameter access.

Backend libraries (`chronos-forecasting`, `timesfm`, `uni2ts`, `tabicl`, `tabpfn-time-series`, `tfc-t0`, `synthefy-nori`, `tsicl`) are imported **lazily** inside the adapter method that needs them, so only the backend for the adapter you actually use needs to be installed.
