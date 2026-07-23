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
|------------------|--------|----------|----------------------------------------------------------------------------|
| `model_id`       | str    | —        | HuggingFace model ID (e.g. `theforecastingcompany/t0-alpha`).             |
| `model`          | obj    | `None`   | Pre-loaded `T0Forecaster`. If `None`, loaded lazily on first `predict`.    |
| `context_length` | int    | `8192`   | Max historical observations kept as context.                               |
| `device_map`     | str    | `'auto'` | Device placement: `'auto'` (CUDA > MPS > CPU), `'cuda'`, `'mps'`, `'cpu'`. |
| `torch_dtype`    | object | `None`   | Torch dtype the loaded model is cast to (e.g. `torch.bfloat16`).           |

Point forecasts use the median (quantile `0.5`). Covariates must be numeric; encode categoricals as numbers before passing them. A series with no future exog is forecast without covariates.

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

## Common Behavior

All adapters implement the same minimal interface:

- `fit(series, exog=None)` — stores context and metadata; no training.
- `predict(steps, context, context_exog, exog, quantiles)` — returns a   `dict[str, np.ndarray]` of shape `(steps, n_quantiles)` keyed by series name.
- `get_params()` / `set_params(**kwargs)` — sklearn-style parameter access.

Backend libraries (`chronos-forecasting`, `timesfm`, `uni2ts`, `tabicl`, `tabpfn-time-series`, `tfc-t0`, `synthefy-nori`, `tsicl`) are imported **lazily** inside the adapter method that needs them, so only the backend for the adapter you actually use needs to be installed.
