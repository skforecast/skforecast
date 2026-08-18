---
name: foundation-forecasting
description: >
  Zero-shot time series forecasting with pre-trained foundation models
  (Amazon Chronos-2, Google TimesFM 2.5, Salesforce Moirai-2, Soda-INRIA TabICL,
  Prior Labs TabPFN-TS, The Forecasting Company T0, EDF Lab TS-ICL, Synthefy Nori) via ForecasterFoundation and
  FoundationModel. Covers single and multi-series
  workflows, exogenous variables, prediction intervals / quantiles,
  backtesting, and inference-time parameter search (context_length tuning).
  Use when the user wants forecasts without task-specific
  training, cold-start baselines, or pre-trained generalist models.
---

# Foundation Model Forecasting (Zero-Shot)

## References

See [references/adapter-parameters.md](references/adapter-parameters.md) for the per-adapter constructor parameters of `ChronosAdapter`, `TimesFMAdapter`, `MoiraiAdapter`, `TabICLAdapter`, `TabPFNAdapter`, `T0Adapter`, `NoriAdapter`, and `TSICLAdapter`.

## When to Use

Use `ForecasterFoundation` when:
- You want a **zero-shot baseline** before investing in model training.
- You have **very short histories** where ML models struggle.
- You need to forecast **cold-start** series (new product, new sensor).
- You want to compare against pre-trained generalist models.

Foundation models are **pre-trained on massive corpora** — `fit()` does not train them; it only stores the recent context and metadata.

## Stop Conditions

Scan before writing code. Each row lists a rule, the symptom when it is broken, and the recovery. Full pitfall catalog: the `troubleshooting-common-errors` skill.

| Rule | Symptom | Recovery |
|------|---------|----------|
| `fit()` stores context only; it never trains the model | Expecting training to happen or weights to update | Treat the model as pre-trained; evaluate with `backtesting_foundation` |
| `cv.refit` and `cv.fixed_train_size` are overridden by `backtesting_foundation` | `IgnoredArgumentWarning` when `refit=True` or `fixed_train_size=False` | Leave them at their defaults; the context window expands per fold either way |
| Only Chronos-2, TabICL, TabPFN-TS, T0, Nori, and TS-ICL use `exog`; TimesFM 2.5 and Moirai-2 ignore it | `exog` silently dropped, no error raised | Pick an exog-capable adapter when covariates matter |
| TimesFM 2.5 and Moirai-2 restrict quantiles to `[0.1, 0.2, ..., 0.9]` | Requested quantile rejected or unsupported | Request only supported quantiles, or use an adapter allowing any quantile in (0, 1) |
| Each backend library must be installed separately | `ModuleNotFoundError` / `ImportError` on first use | `pip install` the matching backend (see Installation) |
| Tuning uses `bayesian_search_foundation`, never `bayesian_search_forecaster*` | `TypeError` on the forecaster type or on `OneStepAheadFold` | Call `bayesian_search_foundation` with a `TimeSeriesFold` |

## Installation

Foundation model backends are **not** bundled with skforecast. Install only the backend(s) you need:

```bash
pip install chronos-forecasting    # For Chronos-2
pip install timesfm                # For TimesFM 2.5
pip install uni2ts                 # For Moirai-2
pip install tabicl[forecast]       # For TabICL
pip install tabpfn-time-series     # For TabPFN-TS
pip install tfc-t0                 # For T0
pip install tsicl                  # For TS-ICL
pip install synthefy-nori          # For Nori
```

Models are downloaded from HuggingFace on first use.

## Quick Start (single series)

```python
import pandas as pd
from skforecast.foundation import FoundationModel, ForecasterFoundation

# Data must have a DatetimeIndex with a frequency
data = pd.read_csv('data.csv', index_col='date', parse_dates=True).asfreq('h')

# 1. Configure a foundation model (adapter is resolved from model_id)
model = FoundationModel(
    model_id='autogluon/chronos-2-small',
    context_length=2048,      # Adapter-specific default: see reference
    device_map='auto',        # 'auto' picks CUDA > MPS > CPU
)

# 2. Wrap it in ForecasterFoundation for the skforecast API
forecaster = ForecasterFoundation(estimator=model)

# 3. "Fit" only stores the last context_length observations (no training)
forecaster.fit(series=data['target'])

# 4. Point forecast — returns long-format DataFrame: columns ['level', 'pred']
predictions = forecaster.predict(steps=24)
```

## Multi-Series (Global Zero-Shot Model)

Pass a wide `DataFrame`, a long-format `DataFrame` (MultiIndex), or a
`dict[str, pd.Series]` to `fit`.

```python
# series: wide DataFrame — each column is one series
forecaster.fit(series=series)

# Forecast all series
predictions = forecaster.predict(steps=24)

# Forecast a subset
predictions = forecaster.predict(steps=24, levels=['series_1', 'series_2'])
```

Chronos-2 supports `cross_learning=True` to share information across series
in the batch (ignored in single-series mode):

```python
model = FoundationModel(
    model_id='autogluon/chronos-2-small',
    cross_learning=True,
)
```

## With Exogenous Variables (Chronos-2, TabICL, TabPFN-TS, TFC-T0, Nori, and TS-ICL)

Chronos-2, TabICL, TabPFN-TS, TFC-T0, Nori, and TS-ICL (`allow_exog=True`) accept exogenous variables. TimesFM 2.5 and Moirai-2 ignore them.

```python
# Historical + future exog (must cover the forecast horizon)
forecaster.fit(series=data['target'], exog=exog_train)

predictions = forecaster.predict(steps=24, exog=exog_test)
```

## Prediction Intervals and Quantiles

Foundation models output native quantile forecasts — no bootstrapping or conformal calibration is required.

```python
# Interval (lower/upper bounds from the model's quantiles)
predictions = forecaster.predict_interval(
    steps=24,
    interval=[0.1, 0.9],   # 80% prediction interval (quantiles, 0-1 scale)
)
# Columns: ['level', 'pred', 'lower_bound', 'upper_bound']

# Explicit quantiles
predictions = forecaster.predict_quantiles(
    steps=24,
    quantiles=[0.1, 0.5, 0.9],
)
# Columns: ['level', 'q_0.1', 'q_0.5', 'q_0.9']
```

For TimesFM 2.5 and Moirai-2, requested quantiles must be a subset of `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. TS-ICL accepts any level on a finer 0.01 grid in `[0.01, 0.99]` (e.g. `0.05`, `0.37`); off-grid levels raise a `ValueError`. Chronos-2, TabICL, TabPFN-TS, TFC-T0 and Nori support any quantile in `(0, 1)`.

## Choosing a Model

| Model (`model_id` prefix)              | Exog | Default context | Best for                                          |
|----------------------------------------|:----:|----------------:|---------------------------------------------------|
| `autogluon/chronos-2-*` (Amazon)       | Yes  | 8192            | General-purpose, exog-friendly, cross-series info |
| `google/timesfm-2.5-*` (Google)        | No   | 512             | Long-horizon point/quantile forecasts             |
| `Salesforce/moirai-2.0-*` (Salesforce) | No   | 2048            | Multivariate pretraining, probabilistic forecasts |
| `soda-inria/tabicl` (Soda-INRIA)       | Yes  | 4096            | Tabular in-context learning, exog-aware           |
| `priorlabs/tabpfn-ts` (Prior Labs)     | Yes  | 32768           | Tabular foundation model, exog-aware, long context |
| `theforecastingcompany/t0` (TFC)       | Yes  | 8192            | Probabilistic forecasts, exog-aware (future covariates) |
| `Synthefy/Nori` (Synthefy)             | Yes  | 4096            | Tabular foundation model, exog-aware               |
| `taharnbl/TS-ICL` (EDF Lab)            | Yes  | 4096            | Past & future covariates, fine-grained (0.01) quantile grid |

The adapter is resolved automatically from the `model_id` prefix — no need to import adapter classes directly.

## Backtesting

Use the dedicated `backtesting_foundation` function — it is the only backtester that accepts a `ForecasterFoundation`. Internally `cv` is deep-copied and forced to `refit=True`, `fixed_train_size=False`, so the context window expands with each fold up to `context_length`; no weights are ever trained. Probabilistic output is requested via `quantiles`, not `interval`.

```python
from skforecast.model_selection import backtesting_foundation, TimeSeriesFold

cv = TimeSeriesFold(
    steps=24,
    initial_train_size=len(series) - 200,
    refit=False,      # Overridden internally; passing True emits IgnoredArgumentWarning
)

metric, predictions = backtesting_foundation(
    forecaster=forecaster,
    series=series,
    cv=cv,
    metric='mean_absolute_error',
    quantiles=[0.1, 0.5, 0.9],   # Native model quantiles; no bootstrapping
)
```

## Tuning Inference-Time Parameters

No weights are trained, so tuning means choosing how the pre-trained model is
queried. `context_length` is the highest-impact parameter. Use
`bayesian_search_foundation` (`TimeSeriesFold` only, no `lags`, no `n_jobs`):

```python
from skforecast.model_selection import bayesian_search_foundation, TimeSeriesFold

def search_space(trial):
    return {
        'context_length': trial.suggest_categorical('context_length', [512, 1024, 2048, 4096]),
    }

results, study = bayesian_search_foundation(
    forecaster=forecaster,
    series=series,
    cv=cv,
    search_space=search_space,
    metric='mean_absolute_error',
    n_trials=30,
    return_best=True,
)
```

Keys are validated against the adapter's `get_params()`. Search
`context_length` and the adapter's quality-relevant parameters; runtime
settings (`device`, `torch_dtype`, `mode`, `show_progress`, `max_horizon`) are
accepted but cannot improve accuracy, and several parameters force an expensive
model reload per trial. Per-adapter matrix:
[references/adapter-parameters.md](references/adapter-parameters.md).

## Override the Stored Context

Pass `context` at predict time to forecast from a different window without
refitting — useful for one-off predictions or custom backtesting loops:

```python
predictions = forecaster.predict(
    steps=24,
    context=new_window,        # pandas Series / DataFrame / dict
    context_exog=new_exog,     # Only with exog-aware adapters
    exog=future_exog,
)
```

If `context` is longer than the adapter's `context_length`, it is trimmed
automatically to the last `context_length` observations.

## Common Pitfalls

1. **Expecting `fit()` to train the model**: it only stores context. The weights come from HuggingFace.
2. **Index without frequency**: call `series.asfreq('h')` (or similar) before `fit` — skforecast requires a frequency.
3. **Passing `exog` to TimesFM 2.5 / Moirai-2**: ignored. Only Chronos-2, TabICL, TabPFN-TS, TFC-T0, Nori, and TS-ICL support exogenous variables.
4. **Requesting unsupported quantiles**: TimesFM 2.5 and Moirai-2 are restricted to the nine deciles `0.1 … 0.9` ; TS-ICL is restricted to a 0.01 grid in `[0.01, 0.99]`.
5. **Large model downloads**: first call can be slow; consider using smaller variants (`*-small`) for experimentation.
6. **Forgetting to install the backend**: each foundation model requires its own library (`chronos-forecasting`, `timesfm`, `uni2ts`, `tabicl`, `tabpfn-time-series`, `tfc-t0`, `synthefy-nori`, `tsicl`). Install only the one(s) you need.
7. **Tuning a parameter that forces a model reload**: `model_id` and device/dtype arguments reload the model on every trial, and `context_length` does the same on TimesFM 2.5, Moirai-2, TabICL and TabPFN-TS.
