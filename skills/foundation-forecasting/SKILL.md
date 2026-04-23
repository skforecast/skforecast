---
name: foundation-forecasting
description: >
  Zero-shot time series forecasting with pre-trained foundation models
  (Amazon Chronos-2, Google TimesFM 2.5, Salesforce Moirai-2, Soda-INRIA TabICL)
  via ForecasterFoundation and FoundationModel. Covers single and multi-series
  workflows, exogenous variables, prediction intervals / quantiles, and
  backtesting. Use when the user wants forecasts without task-specific
  training, cold-start baselines, or pre-trained generalist models.
---

# Foundation Model Forecasting (Zero-Shot)

## References

See [references/adapter-parameters.md](references/adapter-parameters.md) for
the per-adapter constructor parameters of `ChronosAdapter`,
`TimesFMAdapter`, `MoiraiAdapter`, and `TabICLAdapter`.

## When to Use

Use `ForecasterFoundation` when:
- You want a **zero-shot baseline** before investing in model training.
- You have **very short histories** where ML models struggle.
- You need to forecast **cold-start** series (new product, new sensor).
- You want to compare against pre-trained generalist models.

Foundation models are **pre-trained on massive corpora** — `fit()` does not
train them; it only stores the recent context and metadata.

## Installation

Foundation model backends are **not** bundled with skforecast. Install only
the backend(s) you need:

```bash
pip install chronos-forecasting                                 # For Chronos-2
pip install git+https://github.com/google-research/timesfm.git  # For TimesFM 2.5
pip install uni2ts                                              # For Moirai-2
pip install tabicl[forecast]                                    # For TabICL
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

## With Exogenous Variables (Chronos-2 and TabICL)

Chronos-2 and TabICL (`allow_exog=True`) accept exogenous variables.
TimesFM 2.5 and Moirai-2 ignore them.

```python
# Historical + future exog (must cover the forecast horizon)
forecaster.fit(series=data['target'], exog=exog_train)

predictions = forecaster.predict(steps=24, exog=exog_test)
```

## Prediction Intervals and Quantiles

Foundation models output native quantile forecasts — no bootstrapping or
conformal calibration is required.

```python
# Interval (lower/upper bounds from the model's quantiles)
predictions = forecaster.predict_interval(
    steps=24,
    interval=[10, 90],   # 80% prediction interval
)
# Columns: ['level', 'pred', 'lower_bound', 'upper_bound']

# Explicit quantiles
predictions = forecaster.predict_quantiles(
    steps=24,
    quantiles=[0.1, 0.5, 0.9],
)
# Columns: ['level', 'q_0.1', 'q_0.5', 'q_0.9']
```

For TimesFM 2.5 and Moirai-2, requested quantiles must be a subset of
`[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Chronos-2 and TabICL
support any quantile in `(0, 1)`.

## Choosing a Model

| Model (`model_id` prefix)              | Exog | Default context | Best for                                          |
|----------------------------------------|:----:|----------------:|---------------------------------------------------|
| `autogluon/chronos-2-*` (Amazon)       | Yes  | 8192            | General-purpose, exog-friendly, cross-series info |
| `google/timesfm-2.5-*` (Google)        | No   | 512             | Long-horizon point/quantile forecasts             |
| `Salesforce/moirai-2.0-*` (Salesforce) | No   | 2048            | Multivariate pretraining, probabilistic forecasts |
| `soda-inria/tabicl` (Soda-INRIA)       | Yes  | 4096            | Tabular in-context learning, exog-aware           |

The adapter is resolved automatically from the `model_id` prefix — no need
to import adapter classes directly.

## Backtesting

Use the dedicated `backtesting_foundation` function — it is the only
backtester that accepts a `ForecasterFoundation`. Refit is always disabled
internally (the loaded model weights are preserved across folds) and
probabilistic output is requested via `quantiles`, not `interval`.

```python
from skforecast.model_selection import backtesting_foundation, TimeSeriesFold

cv = TimeSeriesFold(
    steps=24,
    initial_train_size=len(series) - 200,
    refit=False,      # Refit is always disabled for foundation forecasters
)

metric, predictions = backtesting_foundation(
    forecaster=forecaster,
    series=series,
    cv=cv,
    metric='mean_absolute_error',
    quantiles=[0.1, 0.5, 0.9],   # Native model quantiles; no bootstrapping
)
```

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

1. **Expecting `fit()` to train the model**: it only stores context. The
   weights come from HuggingFace.
2. **Index without frequency**: call `series.asfreq('h')` (or similar)
   before `fit` — skforecast requires a frequency.
3. **Passing `exog` to TimesFM 2.5 / Moirai-2**: ignored. Only Chronos-2
   and TabICL support exogenous variables.
4. **Requesting unsupported quantiles**: TimesFM 2.5 and Moirai-2 are
   restricted to the nine deciles `0.1 … 0.9`.
5. **Large model downloads**: first call can be slow; consider using
   smaller variants (`*-small`) for experimentation.
6. **Forgetting to install the backend**: each foundation model requires
   its own library (`chronos-forecasting`, `timesfm`, `uni2ts`, `tabicl`).
   Install only the one(s) you need.
