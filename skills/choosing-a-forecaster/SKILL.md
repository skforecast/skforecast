---
name: choosing-a-forecaster
description: >
  Guides selection of the appropriate skforecast forecaster based on the user's
  data characteristics and requirements. Provides a decision matrix mapping
  use cases to forecaster classes. Use when the user is unsure which forecaster
  to use or asks for a recommendation.
---

# Choosing a Forecaster

## When to Use This Skill

Use this skill when the user needs help choosing a forecaster, comparing forecaster types (recursive vs direct, single vs multi-series), or understanding which skforecast class fits their problem.

### Related skills

- **After**: `forecasting-single-series` (apply the chosen forecaster to one target series)
- **After**: `forecasting-multiple-series` (apply the chosen forecaster to several series jointly)
- **After**: `autocorrelation-and-lag-selection` (analyse the series dynamics before configuring `lags`)
- **After**: `feature-engineering` (build the input feature set: calendar, rolling, exogenous)

## Overview

Skforecast is a **machine learning-first** library. The ML forecasters are the primary tools; statistical models (`ForecasterStats`) and naive baselines (`ForecasterEquivalentDate`) serve as comparison benchmarks.

## Step 1 — How Many Series?

| Scenario | Go to |
|----------|-------|
| **1 target series** (with or without exogenous variables) | → Step 2a |
| **Multiple series** to forecast simultaneously | → Step 2b |
| **Multiple series as drivers** to predict one target | → `ForecasterDirectMultiVariate` |
| **Categorical target** (e.g., low/medium/high) | → `ForecasterRecursiveClassifier` |

## Step 2a — Single Series

| Scenario | Recommended Forecaster | Why |
|----------|----------------------|-----|
| **General purpose** (start here) | `ForecasterRecursive` | Default choice. One model, recursive multi-step. Works with any sklearn-compatible estimator (LightGBM, XGBoost, CatBoost, RandomForest, etc.). Supports lags, window features, exog, differentiation, transformers, weight functions, and all probabilistic prediction methods (bootstrapping, conformal, quantiles, distributions) |
| **Horizon-dependent patterns** (e.g., predicting at 1h vs 24h requires different relationships) | `ForecasterDirect` | Trains one independent model per step — no error propagation. Better when the predictive relationship changes significantly across the forecast horizon. Requires `steps` at init; parallelizable with `n_jobs` |
| **Statistical baseline** | `ForecasterStats` | Wraps ARIMA, SARIMAX, ETS, ARAR. Use as a benchmark to compare against ML models, or when the series is very short (< 200 obs) and ML overfits |
| **Zero-shot / cold-start / no training data** | `ForecasterFoundation` | Wraps pre-trained foundation models (Chronos-2, TimesFM 2.5, Moirai-2, TabICL). `fit()` only stores context — no training. Good baseline and cold-start option. See the `foundation-forecasting` skill |
| **Naive baseline** | `ForecasterEquivalentDate` | Predicts using equivalent past dates (e.g., same weekday last week). Use as a sanity-check baseline |

## Step 2b — Multiple Series

| Scenario | Recommended Forecaster | Why |
|----------|----------------------|-----|
| **Forecast many series with a shared model** (start here) | `ForecasterRecursiveMultiSeries` | One global model learns cross-series patterns. Supports DataFrame or dict input (dict allows series with different date ranges). Encoding options: `'ordinal'` (default), `'ordinal_category'`, `'onehot'`, `None`. Supports per-series transformers, per-series differentiation, series_weights |
| **Other series are features for one target** | `ForecasterDirectMultiVariate` | All series become input features to predict a single `level`. Per-series lags via dict (`{'sales': [1,7], 'price': [1]}`). One model per step — no error propagation |
| **Deep learning / complex nonlinear patterns** | `ForecasterRnn` | Keras-based RNN/LSTM/GRU. Single model outputs all steps and levels simultaneously via 3D tensors. Only conformal intervals (no bootstrapping). Requires keras |
| **Zero-shot / pre-trained generalist** | `ForecasterFoundation` | Global zero-shot forecasts via Chronos-2 / TimesFM 2.5 / Moirai-2 / TabICL. `fit()` only stores context. Native quantile intervals. Chronos-2 and TabICL support exog; TimesFM 2.5 & Moirai-2 do not. See the `foundation-forecasting` skill |

## Decision Flowchart

```
How many series?
│
├─► 1 series
│   │
│   ├─► Is it a classification problem? ──► Yes ──► ForecasterRecursiveClassifier
│   │
│   └─► Regression (continuous target)
│       │
│       ├─► Does the forecast relationship change across the horizon?
│       │   ├─► No / Unsure ──► ForecasterRecursive  ← START HERE
│       │   └─► Yes (step-specific patterns) ──► ForecasterDirect
│       │
│       └─► Compare with baselines:
│           ├─► ForecasterStats (Auto-ARIMA, ETS) as statistical benchmark
│           └─► ForecasterEquivalentDate as naive benchmark
│
└─► Multiple series
    │
    ├─► Want to forecast ALL series (global model)?
    │   └─► ForecasterRecursiveMultiSeries  ← START HERE
    │
    ├─► Want to use other series AS FEATURES for one target?
    │   └─► ForecasterDirectMultiVariate
    │
    └─► Need deep learning for very complex patterns?
        └─► ForecasterRnn

Zero-shot / no training data / cold-start (single or multi-series)?
    └─► ForecasterFoundation (Chronos-2 / TimesFM 2.5 / Moirai-2 / TabICL)
```

## Key Comparisons

### Recursive vs Direct (Single Series)

| Aspect | ForecasterRecursive | ForecasterDirect |
|--------|-------------------|-----------------|
| Models trained | 1 | N (one per step) |
| Error propagation | Yes (predictions feed into next step) | No (each step uses only observed data) |
| Forecast horizon | Flexible (any `steps` at predict time) | Fixed at init (`steps` required) |
| Training speed | Fast (one model) | Slower (N models, parallelizable via `n_jobs`) |
| Memory | Lower (1 estimator) | Higher (N estimators stored in `estimators_` dict) |
| Best for | Most cases, especially short-to-medium horizons | Long horizons where patterns change per step |
| Prediction intervals | Bootstrapping + conformal | Bootstrapping + conformal |

### RecursiveMultiSeries vs DirectMultiVariate (Multiple Series)

| Aspect | ForecasterRecursiveMultiSeries | ForecasterDirectMultiVariate |
|--------|-------------------------------|------------------------------|
| Goal | Forecast **all series** with one model | Use all series as **features** for one target |
| Input | DataFrame or dict of Series | DataFrame (all series as columns) |
| Target | All series (selected via `levels`) | One series (specified via `level`) |
| Strategy | Recursive (1 model, predictions feed back) | Direct (1 model per step, no error propagation) |
| Series identification | Encoding: `'ordinal'`, `'ordinal_category'`, `'onehot'`, `None` | All series create separate lag columns |
| Series with different ranges | Yes (via dict input) | No (all must share same range) |
| Per-series lags | No (same lags for all) | Yes (dict: `{'sales': [1,7], 'price': [1]}`) |
| Series weights | Yes (`series_weights` param) | No |
| Per-series transformers | Yes (`transformer_series` dict) | Yes (`transformer_series` dict) |
| Per-series differentiation | Yes (`differentiation` dict) | Yes (via `differentiator_` per series) |

### ML Forecasters vs Statistical Models

| Aspect | ML Forecasters | ForecasterStats |
|--------|---------------|-----------------|
| **Primary role** | Main forecasting tools | Comparison baseline |
| Best for | Medium-to-large datasets, complex patterns, many exogenous features | Short series, interpretability, parametric intervals |
| Estimators | Any sklearn-compatible (LightGBM, XGBoost, RF, etc.) | Arima, Sarimax, Ets, Arar |
| Lags / window features | Full support (`lags`, `RollingFeatures`) | No (model handles its own structure) |
| Differentiation | Built-in (`differentiation` param) | Handled within model (e.g., ARIMA `d` parameter) |
| Exogenous variables | Full support | Only SARIMAX |
| Prediction intervals | Bootstrapping (binned residuals) + conformal | Built-in parametric |
| Tuning | `grid_search_forecaster`, `random_search_forecaster`, `bayesian_search_forecaster` | `grid_search_stats`, `random_search_stats` |
| Backtesting | `backtesting_forecaster` / `backtesting_forecaster_multiseries` | `backtesting_stats` |
| Feature selection | `select_features` / `select_features_multiseries` | Not applicable |

## Feature Support Matrix

| Feature | Recursive | Direct | RecursiveMultiSeries | DirectMultiVariate | Rnn | Stats | EquivalentDate | Classifier |
|---------|:---------:|:------:|:-------------------:|:-----------------:|:---:|:-----:|:--------------:|:----------:|
| Lags | ✓ | ✓ | ✓ | ✓ (per-series dict) | ✓ | — | — | ✓ |
| Window features (`RollingFeatures`) | ✓ | ✓ | ✓ | ✓ | — | — | — | ✓ |
| Exogenous variables | ✓ | ✓ | ✓ | ✓ | ✓ | SARIMAX only | — | ✓ |
| Differentiation | ✓ | ✓ | ✓ (per-series) | ✓ (per-series) | — | Within model | — | — |
| Transformer y/series | ✓ | ✓ | ✓ (per-series) | ✓ (per-series) | ✓ | ✓ | — | — |
| Transformer exog | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| Weight function | ✓ | ✓ | ✓ (per-series) | ✓ | — | — | — | ✓ |
| Bootstrapping intervals | ✓ | ✓ | ✓ | ✓ | — | — | — | — |
| Conformal intervals | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |
| Binned residuals | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |
| Quantile predictions | ✓ | ✓ | ✓ | ✓ | — | — | — | — |
| Distribution fitting | ✓ | ✓ | ✓ | ✓ | — | — | — | — |
| Class probabilities | — | — | — | — | — | — | — | ✓ |
| Feature importances | ✓ | ✓ | ✓ | ✓ | — | — | — | — |

> **Legend:** ✓ = supported, — = not supported/not applicable.

## Quick Start Recommendations

Once you have chosen a forecaster, follow these steps to get started:

1. **Define your problem**: 1 series → `ForecasterRecursive`; multiple series → `ForecasterRecursiveMultiSeries`
2. **Choose an estimator**: LightGBM (`LGBMRegressor`) is the best starting point — fast, handles categoricals, good defaults
3. **Add features**: Use `RollingFeatures` (rolling mean, std, min, max) and `DateTimeFeatureTransformer` or `create_datetime_features` as exogenous variables
4. **Handle non-stationarity**: Use the `differentiation` parameter instead of manual differencing
5. **Evaluate with backtesting**: `backtesting_forecaster` + `TimeSeriesFold` for realistic multi-step evaluation
6. **Tune hyperparameters**: `bayesian_search_forecaster` (Optuna-based) — can include `lags` in the search space
7. **Add prediction intervals**: `predict_interval(method='bootstrapping', use_binned_residuals=True)` for uncertainty quantification
8. **Compare with baselines**: Use `ForecasterStats` (Auto-ARIMA: `Arima(order=None)`) and `ForecasterEquivalentDate` to verify the ML model adds value
