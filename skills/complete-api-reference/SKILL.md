---
name: complete-api-reference
description: >
  Provides complete constructor and method signatures for all skforecast
  forecasters, backtesting functions, search functions, cross-validation
  classes, preprocessing, feature selection, and drift detection.
  Use when the user needs exact parameter names, types, or defaults for
  any skforecast class or function.
---

# Complete API Reference

## When to Use

Use this when you need exact parameter names, types, defaults, or method signatures for any skforecast class or function.

### Related skills

- **Prerequisite**: `choosing-a-forecaster` (pick the class first; this skill only supplies its signature)
- **Alongside**: any workflow skill (they show the idiomatic usage, this one the exact arguments)
- **Next**: `troubleshooting-common-errors` (when a signature looks right but the call still fails)

## Quick Index

### Forecaster Constructors
- `ForecasterRecursive` — single series, recursive strategy
- `ForecasterRecursiveMultiSeries` — multiple series, global model
- `ForecasterDirect` — single series, one model per step
- `ForecasterDirectMultiVariate` — multiple input series, one target
- `ForecasterRecursiveClassifier` — classification-based
- `ForecasterStats` — statistical models (ARIMA, ETS, SARIMAX, ARAR)
- `ForecasterEquivalentDate` — baseline using past offsets
- `ForecasterRnn` — deep learning (RNN/LSTM/GRU)
- `ForecasterFoundation` — zero-shot with foundation models (Chronos-2, TimesFM 2.5, Moirai-2, TabICL, TabPFN-TS, TFC-T0, Nori, TS-IC)
- `FoundationModel` — low-level foundation model wrapper used by `ForecasterFoundation`

### Forecaster Methods
- `fit()` — train the model
- `predict()` — generate point forecasts
- `predict_interval()` — generate prediction intervals

### Model Selection
- `backtesting_forecaster` — backtest single-series forecasters
- `backtesting_forecaster_multiseries` — backtest multi-series forecasters
- `backtesting_stats` — backtest statistical models
- `backtesting_foundation` — backtest `ForecasterFoundation` (zero-shot)
- `grid_search_forecaster` / `grid_search_forecaster_multiseries` / `grid_search_stats`
- `random_search_forecaster` / `random_search_forecaster_multiseries` / `random_search_stats`
- `bayesian_search_forecaster` / `bayesian_search_forecaster_multiseries`
- `bayesian_search_foundation` — tune `ForecasterFoundation` inference-time parameters
- `grid_search_equivalent_date` — tune `ForecasterEquivalentDate` baselines
- `TimeSeriesFold` — multi-step cross-validation
- `OneStepAheadFold` — fast one-step cross-validation

### Feature Selection
- `select_features` — single series
- `select_features_multiseries` — multi-series

### Drift Detection
- `RangeDriftDetector` — lightweight range check
- `PopulationDriftDetector` — statistical tests

### Preprocessing
- `RollingFeatures` — rolling window statistics
- `TimeSeriesDifferentiator` — differencing
- `CalendarFeatures` — calendar features

## References

Full constructor and method signatures for all public skforecast classes and
functions, split by domain:

- **Forecaster constructors**: See [references/forecaster-constructors.md](references/forecaster-constructors.md)
- **Forecaster methods** (`fit()`, `predict()`, `predict_interval()`, `predict_quantiles()`, `predict_dist()`, `set_params()`, `set_lags()`, `set_out_sample_residuals()`, availability matrix): See [references/forecaster-methods.md](references/forecaster-methods.md)
- **Backtesting, search, cross-validation, feature selection**: See [references/model-selection-signatures.md](references/model-selection-signatures.md)
- **Drift detection and preprocessing**: See [references/preprocessing-signatures.md](references/preprocessing-signatures.md)
