---
name: complete-api-reference
description: >
  Complete constructor signatures and method signatures for all skforecast
  forecasters, backtesting functions, search functions, cross-validation
  classes, preprocessing, feature selection, and drift detection.
  Use when the user needs exact parameter names, types, or defaults for
  any skforecast class or function.
---

# Complete API Reference

## When to Use This Skill

Use this when you need exact parameter names, types, defaults, or method signatures for any skforecast class or function.

## Overview

This skill contains the full constructor and method signatures for all
public skforecast classes and functions. See
[references/method-signatures.md](references/method-signatures.md) for
the complete reference, including:

- All forecaster constructors
- `fit()`, `predict()`, `predict_interval()`, `predict_quantiles()`, `predict_dist()` signatures
- `set_params()`, `set_lags()`, `set_out_sample_residuals()` signatures
- Method availability matrix (which forecaster supports which method)
- Backtesting, search, cross-validation, feature selection, and drift detection signatures

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

### Forecaster Methods
- `fit()` — train the model
- `predict()` — generate point forecasts
- `predict_interval()` — generate prediction intervals

### Model Selection
- `backtesting_forecaster` — backtest single-series forecasters
- `backtesting_forecaster_multiseries` — backtest multi-series forecasters
- `backtesting_stats` — backtest statistical models
- `grid_search_forecaster` / `grid_search_forecaster_multiseries` / `grid_search_stats`
- `random_search_forecaster` / `random_search_forecaster_multiseries` / `random_search_stats`
- `bayesian_search_forecaster` / `bayesian_search_forecaster_multiseries`
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
- `DateTimeFeatureTransformer` — calendar features
